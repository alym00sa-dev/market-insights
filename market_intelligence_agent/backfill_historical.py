"""
Historical backfill: Aug 1 2025 → today.

Sources:
  1. Guardian API — all player queries with from-date=2025-08-01, paginated
  2. LLM web search — one targeted search per player × month

Run from market_intelligence_agent/ with:
  PYTHONPATH=. python backfill_historical.py
"""
import sys
import time
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timezone, date

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import settings
from agents.sourcing import _load_players, _tag_players
from agents.extraction import extract
from agents.signal import process

# ── Config ────────────────────────────────────────────────────────────────────

FROM_DATE = "2025-08-01"
TO_DATE   = date.today().isoformat()

PLAYER_KEYS = ["openai", "anthropic", "google", "meta", "microsoft", "emerging"]

PLAYER_SEARCH_TERMS = {
    "openai":    "OpenAI ChatGPT GPT",
    "anthropic": "Anthropic Claude AI",
    "google":    "Google DeepMind Gemini AI",
    "meta":      "Meta AI Llama",
    "microsoft": "Microsoft AI Copilot Azure OpenAI",
    "emerging":  "xAI Grok Mistral AI Cohere Perplexity AI",
}

# Aug 2025 → current month
MONTHS = []
y, m = 2025, 8
while (y, m) <= (date.today().year, date.today().month):
    MONTHS.append((y, m))
    m += 1
    if m > 12:
        m = 1
        y += 1

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash(url: str, title: str) -> str:
    return hashlib.md5(f"{url.strip().lower()}|{title.strip().lower()}".encode()).hexdigest()


def _parse_date(s):
    if s:
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).isoformat()


# ── Guardian API (historical) ─────────────────────────────────────────────────

GUARDIAN_BASE = "https://content.guardianapis.com/search"

def _guardian_query(query: str, from_date: str, to_date: str, page: int = 1) -> dict:
    params = {
        "q": query,
        "from-date": from_date,
        "to-date": to_date,
        "order-by": "newest",
        "page-size": 200,
        "page": page,
        "show-fields": "trailText",
        "api-key": settings.GUARDIAN_API_KEY,
    }
    resp = requests.get(GUARDIAN_BASE, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("response", {})


def fetch_guardian_historical() -> list[dict]:
    if not settings.GUARDIAN_API_KEY:
        print("[Guardian] No API key — skipping")
        return []

    import yaml
    sources = yaml.safe_load((ROOT / "config" / "sources.yaml").read_text())

    articles = []
    seen = set()

    for pk in PLAYER_KEYS:
        for query in sources.get(pk, {}).get("news_queries", []):
            print(f"  [Guardian] {pk} | query: {query!r}")
            page = 1
            while True:
                try:
                    data = _guardian_query(query, FROM_DATE, TO_DATE, page)
                    results = data.get("results", [])
                    total_pages = data.get("pages", 1)

                    for item in results:
                        title = (item.get("webTitle") or "").strip()
                        url = (item.get("webUrl") or "").strip()
                        if not title or not url:
                            continue
                        h = _hash(url, title)
                        if h in seen:
                            continue
                        seen.add(h)
                        articles.append({
                            "title": title,
                            "url": url,
                            "summary": (item.get("fields", {}).get("trailText") or "").strip(),
                            "published_date": _parse_date(item.get("webPublicationDate")),
                            "source_name": "The Guardian",
                            "raw_content_hash": h,
                        })

                    if page >= total_pages or page >= 5:  # cap at 5 pages per query
                        break
                    page += 1
                    time.sleep(0.3)  # be polite to the API

                except Exception as e:
                    print(f"  [Guardian] Error page {page} for {query!r}: {e}")
                    break

    print(f"[Guardian] Fetched {len(articles)} historical articles")
    return articles


# ── LLM Search (month-by-month) ───────────────────────────────────────────────

def fetch_llm_historical() -> list[dict]:
    from scrapers.llm_search import search

    articles = []
    seen = set()

    for pk in PLAYER_KEYS:
        terms = PLAYER_SEARCH_TERMS[pk]
        for year, month in MONTHS:
            month_name = MONTH_NAMES[month]
            query = f"{terms} news announcements {month_name} {year}"
            print(f"  [LLMSearch] {pk} | {month_name} {year}")
            try:
                results = search(query)
                for a in results:
                    h = a.get("raw_content_hash", _hash(a.get("url", ""), a.get("title", "")))
                    if h in seen:
                        continue
                    seen.add(h)
                    articles.append(a)
                time.sleep(1)  # avoid hammering the LLM API
            except Exception as e:
                print(f"  [LLMSearch] Error for {pk} {month_name} {year}: {e}")

    print(f"[LLMSearch] Fetched {len(articles)} historical articles")
    return articles


# ── NYT Archive (historical) ──────────────────────────────────────────────────

def fetch_nyt_archive_historical() -> list[dict]:
    from scrapers.nyt_api import fetch_archive

    articles = []
    seen = set()

    for year, month in MONTHS:
        print(f"  [NYT Archive] Fetching {year}-{month:02d}...")
        try:
            results = fetch_archive(year, month)
            for a in results:
                h = a.get("raw_content_hash", _hash(a.get("url", ""), a.get("title", "")))
                if h in seen:
                    continue
                seen.add(h)
                articles.append(a)
            time.sleep(12)  # Archive API: be conservative, 1 req per 12s
        except Exception as e:
            print(f"  [NYT Archive] Error for {year}-{month:02d}: {e}")

    print(f"[NYT Archive] Fetched {len(articles)} historical AI-relevant articles")
    return articles


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(include_nyt_archive: bool = False):
    players = _load_players()

    print(f"\n=== Historical Backfill: {FROM_DATE} → {TO_DATE} ===\n")

    # Collect
    guardian_articles = fetch_guardian_historical()
    llm_articles = fetch_llm_historical()
    nyt_articles = fetch_nyt_archive_historical() if include_nyt_archive else []
    if not include_nyt_archive:
        print("[Backfill] Skipping NYT Archive (pass include_nyt_archive=True to enable)")
    all_articles = guardian_articles + llm_articles + nyt_articles
    print(f"\n[Backfill] Total raw articles: {len(all_articles)}")

    # Tag
    tagged = []
    for article in all_articles:
        player_keys = _tag_players(article, players)
        if player_keys:
            article["player_keys"] = player_keys
            tagged.append(article)
    print(f"[Backfill] Tagged: {len(tagged)} articles")

    # Extract + Signal
    stats = {"extracted": 0, "inserted": 0, "skipped": 0, "failed": 0}
    for i, article in enumerate(tagged, 1):
        try:
            event = extract(article)
            if event is None:
                stats["failed"] += 1
                continue
            stats["extracted"] += 1

            result = process(event)
            if result is None:
                stats["skipped"] += 1
            else:
                stats["inserted"] += 1

            if i % 10 == 0:
                print(f"  [{i}/{len(tagged)}] extracted={stats['extracted']} inserted={stats['inserted']} skipped={stats['skipped']}")

        except Exception as e:
            stats["failed"] += 1
            print(f"  Error on '{article.get('title', '')[:50]}': {e}")

    print(f"\n=== Done ===")
    print(f"  Extracted:  {stats['extracted']}")
    print(f"  Inserted:   {stats['inserted']}")
    print(f"  Skipped:    {stats['skipped']} (duplicates)")
    print(f"  Failed:     {stats['failed']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nyt-archive", action="store_true", help="Include NYT Archive API (slow, ~12s/month)")
    args = parser.parse_args()
    run(include_nyt_archive=args.nyt_archive)
