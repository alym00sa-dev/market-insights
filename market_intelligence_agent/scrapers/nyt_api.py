"""
New York Times API scraper.
Covers four APIs:
  - Article Search: keyword search with date range (scheduled runs + backfill)
  - Times Wire: real-time feed of recent publishes (scheduled runs)
  - RSS Feeds: handled by the RSS scraper via sources.yaml entries
  - Archive: full month dump for historical backfill (called from backfill scripts only)

Rate limits (free tier): 10 requests/minute, 4,000 requests/day.
Docs: https://developer.nytimes.com/
"""
import hashlib
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

from config.settings import settings

ARTICLE_SEARCH_URL = "https://api.nytimes.com/svc/search/v2/articlesearch.json"
TIMES_WIRE_URL = "https://api.nytimes.com/svc/news/v3/content/all/{section}.json"
ARCHIVE_URL = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"

# Sections to pull from Times Wire for real-time coverage
WIRE_SECTIONS = ["technology", "business"]

# Keywords used to filter Archive API results (which return all articles)
ARCHIVE_FILTER_KEYWORDS = [
    "artificial intelligence", "openai", "anthropic", "google deepmind",
    "meta ai", "microsoft ai", "chatgpt", "claude ai", "gemini", "llama",
    "large language model", "llm", "generative ai", "copilot", "grok",
    "mistral", "cohere", "perplexity", "xai", "agi",
]


def _content_hash(url: str, title: str) -> str:
    normalized = f"{url.strip().lower()}|{title.strip().lower()}"
    return hashlib.md5(normalized.encode()).hexdigest()


def _parse_date(date_str: Optional[str]) -> str:
    if date_str:
        try:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00")).isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).isoformat()


def _article_to_dict(doc: dict) -> Optional[dict]:
    """Normalize a NYT article doc (Article Search / Archive format) to our schema."""
    title = (doc.get("headline", {}).get("main") or "").strip()
    url = (doc.get("web_url") or "").strip()
    if not title or not url:
        return None
    summary = (doc.get("abstract") or doc.get("lead_paragraph") or "").strip()
    return {
        "title": title,
        "url": url,
        "summary": summary,
        "published_date": _parse_date(doc.get("pub_date")),
        "source_name": "The New York Times",
        "raw_content_hash": _content_hash(url, title),
    }


# ── Article Search API ────────────────────────────────────────────────────────

def fetch_article_search(query: str, from_days: int = 7, page_size: int = 10) -> list[dict]:
    """Search NYT articles for a keyword query over the last `from_days` days."""
    if not settings.NYT_API_KEY:
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=from_days)).strftime("%Y%m%d")

    params = {
        "q": query,
        "begin_date": from_date,
        "sort": "newest",
        "page": 0,
        "api-key": settings.NYT_API_KEY,
    }

    try:
        resp = requests.get(ARTICLE_SEARCH_URL, params=params, timeout=12)
        if resp.status_code == 429:
            print("  [NYT Article Search] Rate limit hit — backing off")
            time.sleep(12)
            return []
        if resp.status_code >= 400:
            print(f"  [NYT Article Search] HTTP {resp.status_code} for query: {query!r}")
            return []

        docs = resp.json().get("response", {}).get("docs", [])
        articles = []
        for doc in docs[:page_size]:
            a = _article_to_dict(doc)
            if a:
                articles.append(a)
        return articles

    except Exception as e:
        print(f"  [NYT Article Search] Error for query {query!r}: {e}")
        return []


# ── Times Wire API ────────────────────────────────────────────────────────────

def fetch_times_wire(section: str = "technology", limit: int = 50) -> list[dict]:
    """Fetch recent NYT article publishes from the Times Wire for a section."""
    if not settings.NYT_API_KEY:
        return []

    url = TIMES_WIRE_URL.format(section=section)
    params = {
        "limit": min(limit, 500),
        "api-key": settings.NYT_API_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=12)
        if resp.status_code == 429:
            print(f"  [NYT Times Wire] Rate limit hit on section={section}")
            return []
        if resp.status_code >= 400:
            print(f"  [NYT Times Wire] HTTP {resp.status_code} for section={section}")
            return []

        results = resp.json().get("results", [])
        articles = []
        for item in results:
            title = (item.get("title") or "").strip()
            url_str = (item.get("url") or "").strip()
            if not title or not url_str:
                continue
            summary = (item.get("abstract") or "").strip()
            articles.append({
                "title": title,
                "url": url_str,
                "summary": summary,
                "published_date": _parse_date(item.get("published_date")),
                "source_name": "The New York Times",
                "raw_content_hash": _content_hash(url_str, title),
            })
        return articles

    except Exception as e:
        print(f"  [NYT Times Wire] Error for section={section}: {e}")
        return []


# ── Archive API (backfill only) ───────────────────────────────────────────────

def fetch_archive(year: int, month: int) -> list[dict]:
    """
    Fetch all NYT articles for a given month via the Archive API.
    Filters to AI-relevant articles by keyword matching.
    Intended for use in backfill scripts only — returns potentially thousands of docs.
    """
    if not settings.NYT_API_KEY:
        return []

    url = ARCHIVE_URL.format(year=year, month=month)
    params = {"api-key": settings.NYT_API_KEY}

    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            print(f"  [NYT Archive] Rate limit hit for {year}-{month:02d}")
            return []
        if resp.status_code >= 400:
            print(f"  [NYT Archive] HTTP {resp.status_code} for {year}-{month:02d}")
            return []

        docs = resp.json().get("response", {}).get("docs", [])
        articles = []
        for doc in docs:
            a = _article_to_dict(doc)
            if not a:
                continue
            combined = f"{a['title']} {a['summary']}".lower()
            if any(kw in combined for kw in ARCHIVE_FILTER_KEYWORDS):
                articles.append(a)

        print(f"  [NYT Archive] {year}-{month:02d}: {len(docs)} total → {len(articles)} AI-relevant")
        return articles

    except Exception as e:
        print(f"  [NYT Archive] Error for {year}-{month:02d}: {e}")
        return []


# ── fetch_all (called by sourcing agent) ─────────────────────────────────────

def fetch_all(player_key: str = None) -> list[dict]:
    """
    Run Article Search (per player queries) + Times Wire (technology + business).
    Called by the sourcing agent during scheduled pipeline runs.
    """
    if not settings.NYT_API_KEY:
        print("  [NYT] NYT_API_KEY not set — skipping")
        return []

    from pathlib import Path
    import yaml

    sources_path = Path(__file__).parent.parent / "config" / "sources.yaml"
    with open(sources_path) as f:
        sources = yaml.safe_load(f)

    articles = []
    seen_hashes: set[str] = set()

    def _add(new_articles):
        for a in new_articles:
            h = a.get("raw_content_hash", "")
            if h and h not in seen_hashes:
                seen_hashes.add(h)
                articles.append(a)

    # Article Search — per player queries
    players_to_search = [player_key] if player_key else [
        k for k in sources if k != "general"
    ]
    seen_queries: set[str] = set()
    for pk in players_to_search:
        for query in sources.get(pk, {}).get("news_queries", []):
            if query in seen_queries:
                continue
            print(f"  [NYT Article Search] ({pk}): {query!r}")
            _add(fetch_article_search(query))
            seen_queries.add(query)
            time.sleep(6)  # 10 req/min limit

    # Times Wire — real-time coverage of tech + business sections
    if not player_key:
        for section in WIRE_SECTIONS:
            print(f"  [NYT Times Wire] section={section}")
            _add(fetch_times_wire(section))
            time.sleep(6)

    return articles
