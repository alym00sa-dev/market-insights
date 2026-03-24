"""
NewsAPI.org scraper.
Docs: https://newsapi.org/docs/endpoints/everything
Free tier: 100 requests/day, articles up to 1 month old.
"""
import hashlib
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import requests
import yaml

from config.settings import settings

SOURCES_YAML = Path(__file__).parent.parent / "config" / "sources.yaml"


def _load_sources() -> dict:
    with open(SOURCES_YAML) as f:
        return yaml.safe_load(f)


def _content_hash(url: str, title: str) -> str:
    normalized = f"{url.strip().lower()}|{title.strip().lower()}"
    return hashlib.md5(normalized.encode()).hexdigest()


def _parse_date(date_str: Optional[str]) -> str:
    if date_str:
        try:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return dt.isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).isoformat()


def fetch_for_query(query: str, from_days: int = 7, page_size: int = 20) -> list[dict]:
    """Fetch articles from NewsAPI for a single search query."""
    if not settings.NEWS_API_KEY:
        print("  [NewsAPI] NEWS_API_KEY not set — skipping")
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=from_days)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from": from_date,
        "sortBy": "publishedAt",
        "pageSize": min(page_size, 100),
        "language": "en",
        "apiKey": settings.NEWS_API_KEY,
    }

    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params=params,
            timeout=12,
        )
        if resp.status_code == 401:
            print("  [NewsAPI] Invalid API key")
            return []
        if resp.status_code == 429:
            print("  [NewsAPI] Rate limit hit")
            return []
        if resp.status_code >= 400:
            print(f"  [NewsAPI] HTTP {resp.status_code} for query: {query!r}")
            return []

        data = resp.json()
        articles = []
        for item in data.get("articles", []):
            title = (item.get("title") or "").strip()
            url = (item.get("url") or "").strip()
            if not title or not url or url == "https://removed.com":
                continue
            articles.append({
                "title": title,
                "url": url,
                "summary": (item.get("description") or "").strip(),
                "published_date": _parse_date(item.get("publishedAt")),
                "source_name": item.get("source", {}).get("name", "NewsAPI"),
                "raw_content_hash": _content_hash(url, title),
            })
        return articles

    except Exception as e:
        print(f"  [NewsAPI] Error for query {query!r}: {e}")
        return []


def fetch_all(player_key: str = None) -> list[dict]:
    """
    Fetch from all news_queries in sources.yaml.
    If player_key is given, fetch only that player's queries + general queries.
    """
    sources = _load_sources()
    articles = []
    seen_queries = set()

    # General news_api queries (currently placeholder — skipped if status=placeholder)
    for entry in sources.get("general", {}).get("news_api", []):
        if entry.get("status") == "placeholder":
            continue
        for q in entry.get("queries", []):
            if q not in seen_queries:
                print(f"  [NewsAPI] Query: {q!r}")
                articles.extend(fetch_for_query(q))
                seen_queries.add(q)

    # Per-player queries
    players_to_fetch = [player_key] if player_key else [
        k for k in sources if k != "general"
    ]
    for pk in players_to_fetch:
        for q in sources.get(pk, {}).get("news_queries", []):
            if q not in seen_queries:
                print(f"  [NewsAPI] Query ({pk}): {q!r}")
                articles.extend(fetch_for_query(q))
                seen_queries.add(q)

    return articles
