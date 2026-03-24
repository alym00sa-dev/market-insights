"""
The Guardian API scraper.
Docs: https://open-platform.theguardian.com/documentation/
Free tier: 500 requests/day, full content available.
Strong AI/tech coverage with reliable metadata.
"""
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests

from config.settings import settings

BASE_URL = "https://content.guardianapis.com/search"


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
    """Fetch articles from The Guardian API for a single search query."""
    if not settings.GUARDIAN_API_KEY:
        print("  [Guardian] GUARDIAN_API_KEY not set — skipping")
        return []

    from_date = (datetime.now(timezone.utc) - timedelta(days=from_days)).strftime("%Y-%m-%d")

    params = {
        "q": query,
        "from-date": from_date,
        "order-by": "newest",
        "page-size": min(page_size, 200),
        "show-fields": "trailText",
        "api-key": settings.GUARDIAN_API_KEY,
    }

    try:
        resp = requests.get(BASE_URL, params=params, timeout=12)
        if resp.status_code == 401:
            print("  [Guardian] Invalid API key")
            return []
        if resp.status_code == 429:
            print("  [Guardian] Rate limit hit")
            return []
        if resp.status_code >= 400:
            print(f"  [Guardian] HTTP {resp.status_code} for query: {query!r}")
            return []

        data = resp.json()
        results = data.get("response", {}).get("results", [])

        articles = []
        for item in results:
            title = (item.get("webTitle") or "").strip()
            url = (item.get("webUrl") or "").strip()
            if not title or not url:
                continue
            summary = (item.get("fields", {}).get("trailText") or "").strip()
            articles.append({
                "title": title,
                "url": url,
                "summary": summary,
                "published_date": _parse_date(item.get("webPublicationDate")),
                "source_name": "The Guardian",
                "raw_content_hash": _content_hash(url, title),
            })
        return articles

    except Exception as e:
        print(f"  [Guardian] Error for query {query!r}: {e}")
        return []


def fetch_all(player_key: str = None) -> list[dict]:
    """
    Fetch from all news_queries in sources.yaml using The Guardian API.
    If player_key is given, fetch only that player's queries.
    """
    from pathlib import Path
    import yaml

    sources_path = Path(__file__).parent.parent / "config" / "sources.yaml"
    with open(sources_path) as f:
        sources = yaml.safe_load(f)

    articles = []
    seen_queries: set[str] = set()

    players_to_fetch = [player_key] if player_key else [
        k for k in sources if k != "general"
    ]

    for pk in players_to_fetch:
        for query in sources.get(pk, {}).get("news_queries", []):
            if query in seen_queries:
                continue
            print(f"  [Guardian] Query ({pk}): {query!r}")
            articles.extend(fetch_for_query(query))
            seen_queries.add(query)

    return articles
