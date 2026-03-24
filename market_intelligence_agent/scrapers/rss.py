"""
RSS feed scraper.
Fetches and parses feeds listed under general.rss and per-player rss in sources.yaml.
Returns raw article dicts ready for the Sourcing Agent to tag and route.
"""
import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import feedparser
import requests
import yaml

SOURCES_YAML = Path(__file__).parent.parent / "config" / "sources.yaml"

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MarketIntelAgent/1.0)"}


def _load_sources() -> dict:
    with open(SOURCES_YAML) as f:
        return yaml.safe_load(f)


def _parse_date(entry) -> Optional[str]:
    for field in ("published_parsed", "updated_parsed"):
        t = getattr(entry, field, None)
        if t:
            try:
                dt = datetime(*t[:6], tzinfo=timezone.utc)
                return dt.isoformat()
            except Exception:
                pass
    return datetime.now(timezone.utc).isoformat()


def _content_hash(url: str, title: str) -> str:
    normalized = f"{url.strip().lower()}|{title.strip().lower()}"
    return hashlib.md5(normalized.encode()).hexdigest()


def _fetch_feed(name: str, url: str) -> list[dict]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12, allow_redirects=True)
        if resp.status_code >= 400:
            print(f"  [RSS] {name}: HTTP {resp.status_code}")
            return []

        feed = feedparser.parse(resp.content)
        articles = []
        for entry in feed.entries[:50]:
            title = getattr(entry, "title", "").strip()
            link = getattr(entry, "link", "").strip()
            summary = getattr(entry, "summary", "").strip()
            if not title or not link:
                continue
            articles.append({
                "title": title,
                "url": link,
                "summary": summary,
                "published_date": _parse_date(entry),
                "source_name": name,
                "raw_content_hash": _content_hash(link, title),
            })
        return articles

    except requests.exceptions.Timeout:
        print(f"  [RSS] {name}: timeout")
        return []
    except Exception as e:
        print(f"  [RSS] {name}: {e}")
        return []


def fetch_all(player_key: str = None) -> list[dict]:
    """
    Fetch from all RSS sources.
    If player_key is given, also fetch that player's first-party feeds.
    Returns flat list of raw article dicts.
    """
    sources = _load_sources()
    articles = []

    # General feeds
    for feed in sources.get("general", {}).get("rss", []):
        if feed.get("status") == "blocked":
            continue
        print(f"  [RSS] Fetching: {feed['name']}")
        articles.extend(_fetch_feed(feed["name"], feed["url"]))
        time.sleep(0.3)

    # Per-player first-party feeds
    players_to_fetch = [player_key] if player_key else [
        k for k in sources if k != "general"
    ]
    for pk in players_to_fetch:
        player_sources = sources.get(pk, {})
        for feed in player_sources.get("rss", []):
            if feed.get("status") == "blocked":
                continue
            print(f"  [RSS] Fetching ({pk}): {feed['name']}")
            articles.extend(_fetch_feed(feed["name"], feed["url"]))
            time.sleep(0.3)

    return articles
