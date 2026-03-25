"""
Direct web scraper using BeautifulSoup for static pages.
Playwright support stubbed for JS-rendered targets (Phase 2+).
"""
import hashlib
import re
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; MarketIntelAgent/1.0)"}


def _content_hash(url: str, title: str) -> str:
    normalized = f"{url.strip().lower()}|{title.strip().lower()}"
    return hashlib.md5(normalized.encode()).hexdigest()


def scrape_url(url: str, source_name: str) -> dict | None:
    """
    Scrape a single URL and return a raw article dict, or None on failure.
    Extracts title and best-effort body text.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        if resp.status_code >= 400:
            print(f"  [Web] {source_name}: HTTP {resp.status_code}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        title = ""
        if soup.title:
            title = soup.title.string or ""
        if not title:
            h1 = soup.find("h1")
            title = h1.get_text(strip=True) if h1 else ""

        # Extract body text: prefer <article>, fall back to <main>, then <body>
        body_tag = soup.find("article") or soup.find("main") or soup.find("body")
        paragraphs = body_tag.find_all("p") if body_tag else []
        summary = " ".join(p.get_text(strip=True) for p in paragraphs[:5])

        if not title:
            return None

        return {
            "title": title.strip(),
            "url": url,
            "summary": summary[:1000],
            "published_date": datetime.now(timezone.utc).isoformat(),
            "source_name": source_name,
            "raw_content_hash": _content_hash(url, title),
        }

    except Exception as e:
        print(f"  [Web] {source_name}: {e}")
        return None


def scrape_anthropic_news(limit: int = 20) -> list[dict]:
    """
    Scrape https://www.anthropic.com/news — no RSS available.
    Extracts article links, titles, and summaries from the news index page.
    """
    base_url = "https://www.anthropic.com"
    index_url = f"{base_url}/news"

    try:
        resp = requests.get(index_url, headers=HEADERS, timeout=15)
        if resp.status_code >= 400:
            print(f"  [Web] Anthropic News: HTTP {resp.status_code}")
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        links = soup.find_all("a", href=True)
        news_links = [l for l in links if "/news/" in l.get("href", "")]

        articles = []
        seen_urls = set()
        for link in news_links[:limit]:
            href = link["href"]
            full_url = href if href.startswith("http") else f"{base_url}{href}"
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            # Prefer the heading element inside the link for a clean title
            heading = link.find(["h2", "h3", "h4"])
            if heading:
                title = heading.get_text(strip=True)
            else:
                # Fallback: strip category + date prefix from raw text
                raw = link.get_text(strip=True)
                title = re.sub(
                    r"^(Announcements?|Products?|Policy|Research|News)?"
                    r"\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}"
                    r"(Announcements?|Products?|Policy|Research|News)?",
                    "", raw, flags=re.IGNORECASE,
                ).strip()
            if not title:
                continue

            articles.append({
                "title": title,
                "url": full_url,
                "summary": "",
                "published_date": datetime.now(timezone.utc).isoformat(),
                "source_name": "Anthropic News",
                "raw_content_hash": _content_hash(full_url, title),
            })

        print(f"  [Web] Anthropic News: {len(articles)} articles")
        return articles

    except Exception as e:
        print(f"  [Web] Anthropic News: {e}")
        return []


def fetch_web_targets(player_key: str = None) -> list[dict]:
    """
    Scrape all web_targets defined in sources.yaml for the given player(s).
    Placeholder — populate web_targets in sources.yaml to activate.
    """
    from pathlib import Path
    import yaml

    sources_path = Path(__file__).parent.parent / "config" / "sources.yaml"
    with open(sources_path) as f:
        sources = yaml.safe_load(f)

    articles = []
    players_to_fetch = [player_key] if player_key else [
        k for k in sources if k != "general"
    ]

    for pk in players_to_fetch:
        for target in sources.get(pk, {}).get("web_targets", []):
            url = target.get("url")
            name = target.get("name", pk)
            if not url:
                continue
            print(f"  [Web] Scraping ({pk}): {name}")
            result = scrape_url(url, name)
            if result:
                articles.append(result)

    return articles
