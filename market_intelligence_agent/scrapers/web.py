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


def _extract_published_date(soup: BeautifulSoup) -> str:
    """Try to find the article's published date from common HTML patterns."""
    # 1. <meta property="article:published_time">
    meta = soup.find("meta", {"property": "article:published_time"})
    if meta and meta.get("content"):
        try:
            return datetime.fromisoformat(meta["content"].replace("Z", "+00:00")).isoformat()
        except Exception:
            pass

    # 2. <time datetime="...">
    time_tag = soup.find("time", {"datetime": True})
    if time_tag:
        try:
            return datetime.fromisoformat(time_tag["datetime"].replace("Z", "+00:00")).isoformat()
        except Exception:
            pass

    # 3. Any <time> tag with parseable text
    time_tag = soup.find("time")
    if time_tag:
        try:
            from dateutil import parser as dateparser
            return dateparser.parse(time_tag.get_text(strip=True)).replace(tzinfo=timezone.utc).isoformat()
        except Exception:
            pass

    return datetime.now(timezone.utc).isoformat()


def scrape_url(url: str, source_name: str) -> dict | None:
    """
    Scrape a single URL and return a raw article dict, or None on failure.
    Extracts title, best-effort body text, and published date.
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
            "published_date": _extract_published_date(soup),
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

            # Try to extract date from raw link text
            raw = link.get_text(strip=True)
            date_str = datetime.now(timezone.utc).isoformat()
            date_match = re.search(
                r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}",
                raw, flags=re.IGNORECASE,
            )
            if date_match:
                try:
                    from dateutil import parser as dateparser
                    date_str = dateparser.parse(date_match.group(0)).replace(tzinfo=timezone.utc).isoformat()
                except Exception:
                    pass

            # Prefer the heading element inside the link for a clean title
            heading = link.find(["h2", "h3", "h4"])
            if heading:
                title = heading.get_text(strip=True)
            else:
                # Strip category + date prefix from raw text to get clean title
                title = re.sub(
                    r"^(Announcements?|Products?|Policy|Research|News)?"
                    r"\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}"
                    r"\s*(Announcements?|Products?|Policy|Research|News)?",
                    "", raw, flags=re.IGNORECASE,
                ).strip()
            if not title:
                continue

            articles.append({
                "title": title,
                "url": full_url,
                "summary": "",
                "published_date": date_str,
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
