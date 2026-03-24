"""
LLM-native web search scraper.
Uses the active LLM provider's built-in web search capability to find
recent articles per player. Complements passive RSS/NewsAPI collection
by doing targeted, intelligent searches.

Claude  — uses the web_search tool via the Anthropic API.
OpenAI  — uses the gpt-4o-search-preview model via Chat Completions.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from config.settings import settings

SOURCES_YAML = Path(__file__).parent.parent / "config" / "sources.yaml"

SEARCH_PROMPT_TEMPLATE = """Search for news articles published in the last 7 days about: {query}

Return a JSON array of the most relevant articles you find. Each item must have:
- "title": exact article headline
- "url": direct URL to the article
- "summary": 1-2 sentence factual description of what the article covers
- "published_date": publication date in ISO 8601 format (e.g. 2026-03-20T00:00:00Z), or null if unknown
- "source_name": name of the outlet (e.g. "TechCrunch", "Reuters")

Return only valid JSON. No markdown, no explanation."""


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


def _extract_json(text: str) -> list[dict]:
    """Extract a JSON array from LLM output, tolerating markdown fences."""
    text = text.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        result = json.loads(text.strip())
        return result if isinstance(result, list) else []
    except json.JSONDecodeError:
        # Try to find a JSON array in the text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
    return []


def _normalize_articles(raw: list[dict]) -> list[dict]:
    articles = []
    for item in raw:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        if not title or not url:
            continue
        articles.append({
            "title": title,
            "url": url,
            "summary": (item.get("summary") or "").strip(),
            "published_date": _parse_date(item.get("published_date")),
            "source_name": (item.get("source_name") or "LLM Web Search").strip(),
            "raw_content_hash": _content_hash(url, title),
        })
    return articles


# ── Claude ──────────────────────────────────────────────────────────────────

def _search_claude(query: str) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    prompt = SEARCH_PROMPT_TEMPLATE.format(query=query)

    try:
        response = client.messages.create(
            model=settings.CLAUDE_MODEL,
            max_tokens=4096,
            tools=[{
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }],
            messages=[{"role": "user", "content": prompt}],
        )

        # Extract the final text block (after web search tool use)
        text = ""
        for block in response.content:
            if block.type == "text":
                text = block.text

        return _normalize_articles(_extract_json(text))

    except Exception as e:
        print(f"  [LLMSearch/Claude] Error for query {query!r}: {e}")
        return []


# ── OpenAI ───────────────────────────────────────────────────────────────────

def _search_openai(query: str) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    prompt = SEARCH_PROMPT_TEMPLATE.format(query=query)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-search-preview",
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        return _normalize_articles(_extract_json(text))

    except Exception as e:
        print(f"  [LLMSearch/OpenAI] Error for query {query!r}: {e}")
        return []


# ── Public API ───────────────────────────────────────────────────────────────

def search(query: str) -> list[dict]:
    """Run a single web search query using the configured LLM provider."""
    provider = settings.LLM_PROVIDER.lower()
    print(f"  [LLMSearch/{provider}] Searching: {query!r}")

    if provider == "claude":
        return _search_claude(query)
    elif provider == "openai":
        return _search_openai(query)
    else:
        print(f"  [LLMSearch] Unknown provider: {provider!r}")
        return []


def fetch_all(player_key: str = None) -> list[dict]:
    """
    Run web searches for all news_queries in sources.yaml.
    If player_key is given, fetch only that player's queries.
    Returns flat list of raw article dicts.
    """
    with open(SOURCES_YAML) as f:
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
            results = search(query)
            for article in results:
                article["player_keys_hint"] = pk  # hint for sourcing agent tagging
            articles.extend(results)
            seen_queries.add(query)

    return articles
