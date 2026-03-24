"""
Sourcing Agent ("The Finder").
Pulls raw articles from all scrapers and tags each article with the player(s)
it is relevant to via alias matching against players.yaml.
"""
from pathlib import Path
from typing import Optional

import yaml

PLAYERS_YAML = Path(__file__).parent.parent / "config" / "players.yaml"


def _load_players() -> list[dict]:
    with open(PLAYERS_YAML) as f:
        return yaml.safe_load(f)["players"]


def _tag_players(article: dict, players: list[dict]) -> list[str]:
    """
    Return a list of player keys this article is primarily about.

    Strategy: title is the primary signal. If any player alias appears in the
    title, tag only those players — a player only mentioned in the summary is a
    reference/comparison, not the subject. Fall back to summary matching only
    if no player matched in the title.

    Catch-all (Emerging) only fires if no primary player matched at all.
    """
    title = article.get("title", "").lower()
    summary = article.get("summary", "").lower()

    primary_players = [p for p in players if not p.get("is_catch_all")]
    catch_all = next((p for p in players if p.get("is_catch_all")), None)

    # Pass 1: title-only matches
    title_matched = []
    for player in primary_players:
        for alias in player.get("aliases", []):
            if alias.lower() in title:
                title_matched.append(player["key"])
                break

    if title_matched:
        return title_matched

    # Pass 2: summary fallback (no player found in title)
    summary_matched = []
    for player in primary_players:
        for alias in player.get("aliases", []):
            if alias.lower() in summary:
                summary_matched.append(player["key"])
                break

    if summary_matched:
        return summary_matched

    # Pass 3: catch-all if nothing else matched
    if catch_all:
        for alias in catch_all.get("aliases", []):
            if alias.lower() in f"{title} {summary}":
                return [catch_all["key"]]

    return []


def collect(player_key: Optional[str] = None) -> list[dict]:
    """
    Run all scrapers, tag each article with relevant player(s), and return
    a flat list of tagged raw articles.

    Each article dict gains a `player_keys` field: list of matched player keys.
    Articles with no player match are dropped.
    """
    from scrapers.rss import fetch_all as rss_fetch
    from scrapers.news_api import fetch_all as newsapi_fetch
    from scrapers.guardian_api import fetch_all as guardian_fetch
    from scrapers.nyt_api import fetch_all as nyt_fetch
    from scrapers.artificial_analysis import fetch_all as aa_fetch
    from scrapers.web import fetch_web_targets, scrape_anthropic_news
    from scrapers.llm_search import fetch_all as llm_search_fetch

    players = _load_players()

    print("[Sourcing] Fetching RSS feeds...")
    rss_articles = rss_fetch(player_key)

    print("[Sourcing] Fetching NewsAPI...")
    newsapi_articles = newsapi_fetch(player_key)

    print("[Sourcing] Fetching The Guardian...")
    guardian_articles = guardian_fetch(player_key)

    print("[Sourcing] Fetching NYT (Article Search + Times Wire)...")
    nyt_articles = nyt_fetch(player_key)

    print("[Sourcing] Fetching Artificial Analysis model benchmarks...")
    aa_articles = aa_fetch(player_key)

    print("[Sourcing] Scraping Anthropic News...")
    anthropic_articles = scrape_anthropic_news() if not player_key or player_key == "anthropic" else []
    for a in anthropic_articles:
        a["player_keys"] = ["anthropic"]

    print("[Sourcing] Fetching web targets...")
    web_articles = fetch_web_targets(player_key)

    print("[Sourcing] Running LLM web search...")
    llm_articles = llm_search_fetch(player_key)

    all_articles = rss_articles + newsapi_articles + guardian_articles + nyt_articles + aa_articles + anthropic_articles + web_articles + llm_articles
    print(f"[Sourcing] Total raw articles: {len(all_articles)}")

    tagged = []
    for article in all_articles:
        player_keys = _tag_players(article, players)
        if player_keys:
            article["player_keys"] = player_keys
            tagged.append(article)

    print(f"[Sourcing] Tagged articles (matched to a player): {len(tagged)}")
    return tagged
