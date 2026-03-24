"""
Artificial Analysis API scraper.
Fetches LLM model benchmark data and formats it as articles for the ingestion pipeline.

API: https://artificialanalysis.ai/api/v2/data/llms/models
Auth: x-api-key header
Rate limit: 1,000 requests/day (free tier)
Docs: https://artificialanalysis.ai/

Each model is emitted as a single article. The hash (url+title) is stable so
repeated scrapes won't re-insert unchanged models. If a model's name changes
or a new model appears, it gets a fresh event.
"""
import hashlib
from datetime import datetime, timezone
from typing import Optional

import requests

from config.settings import settings

BASE_URL = "https://artificialanalysis.ai/api/v2"
MODELS_ENDPOINT = "/data/llms/models"
MODEL_PAGE_URL = "https://artificialanalysis.ai/models/{slug}"

# Creator slug → our player key
PLAYER_MAP = {
    "openai":    "openai",
    "anthropic": "anthropic",
    "google":    "google",
    "deepmind":  "google",
    "meta":      "meta",
    "facebook":  "meta",
    "microsoft": "microsoft",
}

# Notable emerging players to include (by creator slug prefix)
EMERGING_SLUGS = {
    "xai", "mistral", "cohere", "perplexity", "ai21", "inflection",
    "stability", "amazon", "nvidia", "aleph", "writer", "together",
}

# Max models per player to include (ranked by intelligence index)
MAX_PER_PLAYER = 5
MAX_EMERGING = 3  # per emerging creator


def _content_hash(url: str, title: str) -> str:
    normalized = f"{url.strip().lower()}|{title.strip().lower()}"
    return hashlib.md5(normalized.encode()).hexdigest()


def _parse_date(date_str: Optional[str]) -> str:
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ).isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).isoformat()


def _player_key_for(creator_slug: str) -> Optional[str]:
    slug = (creator_slug or "").lower()
    for key, player in PLAYER_MAP.items():
        if key in slug:
            return player
    for emerging in EMERGING_SLUGS:
        if emerging in slug:
            return "emerging"
    return None


def _format_summary(model: dict) -> str:
    """Build a rich text summary of a model's benchmarks and pricing."""
    evals = model.get("evaluations") or {}
    pricing = model.get("pricing") or {}
    creator = (model.get("model_creator") or {}).get("name", "Unknown")

    lines = [f"{creator} model benchmarked on Artificial Analysis."]

    # Intelligence / capability indices
    ii = evals.get("artificial_analysis_intelligence_index")
    ci = evals.get("artificial_analysis_coding_index")
    mi = evals.get("artificial_analysis_math_index")
    if ii is not None:
        lines.append(f"Intelligence Index: {ii}")
    if ci is not None:
        lines.append(f"Coding Index: {ci}")
    if mi is not None:
        lines.append(f"Math Index: {mi}")

    # Key benchmarks
    bench_map = {
        "mmlu_pro": "MMLU-Pro",
        "gpqa": "GPQA",
        "livecodebench": "LiveCodeBench",
        "aime_25": "AIME 2025",
        "hle": "HLE",
    }
    bench_lines = []
    for field, label in bench_map.items():
        val = evals.get(field)
        if val is not None:
            bench_lines.append(f"{label}: {val}")
    if bench_lines:
        lines.append("Benchmarks: " + ", ".join(bench_lines))

    # Performance
    speed = model.get("median_output_tokens_per_second")
    ttft = model.get("median_time_to_first_token_seconds")
    if speed is not None:
        lines.append(f"Output speed: {speed:.0f} tokens/sec")
    if ttft is not None:
        lines.append(f"Time to first token: {ttft:.2f}s")

    # Pricing
    p_in = pricing.get("price_1m_input_tokens")
    p_out = pricing.get("price_1m_output_tokens")
    if p_in is not None and p_out is not None:
        lines.append(f"Pricing: ${p_in}/1M input tokens, ${p_out}/1M output tokens")

    return " | ".join(lines)


def _model_to_article(model: dict, player_key: str) -> dict:
    name = (model.get("name") or "Unknown").strip()
    slug = (model.get("slug") or name.lower().replace(" ", "-")).strip()
    creator = (model.get("model_creator") or {}).get("name", "Unknown")
    release_date = model.get("release_date")

    title = f"{creator} — {name}: AI benchmark profile"
    url = MODEL_PAGE_URL.format(slug=slug)
    summary = _format_summary(model)

    return {
        "title": title,
        "url": url,
        "summary": summary,
        "published_date": _parse_date(release_date),
        "source_name": "Artificial Analysis",
        "raw_content_hash": _content_hash(url, title),
        "player_keys": [player_key],
    }


def fetch_all(player_key: str = None) -> list[dict]:
    """
    Fetch LLM model benchmark data from Artificial Analysis.
    Returns article-formatted dicts ready for the sourcing/extraction pipeline.
    If player_key is given, only return models for that player.
    """
    if not settings.ARTIFICIAL_ANALYSIS_API_KEY:
        print("  [ArtificialAnalysis] ARTIFICIAL_ANALYSIS_API_KEY not set — skipping")
        return []

    url = f"{BASE_URL}{MODELS_ENDPOINT}"
    headers = {"x-api-key": settings.ARTIFICIAL_ANALYSIS_API_KEY}

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code == 401:
            print("  [ArtificialAnalysis] Invalid API key")
            return []
        if resp.status_code == 429:
            print("  [ArtificialAnalysis] Rate limit hit")
            return []
        if resp.status_code >= 400:
            print(f"  [ArtificialAnalysis] HTTP {resp.status_code}")
            return []

        data = resp.json()
        # Response: { status, prompt_options, data: [...models] }
        models = data.get("data") or []
        if not models and isinstance(data, dict):
            # Fallback: some versions nest differently
            models = data.get("data", {}).get("data", []) if isinstance(data.get("data"), dict) else models

        print(f"  [ArtificialAnalysis] Fetched {len(models)} total models")

    except Exception as e:
        print(f"  [ArtificialAnalysis] Error fetching models: {e}")
        return []

    # Group by player key, ranked by intelligence index
    by_player: dict[str, list] = {}
    by_emerging: dict[str, list] = {}  # slug → models

    for model in models:
        creator = model.get("model_creator") or {}
        slug = (creator.get("slug") or "").lower()
        pk = _player_key_for(slug)
        if pk is None:
            continue

        if pk == "emerging":
            by_emerging.setdefault(slug, []).append(model)
        else:
            by_player.setdefault(pk, []).append(model)

    def _intel(m):
        return (m.get("evaluations") or {}).get("artificial_analysis_intelligence_index") or 0

    articles = []

    # Primary players: top MAX_PER_PLAYER by intelligence index
    for pk, player_models in by_player.items():
        if player_key and pk != player_key:
            continue
        top = sorted(player_models, key=_intel, reverse=True)[:MAX_PER_PLAYER]
        for model in top:
            articles.append(_model_to_article(model, pk))
        print(f"  [ArtificialAnalysis] {pk}: {len(top)} models selected")

    # Emerging: top MAX_EMERGING per creator
    if not player_key or player_key == "emerging":
        for slug, emerging_models in by_emerging.items():
            top = sorted(emerging_models, key=_intel, reverse=True)[:MAX_EMERGING]
            for model in top:
                articles.append(_model_to_article(model, "emerging"))
        emerging_count = sum(min(len(v), MAX_EMERGING) for v in by_emerging.values())
        print(f"  [ArtificialAnalysis] emerging: {emerging_count} models selected")

    print(f"  [ArtificialAnalysis] Total articles: {len(articles)}")
    return articles
