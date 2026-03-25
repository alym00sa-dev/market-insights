"""
Content Extraction task ("The QA").
Takes a raw article dict and uses an LLM to produce a structured Event.
"""
from datetime import datetime, timezone

from pydantic import BaseModel

from llm.config import get_llm

SYSTEM_PROMPT = """You are a structured data extractor for a market intelligence system tracking major AI companies.

Given a news article title and summary, extract the following as JSON:
- description: a full paragraph in plain English covering what happened, who is involved, the context, and why it matters — this is the primary record of the event
- event_type: one of [product_launch, partnership, funding, hiring, policy, research, infrastructure, legal, earnings, acquisition, model_benchmark, other] — use model_benchmark for articles about model capability scores, benchmarks, or pricing data
- sentiment: one of [positive, neutral, negative]
- significance_score: integer 1-10 (10 = market-defining, 1 = minor/routine)
- key_entities: list of strings — specific people, products, or organizations mentioned beyond the main AI player
- analyst_notes: one sentence explaining why this event matters competitively

Be factual and grounded in the article content. Return JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "event_type": {"type": "string"},
        "sentiment": {"type": "string"},
        "significance_score": {"type": "integer"},
        "key_entities": {"type": "array", "items": {"type": "string"}},
        "analyst_notes": {"type": "string"},
    },
    "required": ["description", "event_type", "sentiment", "significance_score", "key_entities", "analyst_notes"],
}


class ExtractedEvent(BaseModel):
    description: str
    event_type: str
    sentiment: str
    significance_score: int
    key_entities: list[str]
    analyst_notes: str


def extract(article: dict) -> dict | None:
    """
    Run LLM extraction on a raw article.
    Returns a merged dict ready for the Signal task, or None on failure.
    """
    llm = get_llm()
    user_prompt = f"Title: {article['title']}\n\nSummary: {article.get('summary', '')[:800]}"

    try:
        result = llm.complete_structured(SYSTEM_PROMPT, user_prompt, SCHEMA)
        extracted = ExtractedEvent(**result)
    except Exception as e:
        print(f"  [Extraction] Failed for '{article['title'][:60]}': {e}")
        return None

    now = datetime.now(timezone.utc).isoformat()

    return {
        # From raw article
        "title": article["title"],
        "raw_content_hash": article["raw_content_hash"],
        "player_keys": article["player_keys"],

        # Source tracking (multi-source model)
        "sources": [{
            "name": article["source_name"],
            "url": article["url"],
            "published_date": article.get("published_date", now),
        }],
        "source_count": 1,
        "first_seen": now,
        "last_updated": now,

        # LLM-extracted fields
        "description": extracted.description,
        "event_type": extracted.event_type,
        "sentiment": extracted.sentiment,
        "significance_score": extracted.significance_score,
        "key_entities": extracted.key_entities,
        "analyst_notes": extracted.analyst_notes,

        # Raw scraped content (as-is from the source)
        "scraped_content": article.get("summary", "")[:1000],
    }
