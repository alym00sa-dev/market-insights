"""
Backfill script: migrates existing event nodes to the new schema.
  - Renames `summary` → `scraped_content`
  - Generates a `description` paragraph via LLM for events that don't have one

Run from market_intelligence_agent/ with:
  PYTHONPATH=. python backfill_description.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graph.client import get_db
from llm.config import get_llm

DESCRIBE_SYSTEM_PROMPT = """You are a structured data extractor for a market intelligence system tracking major AI companies.

Given a news article title and raw scraped content, write a description field: a full paragraph in plain English covering what happened, who is involved, the context, and why it matters.

Return JSON only:
{"description": "..."}"""

DESCRIBE_SCHEMA = {
    "type": "object",
    "properties": {"description": {"type": "string"}},
    "required": ["description"],
}


def generate_description(title: str, scraped_content: str) -> str:
    llm = get_llm()
    user_prompt = f"Title: {title}\n\nScraped content: {scraped_content[:800]}"
    try:
        result = llm.complete_structured(DESCRIBE_SYSTEM_PROMPT, user_prompt, DESCRIBE_SCHEMA)
        return result["description"]
    except Exception as e:
        print(f"  [Backfill] LLM failed for '{title[:60]}': {e}")
        return ""


def run():
    db = get_db()
    events_col = db.collection("events")

    # Fetch all events
    all_events = list(db.aql.execute("FOR e IN events RETURN e"))
    print(f"[Backfill] Found {len(all_events)} events to process")

    renamed = 0
    generated = 0
    skipped = 0

    for event in all_events:
        key = event["_key"]
        updates = {}

        # Step 1: rename summary → scraped_content if old field exists
        if "summary" in event and "scraped_content" not in event:
            updates["scraped_content"] = event["summary"]
            updates["summary"] = None  # will be removed below
            renamed += 1

        # Step 2: generate description if missing
        if not event.get("description"):
            scraped = event.get("scraped_content") or event.get("summary", "")
            desc = generate_description(event.get("title", ""), scraped)
            if desc:
                updates["description"] = desc
                generated += 1
                print(f"  [Backfill] Generated description for: {event.get('title', '')[:60]}")
            else:
                skipped += 1

        if updates:
            # Apply updates; remove old summary field if present
            patch = {k: v for k, v in updates.items() if v is not None}
            events_col.update({"_key": key, **patch})
            if updates.get("summary") is None and "summary" in event:
                # Remove the old summary field via AQL
                db.aql.execute(
                    "FOR e IN events FILTER e._key == @key UPDATE e WITH {} IN events OPTIONS { keepNull: false }",
                    bind_vars={"key": key},
                )

    print(f"\n[Backfill] Done.")
    print(f"  summary → scraped_content renames: {renamed}")
    print(f"  descriptions generated: {generated}")
    print(f"  skipped (LLM error):    {skipped}")


if __name__ == "__main__":
    run()
