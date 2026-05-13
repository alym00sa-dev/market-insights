"""
Information Signal task ("The Analyst").
Two-layer deduplication:
  Layer 1 — content hash check (fast, no LLM)
  Layer 2 — LLM semantic check against recent events for the same player

If an event is a semantic duplicate, the existing node is updated with the
new source rather than creating a new one (multi-source corroboration model).
"""
import json
import uuid
from datetime import datetime, timezone

from graph.client import run_query
from llm.config import get_llm

DEDUP_SYSTEM_PROMPT = """You are a market intelligence deduplication assistant.

Given a candidate news event and a list of recent events already in the database for the same AI company, determine whether the candidate covers the SAME underlying event as any existing entry.

Same event = same announcement, product, deal, or development — even if reported by different outlets.
Different event = a follow-up, update, reaction, or distinct but related story.

Return JSON only:
{
  "is_duplicate": true | false,
  "duplicate_of": "<event key if is_duplicate, else null>"
}"""

DEDUP_SCHEMA = {
    "type": "object",
    "properties": {
        "is_duplicate": {"type": "boolean"},
        "duplicate_of": {"type": ["string", "null"]},
    },
    "required": ["is_duplicate", "duplicate_of"],
}


def _hash_exists(content_hash: str) -> bool:
    results = run_query(
        "MATCH (e:Event {raw_content_hash: $hash}) RETURN e.key LIMIT 1",
        {"hash": content_hash},
    )
    return len(results) > 0


def _get_recent_events(player_key: str, limit: int = 10) -> list[dict]:
    return run_query(
        """
        MATCH (p:Player {key: $player_key})-[:INVOLVED_IN]->(e:Event)
        RETURN e.key AS _key, e.title AS title,
               e.scraped_content AS scraped_content, e.event_type AS event_type
        ORDER BY e.first_seen DESC
        LIMIT $limit
        """,
        {"player_key": player_key, "limit": limit},
    )


def _check_semantic_duplicate(candidate: dict, recent_events: list[dict]) -> tuple[bool, str | None]:
    if not recent_events:
        return False, None

    llm = get_llm()
    recent_formatted = "\n".join(
        f"- [key: {e['_key']}] {e['title']} ({e.get('event_type', '')})"
        for e in recent_events
    )
    user_prompt = (
        f"Candidate event:\nTitle: {candidate['title']}\nContent: {candidate.get('scraped_content', '')[:400]}\n\n"
        f"Recent events in database:\n{recent_formatted}"
    )

    try:
        result = llm.complete_structured(DEDUP_SYSTEM_PROMPT, user_prompt, DEDUP_SCHEMA)
        return result["is_duplicate"], result.get("duplicate_of")
    except Exception as e:
        print(f"  [Signal] Semantic dedup failed: {e}")
        return False, None


def _append_source(event_key: str, new_source: dict):
    result = run_query(
        "MATCH (e:Event {key: $key}) RETURN e.sources_json AS sources_json",
        {"key": event_key},
    )
    if not result:
        return
    sources = json.loads(result[0].get("sources_json") or "[]")
    sources.append(new_source)
    run_query(
        """
        MATCH (e:Event {key: $key})
        SET e.sources_json = $sources_json,
            e.source_count = e.source_count + 1,
            e.last_updated = $now
        """,
        {
            "key": event_key,
            "sources_json": json.dumps(sources),
            "now": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"  [Signal] Appended source to existing event: {event_key}")


def _insert_event(event: dict) -> str:
    event_key = str(uuid.uuid4())
    player_keys = event.get("player_keys", [])
    sources = event.get("sources", [])

    doc = {k: v for k, v in event.items() if k not in ("player_keys", "sources")}
    doc["key"] = event_key
    doc["sources_json"] = json.dumps(sources)
    doc["published_date"] = sources[0].get("published_date", "") if sources else ""

    run_query("CREATE (e:Event) SET e = $props", {"props": doc})

    for player_key in player_keys:
        run_query(
            """
            MATCH (p:Player {key: $player_key}), (e:Event {key: $event_key})
            CREATE (p)-[:INVOLVED_IN {relationship_type: 'involved_in', relevance_score: 1.0}]->(e)
            """,
            {"player_key": player_key, "event_key": event_key},
        )

    print(f"  [Signal] Inserted new event: {event['title'][:60]}")
    return event_key


def process(event: dict) -> str | None:
    """
    Run the two-layer dedup pipeline for a single extracted event.
    Returns the event key (new or existing), or None if skipped.
    """
    if event.get("significance_score", 0) < 5:
        print(f"  [Signal] Low significance ({event.get('significance_score')}) — skipping: {event['title'][:60]}")
        return None

    if _hash_exists(event["raw_content_hash"]):
        print(f"  [Signal] Hash duplicate — skipping: {event['title'][:60]}")
        return None

    new_source = event["sources"][0]
    for player_key in event.get("player_keys", []):
        recent = _get_recent_events(player_key)
        is_dup, dup_key = _check_semantic_duplicate(event, recent)

        if is_dup and dup_key:
            _append_source(dup_key, new_source)
            return dup_key

    return _insert_event(event)
