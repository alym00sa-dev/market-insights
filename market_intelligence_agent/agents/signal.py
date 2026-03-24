"""
Information Signal task ("The Analyst").
Two-layer deduplication:
  Layer 1 — content hash check (fast, no LLM)
  Layer 2 — LLM semantic check against recent events for the same player

If an event is a semantic duplicate, the existing node is updated with the
new source rather than creating a new one (multi-source corroboration model).
"""
from datetime import datetime, timezone

from graph.client import get_db
from llm.config import get_llm

DEDUP_SYSTEM_PROMPT = """You are a market intelligence deduplication assistant.

Given a candidate news event and a list of recent events already in the database for the same AI company, determine whether the candidate covers the SAME underlying event as any existing entry.

Same event = same announcement, product, deal, or development — even if reported by different outlets.
Different event = a follow-up, update, reaction, or distinct but related story.

Return JSON only:
{
  "is_duplicate": true | false,
  "duplicate_of": "<event _key if is_duplicate, else null>"
}"""

DEDUP_SCHEMA = {
    "type": "object",
    "properties": {
        "is_duplicate": {"type": "boolean"},
        "duplicate_of": {"type": ["string", "null"]},
    },
    "required": ["is_duplicate", "duplicate_of"],
}


def _hash_exists(db, content_hash: str) -> bool:
    """Layer 1: check if this exact content hash already exists."""
    aql = "FOR e IN events FILTER e.raw_content_hash == @hash LIMIT 1 RETURN e._key"
    cursor = db.aql.execute(aql, bind_vars={"hash": content_hash})
    return len(list(cursor)) > 0


def _get_recent_events(db, player_key: str, limit: int = 10) -> list[dict]:
    """Fetch recent events for a player to compare against."""
    aql = """
    FOR v, edge IN 1..1 OUTBOUND @player_key player_events
        LET e = DOCUMENT(edge._to)
        SORT e.first_seen DESC
        LIMIT @limit
        RETURN { _key: e._key, title: e.title, scraped_content: e.scraped_content, event_type: e.event_type }
    """
    cursor = db.aql.execute(
        aql,
        bind_vars={"player_key": f"ai_players/{player_key}", "limit": limit},
    )
    return list(cursor)


def _check_semantic_duplicate(candidate: dict, recent_events: list[dict]) -> tuple[bool, str | None]:
    """Layer 2: LLM semantic duplicate check."""
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


def _append_source(db, event_key: str, new_source: dict):
    """Append a new source to an existing event node."""
    now = datetime.now(timezone.utc).isoformat()
    aql = """
    FOR e IN events
        FILTER e._key == @key
        UPDATE e WITH {
            sources: APPEND(e.sources, [@source]),
            source_count: e.source_count + 1,
            last_updated: @now
        } IN events
    """
    db.aql.execute(aql, bind_vars={"key": event_key, "source": new_source, "now": now})
    print(f"  [Signal] Appended source to existing event: {event_key}")


def _insert_event(db, event: dict) -> str:
    """Insert a new event node and create edges to its player(s)."""
    events_col = db.collection("events")
    player_events_col = db.collection("player_events")

    doc = {k: v for k, v in event.items() if k != "player_keys"}
    meta = events_col.insert(doc)
    event_key = meta["_key"]

    for player_key in event.get("player_keys", []):
        player_events_col.insert({
            "_from": f"ai_players/{player_key}",
            "_to": f"events/{event_key}",
            "relationship_type": "involved_in",
            "relevance_score": 1.0,
        })

    print(f"  [Signal] Inserted new event: {event['title'][:60]}")
    return event_key


def process(event: dict) -> str | None:
    """
    Run the two-layer dedup pipeline for a single extracted event.
    Returns the event _key (new or existing), or None if skipped.
    """
    db = get_db()

    # Layer 1: hash check
    if _hash_exists(db, event["raw_content_hash"]):
        print(f"  [Signal] Hash duplicate — skipping: {event['title'][:60]}")
        return None

    # Layer 2: semantic check (per player)
    new_source = event["sources"][0]
    for player_key in event.get("player_keys", []):
        recent = _get_recent_events(db, player_key)
        is_dup, dup_key = _check_semantic_duplicate(event, recent)

        if is_dup and dup_key:
            _append_source(db, dup_key, new_source)
            return dup_key

    # New unique event — insert
    return _insert_event(db, event)
