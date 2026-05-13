"""
Cypher query templates for the market_intel knowledge graph.
All functions return plain Python lists/dicts.

Note: Neo4j doesn't support lists-of-maps as node properties, so the
`sources` array (list of {name, url, published_date} dicts) is stored
as a JSON string (`sources_json`) on Event nodes. _deserialize_event()
converts it back to a list before returning, keeping the rest of the
codebase unaware of this detail.
"""
import json
from datetime import datetime, timedelta, timezone

from graph.client import run_query


def _days_ago_iso(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _deserialize_event(event: dict) -> dict:
    if "sources_json" in event:
        event["sources"] = json.loads(event.pop("sources_json") or "[]")
    event["_key"] = event.get("key", "")
    return event


def get_events_for_player(
    player_key: str,
    limit: int = 20,
    event_type: str = None,
    days: int = None,
) -> list[dict]:
    params = {"player_key": player_key, "limit": limit}
    filters = ["e.significance_score >= 5"]

    if event_type:
        filters.append("e.event_type = $event_type")
        params["event_type"] = event_type
    else:
        filters.append("e.event_type <> 'model_benchmark'")

    if days:
        filters.append("e.published_date >= $cutoff")
        params["cutoff"] = _days_ago_iso(days)

    where_clause = " AND ".join(filters)
    cypher = f"""
    MATCH (p:Player {{key: $player_key}})-[:INVOLVED_IN]->(e:Event)
    WHERE {where_clause}
    RETURN e {{.*}} AS event
    ORDER BY e.published_date DESC
    LIMIT $limit
    """
    return [_deserialize_event(r["event"]) for r in run_query(cypher, params)]


def get_recent_events_all_players(limit: int = 50, days: int = 7) -> list[dict]:
    cypher = """
    MATCH (e:Event)
    WHERE e.published_date >= $cutoff
    WITH e, [(p:Player)-[:INVOLVED_IN]->(e) | p.name] AS players
    RETURN e {.*, players: players} AS event
    ORDER BY e.published_date DESC
    LIMIT $limit
    """
    return [
        _deserialize_event(r["event"])
        for r in run_query(cypher, {"cutoff": _days_ago_iso(days), "limit": limit})
    ]


def get_player_event_counts() -> list[dict]:
    cypher = """
    MATCH (p:Player)
    OPTIONAL MATCH (p)-[:INVOLVED_IN]->(e:Event)
    RETURN p.name AS player, p.key AS key, count(e) AS count
    """
    return run_query(cypher)


def search_events(
    query_terms: list[str],
    player_key: str = None,
    limit: int = 30,
) -> list[dict]:
    params = {"terms": [t.lower() for t in query_terms], "limit": limit}

    if player_key:
        cypher = """
        MATCH (p:Player {key: $player_key})-[:INVOLVED_IN]->(e:Event)
        WHERE ALL(term IN $terms WHERE
            toLower(e.title) CONTAINS term OR toLower(e.description) CONTAINS term)
        RETURN e {.*} AS event
        ORDER BY e.published_date DESC
        LIMIT $limit
        """
        params["player_key"] = player_key
    else:
        cypher = """
        MATCH (e:Event)
        WHERE ALL(term IN $terms WHERE
            toLower(e.title) CONTAINS term OR toLower(e.description) CONTAINS term)
        RETURN e {.*} AS event
        ORDER BY e.published_date DESC
        LIMIT $limit
        """

    return [_deserialize_event(r["event"]) for r in run_query(cypher, params)]


def get_knowledge_range() -> dict:
    cypher = """
    MATCH (e:Event)
    WHERE e.published_date IS NOT NULL AND e.published_date <> ''
    RETURN min(e.published_date) AS earliest, max(e.published_date) AS latest
    """
    result = run_query(cypher)
    return result[0] if result else {"earliest": None, "latest": None}


def get_ingestion_stats() -> dict:
    now = datetime.now(timezone.utc)
    cypher = """
    MATCH (e:Event)
    RETURN
        count(e) AS total,
        sum(CASE WHEN e.first_seen >= $today_cutoff THEN 1 ELSE 0 END) AS today,
        sum(CASE WHEN e.first_seen >= $week_cutoff THEN 1 ELSE 0 END) AS this_week
    """
    result = run_query(cypher, {
        "today_cutoff": (now - timedelta(days=1)).isoformat(),
        "week_cutoff": (now - timedelta(days=7)).isoformat(),
    })
    return result[0] if result else {"total": 0, "today": 0, "this_week": 0}


def get_source_health() -> list[dict]:
    rows = run_query("MATCH (e:Event) RETURN e.sources_json AS sources_json")
    counts: dict[str, int] = {}
    latest: dict[str, str] = {}
    for row in rows:
        sources = json.loads(row.get("sources_json") or "[]")
        for s in sources:
            name = s.get("name", "")
            if not name:
                continue
            counts[name] = counts.get(name, 0) + 1
            date = s.get("published_date", "")
            if date and date > latest.get(name, ""):
                latest[name] = date
    return sorted(
        [{"source": k, "count": v, "latest": latest.get(k)} for k, v in counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )
