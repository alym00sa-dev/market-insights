"""
Reusable AQL query templates for the market_intel knowledge graph.
All functions return plain Python lists/dicts — no ArangoDB objects leak out.
"""
from graph.client import get_db


def get_events_for_player(
    player_key: str,
    limit: int = 20,
    event_type: str = None,
    days: int = None,
) -> list[dict]:
    db = get_db()
    filters = []
    bind_vars = {"player_key": f"ai_players/{player_key}", "limit": limit}

    if event_type:
        filters.append("FILTER e.event_type == @event_type")
        bind_vars["event_type"] = event_type

    if days:
        filters.append(
            "FILTER DATE_DIFF(DATE_ISO8601(e.sources[0].published_date), DATE_NOW(), 'd') <= @days"
        )
        bind_vars["days"] = days

    filter_str = "\n        ".join(filters)

    aql = f"""
    FOR v, edge IN 1..1 OUTBOUND @player_key player_events
        LET e = DOCUMENT(edge._to)
        LET pub_date = e.sources[0].published_date
        {filter_str}
        SORT pub_date DESC
        LIMIT @limit
        RETURN e
    """
    return list(db.aql.execute(aql, bind_vars=bind_vars))


def get_recent_events_all_players(limit: int = 50, days: int = 7) -> list[dict]:
    db = get_db()
    aql = """
    FOR e IN events
        FILTER DATE_DIFF(DATE_ISO8601(e.published_date), DATE_NOW(), 'd') <= @days
        SORT e.published_date DESC
        LIMIT @limit
        RETURN MERGE(e, {
            players: (
                FOR v, edge IN 1..1 INBOUND e._id player_events
                RETURN v.name
            )
        })
    """
    return list(db.aql.execute(aql, bind_vars={"days": days, "limit": limit}))


def get_player_event_counts() -> list[dict]:
    db = get_db()
    aql = """
    FOR p IN ai_players
        LET count = LENGTH(
            FOR v, e IN 1..1 OUTBOUND p._id player_events RETURN 1
        )
        RETURN { player: p.name, key: p._key, count: count }
    """
    return list(db.aql.execute(aql))


def search_events(
    query_terms: list[str],
    player_key: str = None,
    limit: int = 30,
) -> list[dict]:
    db = get_db()
    bind_vars = {"limit": limit}

    player_filter = ""
    if player_key:
        player_filter = """
        LET player_event_ids = (
            FOR v, edge IN 1..1 OUTBOUND CONCAT('ai_players/', @player_key) player_events
            RETURN edge._to
        )
        FILTER e._id IN player_event_ids
        """
        bind_vars["player_key"] = player_key

    term_filters = " AND ".join(
        f"(CONTAINS(LOWER(e.title), LOWER(@term{i})) OR CONTAINS(LOWER(e.description), LOWER(@term{i})))"
        for i, _ in enumerate(query_terms)
    )
    for i, term in enumerate(query_terms):
        bind_vars[f"term{i}"] = term

    aql = f"""
    FOR e IN events
        {player_filter}
        FILTER {term_filters}
        SORT e.sources[0].published_date DESC
        LIMIT @limit
        RETURN e
    """
    return list(db.aql.execute(aql, bind_vars=bind_vars))


def get_knowledge_range() -> dict:
    """Returns the earliest first_seen and latest last_updated across all events."""
    db = get_db()
    aql = """
    LET earliest = MIN(
        FOR e IN events
            FOR s IN e.sources
                FILTER s.published_date != null AND s.published_date != ""
                RETURN s.published_date
    )
    LET latest = MAX(
        FOR e IN events
            FOR s IN e.sources
                FILTER s.published_date != null AND s.published_date != ""
                RETURN s.published_date
    )
    RETURN { earliest: earliest, latest: latest }
    """
    result = list(db.aql.execute(aql))
    return result[0] if result else {"earliest": None, "latest": None}


def get_ingestion_stats() -> dict:
    """Returns total events, events today, and events this week."""
    db = get_db()
    aql = """
    LET total = LENGTH(events)
    LET today = LENGTH(
        FOR e IN events
            FILTER DATE_DIFF(DATE_ISO8601(e.first_seen), DATE_NOW(), 'd') < 1
            RETURN 1
    )
    LET this_week = LENGTH(
        FOR e IN events
            FILTER DATE_DIFF(DATE_ISO8601(e.first_seen), DATE_NOW(), 'd') <= 7
            RETURN 1
    )
    RETURN { total: total, today: today, this_week: this_week }
    """
    result = list(db.aql.execute(aql))
    return result[0] if result else {"total": 0, "today": 0, "this_week": 0}


def get_source_health() -> list[dict]:
    """Returns event counts per source outlet for the dashboard."""
    db = get_db()
    aql = """
    FOR e IN events
        FOR s IN e.sources
            FILTER s.name != null AND s.name != ""
            COLLECT source = s.name
            AGGREGATE count = LENGTH(1), latest = MAX(s.published_date)
            SORT count DESC
            RETURN { source, count, latest }
    """
    return list(db.aql.execute(aql))
