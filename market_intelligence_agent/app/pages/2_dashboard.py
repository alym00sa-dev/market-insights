import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from graph.queries import (
    get_ingestion_stats,
    get_knowledge_range,
    get_player_event_counts,
    get_source_health,
    get_events_for_player,
)
from scheduler import cron

# Start the background scheduler once per process
cron.start()

st.title("Dashboard")

# ── Sidebar: pipeline controls + source health ────────────────────────────────
with st.sidebar:
    st.header("Market Sourcing Pipeline")

    sched_status = cron.get_status()

    # Status + trigger
    status_label = "Running..." if sched_status["is_running"] else "Idle"
    st.caption(f"Status: **{status_label}**")

    if sched_status["last_run_time"]:
        st.caption(f"Last pull: {sched_status['last_run_time'][:19].replace('T', ' ')}")
    else:
        st.caption("Last pull: Never")

    if sched_status["is_running"]:
        st.button("Running...", disabled=True, use_container_width=True)
    else:
        if st.button("Run Pipeline Now", type="primary", use_container_width=True):
            cron.trigger_now()
            st.success("Pipeline triggered.")
            st.rerun()

    # Last run stats
    if sched_status["last_run_stats"] and not sched_status["is_running"]:
        s = sched_status["last_run_stats"]
        if "error" not in s:
            with st.expander("Last run"):
                st.metric("Articles collected", s.get("raw_articles", 0))
                st.metric("New events", s.get("inserted", 0))
                st.metric("Skipped (dups)", s.get("hash_skipped", 0))
                st.metric("Duration (s)", s.get("duration_seconds", 0))

    st.divider()

    # Source health
    st.header("Sources")
    try:
        source_health = get_source_health()
        if source_health:
            for s in source_health:
                count = s.get("count", 0)
                source = s.get("source") or "Unknown"
                latest = (s.get("latest") or "")[:10]
                label = f"**{source}** — {count}"
                if latest:
                    label += f" · {latest}"
                st.caption(label)
        else:
            st.caption("No source data yet.")
    except Exception:
        st.caption("Could not load source data.")

# ── Top metrics ────────────────────────────────────────────────────────────────
try:
    stats = get_ingestion_stats()
    krange = get_knowledge_range()
    count_data = get_player_event_counts()
    count_map = {row["key"]: row["count"] for row in count_data}
except Exception as e:
    st.error(f"Could not connect to knowledge graph: {e}")
    st.stop()

earliest = krange["earliest"][:10] if krange.get("earliest") else "—"
latest = krange["latest"][:10] if krange.get("latest") else "—"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Knowledge Start", earliest)
col2.metric("Last Updated", latest)
col3.metric("Total Events", stats["total"])
col4.metric("This Week", stats["this_week"])
col5.metric("Today", stats["today"])

st.divider()

# ── Per-player event feed ─────────────────────────────────────────────────────
PLAYERS = [
    ("openai", "OpenAI"),
    ("anthropic", "Anthropic"),
    ("google", "Google / DeepMind"),
    ("meta", "Meta AI"),
    ("microsoft", "Microsoft"),
    ("emerging", "Emerging Players"),
]

tab_labels = [f"{name} ({count_map.get(key, 0)})" for key, name in PLAYERS]
tabs = st.tabs(tab_labels)

SENTIMENT_COLOR = {"positive": "green", "negative": "red", "neutral": "gray"}

for tab, (player_key, player_name) in zip(tabs, PLAYERS):
    with tab:
        try:
            events = get_events_for_player(player_key, limit=15, days=90)
        except Exception as e:
            st.error(f"Could not load events: {e}")
            continue

        if not events:
            st.caption(f"No events found for {player_name} in the last 90 days.")
            continue

        for event in events:
            sources = event.get("sources", [])
            date = sources[0].get("published_date", "")[:10] if sources else ""
            source_name = sources[0].get("name", "") if sources else ""
            url = sources[0].get("url", "") if sources else ""
            sentiment = event.get("sentiment", "neutral")
            sig = event.get("significance_score", "?")
            event_type = event.get("event_type", "other").replace("_", " ").title()

            with st.container(border=True):
                header_col, badge_col = st.columns([5, 1])
                with header_col:
                    if url:
                        st.markdown(f"**[{event.get('title', 'Untitled')}]({url})**")
                    else:
                        st.markdown(f"**{event.get('title', 'Untitled')}**")
                    st.caption(f"{event_type} · {source_name} · {date} · Significance: {sig}/10 · :{SENTIMENT_COLOR.get(sentiment, 'gray')}[{sentiment}]")
                with badge_col:
                    source_count = event.get("source_count", 1)
                    if source_count > 1:
                        st.metric("Sources", source_count)

                description = event.get("description", "")
                if description:
                    st.write(description)

                analyst_notes = event.get("analyst_notes", "")
                if analyst_notes:
                    st.caption(f"**Analyst:** {analyst_notes}")
