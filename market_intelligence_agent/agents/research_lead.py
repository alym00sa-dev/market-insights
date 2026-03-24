"""
Research Lead ("Team Lead") — one per player.
On receiving a question:
  1. Queries the knowledge graph for relevant events.
  2. Spins up sub-researchers in parallel for different analytical angles.
  3. Synthesizes their findings into a concise cited brief.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

from graph.queries import get_events_for_player, search_events
from agents.sub_researcher import analyze
from llm.config import get_llm

SYNTHESIS_SYSTEM_PROMPT = """You are a senior research analyst specializing in AI market intelligence.

You have received findings from your research team about a specific AI company. Synthesize their findings into a concise, cited brief that directly answers the user's question.

Rules:
- Be factual and evidence-based — only assert what the cited events support.
- If evidence is thin, say so clearly.
- Include all relevant citations.
- Do not speculate beyond what the events show.

Return JSON only:
{
  "player": "player name",
  "brief": "2-3 paragraph synthesis directly answering the question",
  "key_points": ["bullet point 1", "bullet point 2", ...],
  "citations": [
    {"title": "...", "date": "YYYY-MM-DD", "url": "...", "source": "..."}
  ],
  "confidence": "high | medium | low",
  "data_coverage": "assessment of how complete the KG data is for this question"
}"""

SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "player": {"type": "string"},
        "brief": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "url": {"type": "string"},
                    "source": {"type": "string"},
                },
            },
        },
        "confidence": {"type": "string"},
        "data_coverage": {"type": "string"},
    },
    "required": ["player", "brief", "key_points", "citations", "confidence", "data_coverage"],
}

# Sub-researcher focus areas — run in parallel
FOCUS_AREAS = [
    "product launches, model releases, and technical capabilities",
    "partnerships, integrations, business deals, and market positioning",
    "policy, governance, regulation, and public statements",
    "funding, hiring, infrastructure, and organizational changes",
]


def _fetch_events(player_key: str, search_terms: list[str], days: int) -> list[dict]:
    """Pull relevant events from the KG for this player."""
    events = []

    # Broad recent events for this player
    recent = get_events_for_player(player_key, limit=30, days=days)
    events.extend(recent)

    # Search-term-specific events
    if search_terms:
        searched = search_events(search_terms, player_key=player_key, limit=20)
        # Deduplicate by _key
        existing_keys = {e.get("_key") for e in events}
        for e in searched:
            if e.get("_key") not in existing_keys:
                events.append(e)
                existing_keys.add(e.get("_key"))

    return events


def _run_sub_researchers(
    question: str,
    player_name: str,
    events: list[dict],
) -> list[dict]:
    """Run sub-researchers in parallel across focus areas."""
    if not events:
        return []

    # Split events across focus areas (each sub-researcher sees all events
    # but focuses on a different analytical lens)
    results = []
    with ThreadPoolExecutor(max_workers=len(FOCUS_AREAS)) as executor:
        futures = {
            executor.submit(analyze, question, player_name, events, focus): focus
            for focus in FOCUS_AREAS
        }
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"  [ResearchLead] Sub-researcher error: {e}")

    return results


def _synthesize(
    question: str,
    player_name: str,
    sub_results: list[dict],
) -> dict:
    """Synthesize sub-researcher findings into a final brief."""
    llm = get_llm()

    combined_findings = []
    combined_citations = []
    for r in sub_results:
        combined_findings.extend(r.get("findings", []))
        combined_citations.extend(r.get("cited_events", []))

    # Deduplicate citations by URL
    seen_urls = set()
    unique_citations = []
    for c in combined_citations:
        url = c.get("source_url", "")
        if url and url not in seen_urls:
            unique_citations.append(c)
            seen_urls.add(url)

    user_prompt = (
        f"User question: {question}\n\n"
        f"Player: {player_name}\n\n"
        f"Research team findings:\n"
        + "\n".join(f"- {f}" for f in combined_findings)
        + f"\n\nAvailable citations ({len(unique_citations)}):\n"
        + "\n".join(
            f"- [{c.get('date', '')}] {c.get('title', '')} "
            f"({c.get('source_name', '')}) — {c.get('source_url', '')}"
            for c in unique_citations[:20]
        )
    )

    try:
        return llm.complete_structured(SYNTHESIS_SYSTEM_PROMPT, user_prompt, SYNTHESIS_SCHEMA)
    except Exception as e:
        return {
            "player": player_name,
            "brief": f"Synthesis failed: {e}",
            "key_points": [],
            "citations": [],
            "confidence": "low",
            "data_coverage": "unknown",
        }


def research(
    question: str,
    player_key: str,
    player_name: str,
    search_terms: list[str] = None,
    days: int = 90,
) -> dict:
    """
    Full research pipeline for one player.
    Returns a structured brief with citations.
    """
    print(f"  [ResearchLead:{player_key}] Starting research...")

    events = _fetch_events(player_key, search_terms or [], days)
    print(f"  [ResearchLead:{player_key}] Retrieved {len(events)} events from KG")

    if not events:
        return {
            "player": player_name,
            "brief": f"No events found in the knowledge graph for {player_name} matching this query.",
            "key_points": [],
            "citations": [],
            "confidence": "low",
            "data_coverage": "No data available — run the ingestion pipeline first.",
        }

    sub_results = _run_sub_researchers(question, player_name, events)
    brief = _synthesize(question, player_name, sub_results)

    print(f"  [ResearchLead:{player_key}] Brief ready (confidence: {brief.get('confidence')})")
    return brief
