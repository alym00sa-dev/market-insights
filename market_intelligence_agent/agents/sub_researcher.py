"""
Sub-Researcher ("Strategist").
Spun up by a Research Lead to analyze a specific batch of KG events
and extract findings relevant to the user's question.
Multiple sub-researchers can run in parallel on different event subsets.
"""
from llm.config import get_llm

SYSTEM_PROMPT = """You are a market intelligence researcher analyzing events about an AI company.

Given a user question and a set of events retrieved from our knowledge graph, identify the most relevant events, extract key facts, and compile structured findings with citations.

Return JSON only:
{
  "findings": ["key insight 1", "key insight 2", ...],
  "cited_events": [
    {
      "title": "event title",
      "date": "YYYY-MM-DD",
      "source_url": "url",
      "source_name": "outlet name",
      "relevance_note": "why this event is relevant to the question"
    }
  ],
  "confidence": "high | medium | low",
  "gaps": "what information is missing or unclear from these events"
}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "array", "items": {"type": "string"}},
        "cited_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string"},
                    "source_url": {"type": "string"},
                    "source_name": {"type": "string"},
                    "relevance_note": {"type": "string"},
                },
            },
        },
        "confidence": {"type": "string"},
        "gaps": {"type": "string"},
    },
    "required": ["findings", "cited_events", "confidence", "gaps"],
}


def _format_events(events: list[dict]) -> str:
    if not events:
        return "No events found."
    lines = []
    for e in events:
        date = ""
        sources = e.get("sources", [])
        if sources:
            date = sources[0].get("published_date", "")[:10]
            urls = [s.get("url", "") for s in sources[:2]]
            source_names = [s.get("name", "") for s in sources[:2]]
        else:
            urls = []
            source_names = []

        lines.append(
            f"- [{date}] {e.get('title', '')} "
            f"(type: {e.get('event_type', '')}, "
            f"significance: {e.get('significance_score', '?')}/10, "
            f"sources: {', '.join(source_names) or 'unknown'}, "
            f"url: {urls[0] if urls else 'n/a'})\n"
            f"  Description: {e.get('description', e.get('scraped_content', ''))[:400]}"
        )
    return "\n".join(lines)


def analyze(
    question: str,
    player_name: str,
    events: list[dict],
    focus: str = "",
) -> dict:
    """
    Analyze a set of events and return structured findings.

    Args:
        question: The original user question.
        player_name: Human-readable player name (e.g. "Anthropic").
        events: List of event dicts from the knowledge graph.
        focus: Optional focus area (e.g. "product launches", "partnerships").
    """
    llm = get_llm()
    focus_line = f"\nFocus specifically on: {focus}" if focus else ""

    user_prompt = (
        f"User question: {question}{focus_line}\n\n"
        f"Player: {player_name}\n\n"
        f"Events from knowledge graph ({len(events)} total):\n"
        f"{_format_events(events)}"
    )

    try:
        return llm.complete_structured(SYSTEM_PROMPT, user_prompt, SCHEMA)
    except Exception as e:
        return {
            "findings": [f"Analysis failed: {e}"],
            "cited_events": [],
            "confidence": "low",
            "gaps": "Sub-researcher encountered an error.",
        }
