"""
Market Agent Query Manager ("Portfolio Coordinator").
Handles all user-facing queries from the Streamlit chat interface:
  1. Decomposes the question — identifies players, time frame, search terms.
  2. Routes to Research Lead(s) in parallel.
  3. Assembles a coherent, cited final response.
"""
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from agents.research_lead import research
from llm.config import get_llm
from scrapers.llm_search import search_for_context

PLAYERS_YAML = Path(__file__).parent.parent / "config" / "players.yaml"

DECOMPOSE_SYSTEM_PROMPT = """You are a market intelligence coordinator routing questions about AI companies.

Given a user question, decompose it into a structured routing plan.

Player keys available: openai, anthropic, google, meta, microsoft, nvidia, emerging

Return JSON only:
{
  "relevant_players": ["player_key1", "player_key2"],
  "time_frame_days": 90,
  "search_terms": ["keyword1", "keyword2"],
  "query_type": "single_player | comparison | timeline | general",
  "reasoning": "brief explanation of routing decision"
}

Guidelines:
- If the question names a specific company, include only that player.
- If the question asks to compare named companies, include only those players.
- If the question is general or asks about "any", "all", or "the hyperscalers" without naming specific ones — include ALL 5 main players: openai, anthropic, google, meta, microsoft. Do not guess which subset is most likely based on your training data. The knowledge graph may have data you don't expect.
- search_terms should be 2-5 specific keywords relevant to the question.
- time_frame_days: 7 for "this week", 30 for "this month", 90 for default, 365 for "this year"."""

DECOMPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "relevant_players": {"type": "array", "items": {"type": "string"}},
        "time_frame_days": {"type": "integer"},
        "search_terms": {"type": "array", "items": {"type": "string"}},
        "query_type": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["relevant_players", "time_frame_days", "search_terms", "query_type", "reasoning"],
}

ASSEMBLE_SYSTEM_PROMPT = """You are a market intelligence briefing service. Your job is to report what happened — clearly, accurately, and without opinion.

The five frontier AI hyperscalers tracked in this system are: OpenAI, Anthropic, Google/DeepMind, Meta AI, and Microsoft. There is also an Emerging Players category covering AI companies that are significant but not one of these five (e.g. xAI, Mistral, Cohere, Perplexity). Do not qualify or question the status of any of the five hyperscalers.

Guidelines:
- Write in plain prose paragraphs. No headers, no bullets, no numbered sections, no bold, no formatting of any kind.
- Report the facts. State what happened, who was involved, and how events connect. Do not editorialize or offer personal views.
- Cite sources inline at the end of the sentence they support: "Anthropic acquired Vercept in March 2026 ([The Guardian, 2026-03-23](URL))."
- Always put a space before the opening parenthesis. Never attach a citation directly to a word or number.
- If data is limited, state that plainly and move on.
- Do not use phrases like "I think", "what this really means", "it's worth noting", or any editorial framing. Just report."""


def _load_players() -> dict[str, str]:
    """Returns {player_key: player_name}."""
    with open(PLAYERS_YAML) as f:
        data = yaml.safe_load(f)
    return {p["key"]: p["name"] for p in data["players"]}


def _decompose(question: str) -> dict:
    """Decompose the user question into a routing plan."""
    llm = get_llm()
    try:
        result = llm.complete_structured(
            DECOMPOSE_SYSTEM_PROMPT,
            f"User question: {question}",
            DECOMPOSE_SCHEMA,
        )
        print(f"[QueryManager] Routing to: {result['relevant_players']} | type: {result['query_type']}")
        print(f"[QueryManager] Search terms: {result['search_terms']}")
        return result
    except Exception as e:
        print(f"[QueryManager] Decomposition failed: {e} — falling back to all players")
        return {
            "relevant_players": ["openai", "anthropic", "google", "meta", "microsoft"],
            "time_frame_days": 90,
            "search_terms": [],
            "query_type": "general",
            "reasoning": "fallback",
        }


def _assemble(question: str, briefs: list[dict], history: list[dict] = None, web_context: str = "") -> str:
    """Assemble research briefs into a final markdown response."""
    llm = get_llm()

    briefs_text = ""
    for b in briefs:
        briefs_text += f"\n\n## {b.get('player', 'Unknown')} (confidence: {b.get('confidence', '?')})\n"
        briefs_text += f"{b.get('brief', '')}\n"
        if b.get("key_points"):
            briefs_text += "\nKey points:\n" + "\n".join(f"- {p}" for p in b["key_points"])
        if b.get("citations"):
            briefs_text += "\n\nCitations:\n"
            for c in b["citations"]:
                briefs_text += f"- [{c.get('source', c.get('title', '?'))}, {c.get('date', '')}]({c.get('url', '')})\n"

    history_text = ""
    if history:
        history_text = "Conversation so far:\n"
        for msg in history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_text += f"{role}: {msg['content'][:400]}\n"
        history_text += "\n"

    web_section = f"\n\nLive web search results:\n{web_context}" if web_context else ""
    user_prompt = f"{history_text}User question: {question}\n\nResearch briefs:{briefs_text}{web_section}"

    try:
        return llm.complete(ASSEMBLE_SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        return f"Assembly failed ({e}). Raw findings:\n{briefs_text}"


def _parse_result_limit(question: str, default: int = 10) -> int:
    """Extract a requested result count from the query (1-99). Defaults to 10."""
    match = re.search(r'\b([1-9][0-9]?)\b', question)
    if match:
        return int(match.group(1))
    return default


def _retrieve(question: str, relevant_players: list[str], player_names: dict, search_terms: list[str], days: int, limit: int = 10) -> str:
    """
    Retriever mode: fetch matching events from the KG and return a formatted
    article list. No LLM synthesis — just raw sources ranked by significance.
    """
    from graph.queries import get_events_for_player, search_events

    fetch_per_player = max(limit * 2, 30)
    all_events = []
    seen_keys = set()

    for player_key in relevant_players:
        events = get_events_for_player(player_key, limit=fetch_per_player, days=days)
        if search_terms:
            searched = search_events(search_terms, player_key=player_key, limit=fetch_per_player)
            events_by_key = {e.get("_key"): e for e in events}
            for e in searched:
                if e.get("_key") not in events_by_key:
                    events.append(e)

        for e in events:
            key = e.get("_key")
            if key and key not in seen_keys:
                e["_player_name"] = player_names.get(player_key, player_key)
                all_events.append(e)
                seen_keys.add(key)

    if not all_events:
        return "No matching articles found in the knowledge graph for this query."

    # Sort by date desc, then significance desc as tiebreaker
    all_events.sort(key=lambda e: (e.get("published_date") or "", e.get("significance_score") or 0), reverse=True)
    all_events = all_events[:limit]

    lines = [f"**Top {len(all_events)} articles** for: _{question}_\n"]
    for i, e in enumerate(all_events, 1):
        sources = e.get("sources", [])
        url = sources[0].get("url", "") if sources else ""
        source_name = sources[0].get("name", "") if sources else ""
        date = (e.get("published_date") or "")[:10]
        sig = e.get("significance_score", "?")
        event_type = (e.get("event_type") or "other").replace("_", " ").title()
        player = e.get("_player_name", "")
        title = e.get("title", "Untitled")

        title_md = f"[{title}]({url})" if url else title
        meta = " · ".join(filter(None, [player, event_type, source_name, date, f"Significance: {sig}/10"]))
        lines.append(f"{i}. {title_md}  \n   _{meta}_")

    return "\n".join(lines)


def ask(question: str, history: list[dict] = None, days_filter: int = None, mode: str = "synthesizer") -> str:
    """
    Main entry point for the chat interface.
    Takes a natural language question and returns a markdown-formatted answer.
    history: list of {"role": "user"|"assistant", "content": str} from prior turns.
    days_filter: if provided, overrides the LLM-inferred time frame (None = all time).
    mode: "synthesizer" (default) or "retriever" (raw article list, no LLM synthesis).
    """
    players = _load_players()

    ALL_PLAYERS = ["openai", "anthropic", "google", "meta", "microsoft", "nvidia"]

    # Step 1: Decompose
    plan = _decompose(question)
    relevant_players = [
        pk for pk in plan["relevant_players"] if pk in players
    ]

    # If no specific company is named in the question, always route to all 5.
    # Don't let the LLM guess a subset based on its priors.
    q_lower = question.lower()
    named = [pk for pk in ALL_PLAYERS if pk in q_lower
             or players.get(pk, "").lower() in q_lower]
    if not named:
        relevant_players = ALL_PLAYERS

    if not relevant_players:
        relevant_players = ALL_PLAYERS

    search_terms = plan.get("search_terms", [])
    # Sidebar filter overrides LLM-inferred time frame when explicitly set
    days = days_filter if days_filter is not None else plan.get("time_frame_days", 90)

    # Retriever mode: skip synthesis, return raw article list
    if mode == "retriever":
        limit = _parse_result_limit(question, default=10)
        print(f"[QueryManager] Retriever mode — fetching top {limit} articles for {relevant_players}")
        return _retrieve(question, relevant_players, players, search_terms, days, limit=limit)

    # Step 2: Route to Research Leads + live web search in parallel
    briefs = []
    web_context = ""
    print(f"[QueryManager] Dispatching to {len(relevant_players)} research lead(s) + web search...")

    web_query = question if not search_terms else f"{question} ({', '.join(search_terms[:3])})"

    with ThreadPoolExecutor(max_workers=len(relevant_players) + 1) as executor:
        web_future = executor.submit(search_for_context, web_query)

        kg_futures = {
            executor.submit(
                research,
                question,
                player_key,
                players[player_key],
                search_terms,
                days,
            ): player_key
            for player_key in relevant_players
        }
        for future in as_completed(kg_futures):
            player_key = kg_futures[future]
            try:
                briefs.append(future.result())
            except Exception as e:
                print(f"[QueryManager] ResearchLead error for {player_key}: {e}")

        try:
            web_context = web_future.result()
            print(f"[QueryManager] Web search returned {len(web_context)} chars")
        except Exception as e:
            print(f"[QueryManager] Web search error: {e}")

    # Step 3: Assemble
    print(f"[QueryManager] Assembling response from {len(briefs)} brief(s)...")
    return _assemble(question, briefs, history=history, web_context=web_context)
