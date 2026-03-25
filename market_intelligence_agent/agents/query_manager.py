"""
Market Agent Query Manager ("Portfolio Coordinator").
Handles all user-facing queries from the Streamlit chat interface:
  1. Decomposes the question — identifies players, time frame, search terms.
  2. Routes to Research Lead(s) in parallel.
  3. Assembles a coherent, cited final response.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from agents.research_lead import research
from llm.config import get_llm

PLAYERS_YAML = Path(__file__).parent.parent / "config" / "players.yaml"

DECOMPOSE_SYSTEM_PROMPT = """You are a market intelligence coordinator routing questions about AI companies.

Given a user question, decompose it into a structured routing plan.

Player keys available: openai, anthropic, google, meta, microsoft, emerging

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


def _assemble(question: str, briefs: list[dict], history: list[dict] = None) -> str:
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

    user_prompt = f"{history_text}User question: {question}\n\nResearch briefs:{briefs_text}"

    try:
        return llm.complete(ASSEMBLE_SYSTEM_PROMPT, user_prompt)
    except Exception as e:
        return f"Assembly failed ({e}). Raw findings:\n{briefs_text}"


def ask(question: str, history: list[dict] = None, days_filter: int = None) -> str:
    """
    Main entry point for the chat interface.
    Takes a natural language question and returns a markdown-formatted answer.
    history: list of {"role": "user"|"assistant", "content": str} from prior turns.
    days_filter: if provided, overrides the LLM-inferred time frame (None = all time).
    """
    players = _load_players()

    ALL_PLAYERS = ["openai", "anthropic", "google", "meta", "microsoft"]

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

    # Step 2: Route to Research Leads in parallel
    briefs = []
    print(f"[QueryManager] Dispatching to {len(relevant_players)} research lead(s)...")

    with ThreadPoolExecutor(max_workers=len(relevant_players)) as executor:
        futures = {
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
        for future in as_completed(futures):
            player_key = futures[future]
            try:
                briefs.append(future.result())
            except Exception as e:
                print(f"[QueryManager] ResearchLead error for {player_key}: {e}")

    # Step 3: Assemble
    print(f"[QueryManager] Assembling response from {len(briefs)} brief(s)...")
    return _assemble(question, briefs, history=history)
