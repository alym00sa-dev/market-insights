import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from agents.query_manager import ask
from llm.config import set_active_model

st.title("Market Intelligence Agent")

st.info(
    "**v1 — General AI Market Intelligence.** "
    "This agent tracks publicly available news and model data across the five frontier AI hyperscalers. "
    "It does not yet have access to Gates Foundation-specific context, vertical-specific signals, or proprietary data sources.",
    icon="ℹ️",
)

# Sidebar filters
MODEL_OPTIONS = {
    "Claude Sonnet 4.6": ("claude", "claude-sonnet-4-6"),
    "Claude Opus 4.6": ("claude", "claude-opus-4-6"),
    "Claude Haiku 4.5": ("claude", "claude-haiku-4-5-20251001"),
    "GPT-5.4": ("openai", "gpt-5.4"),
    "GPT-5.4 mini": ("openai", "gpt-5.4-mini"),
    "GPT-4o": ("openai", "gpt-4o"),
    "GPT-4o mini": ("openai", "gpt-4o-mini"),
}

DATE_OPTIONS = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 60 days": 60,
    "Last 90 days": 90,
    "All time": None,
}

with st.sidebar:
    st.header("Filters")

    # Model selector
    model_choice = st.selectbox("Model", list(MODEL_OPTIONS.keys()), index=0)
    selected_provider, selected_model = MODEL_OPTIONS[model_choice]
    set_active_model(selected_provider, selected_model)

    # Date range filter
    selected_range = st.selectbox("Time Frame", list(DATE_OPTIONS.keys()), index=3)
    days_filter = DATE_OPTIONS[selected_range]

    st.divider()

    # URL Analysis toggle
    if "url_analysis_enabled" not in st.session_state:
        st.session_state.url_analysis_enabled = False
    st.session_state.url_analysis_enabled = st.toggle(
        "🔗 URL Analysis",
        value=st.session_state.url_analysis_enabled,
        help="When on, paste a URL anywhere in your query and it will be fetched and analyzed before the agent responds.",
    )
    if st.session_state.url_analysis_enabled:
        st.caption("Include a URL in your query to analyze it alongside your question.")

    if st.button("Reset Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("$", "\\$"))

if prompt := st.chat_input("Ask about hyperscaler activity..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        # URL Analysis: extract and analyze any URL in the query before the agent responds
        if st.session_state.get("url_analysis_enabled"):
            import re
            urls = re.findall(r"https?://[^\s<>\"{}|\\^`\[\]]+", prompt)
            if urls:
                url = urls[0]
                with st.spinner(f"Fetching URL: {url}"):
                    try:
                        from scrapers.web import scrape_url
                        from llm.config import get_llm

                        article = scrape_url(url, source_name="URL Analysis")
                        if article and article.get("title"):
                            llm = get_llm()
                            analysis_prompt = f"""You are a market intelligence analyst. Analyze this article and compare it against what you know about the AI competitive landscape.

URL: {url}
Title: {article.get('title', '')}
Content: {article.get('summary', '')[:3000]}

Return JSON only:
{{
  "company": "which AI company this is about",
  "event_type": "product_launch | partnership | funding | hiring | policy | research | infrastructure | legal | earnings | acquisition | other",
  "headline": "one sentence summary of what happened",
  "description": "full paragraph explaining what happened, who is involved, and the context",
  "competitive_significance": "paragraph on what this means competitively for the AI landscape",
  "date": "YYYY-MM-DD or empty string"
}}"""
                            SCHEMA = {
                                "type": "object",
                                "properties": {
                                    "company": {"type": "string"},
                                    "event_type": {"type": "string"},
                                    "headline": {"type": "string"},
                                    "description": {"type": "string"},
                                    "competitive_significance": {"type": "string"},
                                    "date": {"type": "string"},
                                },
                                "required": ["company", "event_type", "headline", "description", "competitive_significance"],
                            }
                            result = llm.complete_structured(
                                "You are a market intelligence analyst. Return JSON only.",
                                analysis_prompt,
                                SCHEMA,
                            )
                            with st.expander(f"🔗 URL Analysis: {result.get('company', 'Unknown')} — {result.get('headline', '')}", expanded=True):
                                if result.get("date"):
                                    st.caption(f"{result['event_type'].replace('_', ' ').title()} · {result['date']}")
                                st.write(result.get("description", ""))
                                st.markdown(f"**Competitive significance:** {result.get('competitive_significance', '')}")
                    except Exception as e:
                        st.warning(f"URL fetch failed: {e}")

        with st.spinner("Researching..."):
            try:
                history = st.session_state.messages[-8:-1]  # last 4 exchanges, excluding current
                reply = ask(prompt, history=history, days_filter=days_filter)
            except Exception as e:
                reply = f"_Agent error: {e}_"
        st.markdown(reply.replace("$", "\\$"))
    st.session_state.messages.append({"role": "assistant", "content": reply})
