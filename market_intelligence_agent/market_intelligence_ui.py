"""
Market Intelligence Agent - Streamlit UI
Chat interface for competitive intelligence queries
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.analyst_copilot import AnalystCopilot
from src.storage import EventDatabase, EventVectorStore
from src.llm import LLMProvider


def render():
    """Render the Market Intelligence Agent interface."""

    # Initialize components
    @st.cache_resource
    def init_components():
        """Initialize agent components (cached)."""
        try:
            config_path = Path(__file__).parent / "config" / "config.yaml"
            llm = LLMProvider(str(config_path))
            db = EventDatabase(str(Path(__file__).parent / "data" / "events.db"))
            vector_store = EventVectorStore(str(Path(__file__).parent / "data" / "vector_store"))

            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Import and initialize reasoning agent
            from src.agents.competitive_reasoning import CompetitiveReasoning
            reasoning = CompetitiveReasoning(llm, db, vector_store, config)

            copilot = AnalystCopilot(llm, db, vector_store, reasoning, config)
            return copilot, db
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None, None

    copilot, db = init_components()

    # Sidebar content (minimal)
    with st.sidebar:
        st.markdown("#### 📊 Competitive Analysis")

        if db:
            # Show database stats
            try:
                providers = db.get_all_providers()
                total_events = sum(db.get_event_count(provider=p) for p in providers)

                st.markdown(f"**{total_events}** events tracked")
                st.markdown(f"**{len(providers)}** providers")

                with st.expander("📈 Event Breakdown", expanded=False):
                    for provider in sorted(providers):
                        count = db.get_event_count(provider=provider)
                        st.markdown(f"- **{provider}**: {count}")
            except Exception as e:
                st.error(f"Error loading stats: {e}")

    # Main interface
    st.markdown("## 📊 Market Intelligence Agent")
    st.markdown("*Track and analyze frontier AI competitive dynamics using the I³ Index framework*")

    # Initialize session state
    if 'time_filter' not in st.session_state:
        st.session_state.time_filter = 'all'
    if 'mi_messages' not in st.session_state:
        st.session_state.mi_messages = []

    # Time filter buttons
    st.markdown("### 📅 Time Filter")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("~30 Days", key="30days", use_container_width=True,
                     type="primary" if st.session_state.time_filter == 30 else "secondary"):
            st.session_state.time_filter = 30

    with col2:
        if st.button("~90 Days", key="90days", use_container_width=True,
                     type="primary" if st.session_state.time_filter == 90 else "secondary"):
            st.session_state.time_filter = 90

    with col3:
        if st.button("All Time", key="alltime", use_container_width=True,
                     type="primary" if st.session_state.time_filter == 'all' else "secondary"):
            st.session_state.time_filter = 'all'

    # Calculate date range
    if st.session_state.time_filter == 'all':
        time_filter_text = "All Time"
        start_date = None
        end_date = None
    else:
        days = st.session_state.time_filter
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        time_filter_text = f"Last {days} Days"

    st.caption(f"*Currently viewing: {time_filter_text}*")
    st.divider()

    # Chat interface
    st.markdown("### 💬 Ask Questions")

    # Display chat history
    for message in st.session_state.mi_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show sources if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Sources", expanded=False):
                    for source in message["sources"]:
                        st.markdown(f"- [{source.get('title', 'Event')}]({source.get('url', '#')}) - {source.get('provider', 'Unknown')}")

    # Chat input
    if prompt := st.chat_input("Ask about competitive dynamics, market trends, or specific events..."):
        if not copilot:
            st.error("Agent not initialized. Please check configuration.")
            return

        # Add user message
        st.session_state.mi_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing competitive intelligence..."):
                try:
                    # Query the copilot
                    response = copilot.chat(
                        query=prompt,
                        start_date=start_date.isoformat() if start_date else None,
                        end_date=end_date.isoformat() if end_date else None
                    )

                    # Display response
                    st.markdown(response['response'])

                    # Store message with sources
                    st.session_state.mi_messages.append({
                        "role": "assistant",
                        "content": response['response'],
                        "sources": response.get('sources', [])
                    })

                    # Show sources
                    if response.get('sources'):
                        with st.expander("📚 Sources", expanded=False):
                            for source in response['sources']:
                                st.markdown(f"- [{source.get('title', source.get('what_changed', 'Event')[:60])}]({source.get('source_url', '#')}) - {source.get('provider', 'Unknown')}")

                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.mi_messages.append({
                        "role": "assistant",
                        "content": f"⚠️ {error_msg}"
                    })
                    import traceback
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

    # Helpful prompts
    if not st.session_state.mi_messages:
        st.markdown("---")
        st.markdown("#### 💡 Example Questions")
        example_questions = [
            "Who is leading on memory portability and why?",
            "How do OpenAI and Anthropic differ on AI safety approaches?",
            "What are the latest partnerships in healthcare AI?",
            "Show me technical capability developments in the last 30 days",
            "Which providers are most active in education initiatives?"
        ]

        for question in example_questions:
            if st.button(f"💭 {question}", key=f"example_{hash(question)}", use_container_width=True):
                # Trigger the question
                st.session_state.mi_messages.append({"role": "user", "content": question})
                st.rerun()
