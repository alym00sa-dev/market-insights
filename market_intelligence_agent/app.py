"""
Market Intelligence Agent - Streamlit Interface
"""
import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
import json
import hashlib

# Page configuration
st.set_page_config(
    page_title="Market Intelligence Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add market_intelligence_agent to path
mi_agent_path = Path(__file__).parent / "market_intelligence_agent"
if str(mi_agent_path) not in sys.path:
    sys.path.insert(0, str(mi_agent_path))

# Custom CSS for clean interface
st.markdown("""
<style>
    /* Import Garamond-like font */
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;600;700&display=swap');

    /* Apply Garamond font globally */
    html, body, [class*="css"], * {
        font-family: 'EB Garamond', Garamond, 'Apple Garamond', 'Times New Roman', serif !important;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Hide keyboard shortcut hint - comprehensive */
    [data-testid="stSidebarNav"] button[kind="header"] span:last-child,
    .css-1544g2n,
    [class*="keyboardShortcut"],
    button[kind="header"] kbd,
    [data-testid="stSidebarNav"] kbd,
    .st-key,
    [class*="KeyboardShortcut"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
    }

    /* Hide all kbd elements */
    kbd {
        display: none !important;
    }

    /* Enable scrollbar in sidebar with proper overflow */
    [data-testid="stSidebar"] {
        overflow-y: auto !important;
    }

    [data-testid="stSidebar"] > div:first-child {
        overflow-y: auto !important;
        max-height: 100vh !important;
        padding-top: 0 !important;
    }

    /* Selectbox spacing */
    div[data-testid="stSelectbox"] {
        margin-bottom: 0.75rem !important;
    }

    /* Dropdown hover effect */
    [data-baseweb="popover"] ul li:hover {
        background-color: #f0f2f6 !important;
    }

    /* Session cards - clean like ChatGPT/Claude */
    [data-testid="stSidebar"] button[kind="secondary"] {
        height: auto;
        min-height: 60px;
        white-space: normal;
        word-wrap: break-word;
        text-align: left;
        background-color: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.875rem 0.75rem !important;
        transition: all 0.2s ease;
        border-radius: 8px !important;
        margin-bottom: 0.5rem !important;
    }

    /* Hover effect - subtle background */
    [data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        opacity: 1;
    }

    /* Active session highlight - like the reference app */
    [data-testid="stSidebar"] button[kind="secondary"][data-active="true"] {
        background-color: #eff6ff !important;
        border: 1px solid #3b82f6 !important;
    }

    /* Sessions toggle button - make it look like a header */
    [data-testid="stSidebar"] button[key="sessions_toggle"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0.5rem !important;
        text-align: left !important;
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        color: #6b7280 !important;
        height: auto !important;
        min-height: auto !important;
    }

    [data-testid="stSidebar"] button[key="sessions_toggle"]:hover {
        background: #f9fafb !important;
        color: #6b7280 !important;
    }

    /* Delete button styling - minimal × button */
    [data-testid="stSidebar"] button[key*="delete_"] {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #9ca3af !important;
        padding: 0.5rem !important;
        font-size: 1.25rem !important;
        min-height: auto !important;
        height: auto !important;
        width: 100% !important;
        transition: color 0.15s ease !important;
    }

    [data-testid="stSidebar"] button[key*="delete_"]:hover {
        color: #ef4444 !important;
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Session management functions
def get_sessions_file():
    """Get path to sessions file."""
    sessions_dir = Path(__file__).parent / "market_intelligence_agent" / "data"
    sessions_dir.mkdir(exist_ok=True)
    return sessions_dir / "chat_sessions.json"

def load_sessions():
    """Load all chat sessions."""
    sessions_file = get_sessions_file()
    if sessions_file.exists():
        with open(sessions_file, 'r') as f:
            return json.load(f)
    return {}

def save_sessions(sessions):
    """Save all chat sessions."""
    sessions_file = get_sessions_file()
    with open(sessions_file, 'w') as f:
        json.dump(sessions, f, indent=2, default=str)

def create_session_id(first_message):
    """Create a unique session ID based on first message."""
    return hashlib.md5(f"{first_message}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]

def get_or_create_session():
    """Get current session or create new one."""
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    return st.session_state.current_session_id

try:
    from src.agents.analyst_copilot import AnalystCopilot
    from src.agents.competitive_reasoning import CompetitiveReasoning
    from src.storage import EventDatabase, EventVectorStore
    from src.llm import LLMProvider
    import yaml

    # Initialize components
    @st.cache_resource
    def init_components():
        """Initialize agent components (cached)."""
        try:
            config_path = mi_agent_path / "config" / "config.yaml"
            llm = LLMProvider(str(config_path))
            db = EventDatabase(str(mi_agent_path / "data" / "events.db"))
            vector_store = EventVectorStore(str(mi_agent_path / "data" / "vector_store"))

            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Create CompetitiveReasoning agent first
            reasoning = CompetitiveReasoning(llm, db, vector_store, config)

            # Then create AnalystCopilot with the reasoning agent
            copilot = AnalystCopilot(llm, db, vector_store, reasoning, config)
            return copilot, db
        except Exception as e:
            st.error(f"Failed to initialize: {e}")
            return None, None

    copilot, db = init_components()

    # Initialize session state
    if 'time_filter' not in st.session_state:
        st.session_state.time_filter = 'all'
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'sessions_expanded' not in st.session_state:
        st.session_state.sessions_expanded = False
    if 'response_length' not in st.session_state:
        st.session_state.response_length = 'medium'
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    if 'num_sources' not in st.session_state:
        st.session_state.num_sources = 10

    # Sidebar
    with st.sidebar:
        # Time filter dropdown
        time_options = {
            "Last 30 Days": 30,
            "Last 60 Days": 60,
            "Last 90 Days": 90,
            "All Time": 'all'
        }

        # Reverse mapping for display
        time_labels = {v: k for k, v in time_options.items()}

        def on_time_change():
            selected = st.session_state.time_filter_dropdown
            st.session_state.time_filter = time_options[selected]

        selected_time = st.selectbox(
            "Time Range",
            options=list(time_options.keys()),
            index=list(time_options.keys()).index(time_labels[st.session_state.time_filter]),
            key="time_filter_dropdown",
            on_change=on_time_change
        )

        st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

        # Response length dropdown
        response_length_options = {
            "Concise": "short",
            "Balanced": "medium",
            "Comprehensive": "long"
        }

        # Reverse mapping for display
        length_labels = {v: k for k, v in response_length_options.items()}

        def on_length_change():
            selected = st.session_state.response_length_dropdown
            st.session_state.response_length = response_length_options[selected]

        st.selectbox(
            "Response Length",
            options=list(response_length_options.keys()),
            index=list(response_length_options.keys()).index(length_labels[st.session_state.response_length]),
            key="response_length_dropdown",
            on_change=on_length_change,
            help="Choose response detail level: Concise (~4K tokens, ~3K words), Balanced (~8K tokens, ~6K words), Comprehensive (~16K tokens, ~12K words)"
        )

        st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

        # Temperature slider
        st.session_state.temperature = st.slider(
            "Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Lower = more focused and deterministic, Higher = more creative and varied (0.0-1.0)"
        )

        st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

        # Number of sources dropdown
        num_sources_options = {
            "10 sources": 10,
            "15 sources": 15,
            "20 sources": 20
        }

        # Reverse mapping for display
        sources_labels = {v: k for k, v in num_sources_options.items()}

        def on_sources_change():
            selected = st.session_state.num_sources_dropdown
            st.session_state.num_sources = num_sources_options[selected]

        st.selectbox(
            "Number of Sources",
            options=list(num_sources_options.keys()),
            index=list(num_sources_options.keys()).index(sources_labels.get(st.session_state.num_sources, "10 sources")),
            key="num_sources_dropdown",
            on_change=on_sources_change,
            help="How many source events to retrieve for context (more sources = more comprehensive but slower)"
        )

        st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

        # URL Analysis toggle (experimental feature)
        if 'url_analysis_enabled' not in st.session_state:
            st.session_state.url_analysis_enabled = False

        st.session_state.url_analysis_enabled = st.toggle(
            "🔗 URL Analysis (Beta)",
            value=st.session_state.url_analysis_enabled,
            help="Experimental feature: Paste a link in your query to compare external content with our database"
        )

        if st.session_state.url_analysis_enabled:
            st.markdown(
                '<p style="font-size: 0.75rem; color: #6b7280; margin-top: 0.5rem;">ℹ️ Experimental: Another way to make the agent multimodal beyond file uploads. Paste a URL in your query to analyze external announcements. bthosURLs must be publicly accessible (not behind paywalls or login).</p>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='margin: 0.5rem 0;'></div>", unsafe_allow_html=True)

        # New chat button
        if st.button("New Chat", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.current_session_id = None
            st.rerun()

        st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)

        # Sessions toggle
        st.markdown("---")
        caret_icon = "▼" if st.session_state.sessions_expanded else "▶"
        if st.button(f"{caret_icon}  CHAT HISTORY", key="sessions_toggle", use_container_width=True, type="secondary"):
            st.session_state.sessions_expanded = not st.session_state.sessions_expanded
            st.rerun()

        # Show sessions
        if st.session_state.sessions_expanded:
            sessions = load_sessions()
            if sessions:
                # Sort by timestamp (newest first)
                sorted_sessions = sorted(sessions.items(), key=lambda x: x[1].get('created_at', ''), reverse=True)

                for session_id, session_data in sorted_sessions[:20]:  # Show last 20
                    first_query = session_data.get('first_query', 'Chat session')
                    # Truncate to 70 characters for display
                    display_query = first_query if len(first_query) <= 70 else first_query[:70] + "..."

                    created_at = datetime.fromisoformat(session_data.get('created_at', datetime.now().isoformat()))
                    time_str = created_at.strftime('%b %d, %I:%M %p')
                    msg_count = len(session_data.get('messages', []))

                    is_active = st.session_state.get('current_session_id') == session_id

                    # Display session card in two columns
                    col1, col2 = st.columns([9, 1])

                    with col1:
                        # Create button with session info
                        button_label = f"{display_query}\n{time_str} • {msg_count} messages"
                        if st.button(
                            button_label,
                            key=f"session_{session_id}",
                            use_container_width=True,
                            type="secondary"
                        ):
                            # Load this session
                            st.session_state.current_session_id = session_id
                            st.session_state.messages = session_data.get('messages', [])
                            st.session_state.time_filter = session_data.get('time_filter', 'all')
                            st.rerun()

                    with col2:
                        if st.button("×", key=f"delete_{session_id}", help="Delete session", use_container_width=True):
                            # Delete session
                            sessions = load_sessions()
                            if session_id in sessions:
                                del sessions[session_id]
                                save_sessions(sessions)
                            if st.session_state.get('current_session_id') == session_id:
                                st.session_state.current_session_id = None
                                st.session_state.messages = []
                            st.rerun()
            else:
                st.markdown('<p style="text-align: center; color: #6b7280; font-size: 0.875rem; padding: 2rem 0;">No chat history yet</p>', unsafe_allow_html=True)

    # Main content
    st.title("Market Intelligence Agent")
    st.warning("⚠️ This is a beta tool and not open for foundation distribution. Additional testing and validation are required before broader use.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask about competitive dynamics, market trends, or specific events...")

    if prompt:
        if not copilot:
            st.error("Agent not initialized. Please check configuration.")
        else:
            # Create session if this is first message
            if not st.session_state.messages:
                session_id = create_session_id(prompt)
                st.session_state.current_session_id = session_id

                # Save new session
                sessions = load_sessions()
                sessions[session_id] = {
                    'session_id': session_id,
                    'first_query': prompt,
                    'created_at': datetime.now().isoformat(),
                    'time_filter': st.session_state.time_filter,
                    'messages': []
                }
                save_sessions(sessions)

            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Save to session
            if st.session_state.current_session_id:
                sessions = load_sessions()
                if st.session_state.current_session_id in sessions:
                    sessions[st.session_state.current_session_id]['messages'] = st.session_state.messages
                    sessions[st.session_state.current_session_id]['time_filter'] = st.session_state.time_filter
                    save_sessions(sessions)

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        # Add time context to query if filter is active
                        query_with_time = prompt
                        if st.session_state.time_filter == 30:
                            query_with_time = f"{prompt} (in the last 30 days)"
                        elif st.session_state.time_filter == 60:
                            query_with_time = f"{prompt} (in the last 60 days)"
                        elif st.session_state.time_filter == 90:
                            query_with_time = f"{prompt} (in the last 90 days)"

                        # Map response length to token limits
                        token_limits = {
                            'short': 6144,     # ~4,500 words
                            'medium': 12288,   # ~9,000 words
                            'long': 20480      # ~15,000 words
                        }
                        max_tokens = token_limits.get(st.session_state.response_length, 8192)

                        # Query the copilot with per-hyperscaler search
                        # Convert time_filter to days (None if 'all')
                        time_filter_days = None if st.session_state.time_filter == 'all' else st.session_state.time_filter

                        # Check for URL analysis if feature is enabled
                        external_url = None
                        if st.session_state.url_analysis_enabled:
                            external_url = copilot.extract_url(prompt)
                            if external_url:
                                st.info(f"🔗 Analyzing external URL: {external_url}")
                                # Test fetch
                                try:
                                    test_fetch = copilot.fetch_url_content(external_url)
                                    if test_fetch and 'error' in test_fetch:
                                        st.warning(f"⚠️ URL fetch error: {test_fetch['error']}")
                                    elif test_fetch:
                                        st.success(f"✓ Fetched: {test_fetch.get('provider', 'Unknown')} - {test_fetch.get('date', 'No date')}")
                                    else:
                                        st.warning("⚠️ URL fetch returned no content")
                                except Exception as e:
                                    st.error(f"❌ URL fetch exception: {str(e)}")

                        response = copilot.query(
                            user_query=query_with_time,
                            include_raw_data=False,
                            max_tokens=max_tokens,
                            temperature=st.session_state.temperature,
                            search_mode='per_provider',  # Gather results from each hyperscaler
                            num_sources=st.session_state.num_sources,
                            response_length=st.session_state.response_length,  # Pass length setting to adjust detail
                            time_filter_days=time_filter_days,  # Actually filter by date
                            external_url=external_url  # Pass URL for analysis if detected
                        )

                        # Display response
                        st.markdown(response['answer'])

                        # Display token usage info with limit
                        if response.get('usage'):
                            usage = response['usage']
                            input_tokens = usage.get('input_tokens', 0)
                            output_tokens = usage.get('output_tokens', 0)
                            total_tokens = input_tokens + output_tokens

                            # Show limit vs actual
                            percentage = (output_tokens / max_tokens * 100) if max_tokens > 0 else 0
                            st.caption(f"📊 Tokens: {output_tokens:,} / {max_tokens:,} output ({percentage:.1f}% of limit) • {input_tokens:,} input • {total_tokens:,} total")

                        # Fetch full event details for sources (event IDs)
                        source_events = []
                        if response.get('sources'):
                            for event_id in response['sources']:
                                event = db.get_event(event_id)
                                if event:
                                    source_events.append({
                                        'event_id': event.event_id,
                                        'what_changed': event.what_changed,
                                        'source_url': event.source_url,
                                        'provider': event.provider
                                    })

                        # Store message with sources
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response['answer'],
                            "sources": source_events
                        })

                        # Save to session
                        if st.session_state.current_session_id:
                            sessions = load_sessions()
                            if st.session_state.current_session_id in sessions:
                                sessions[st.session_state.current_session_id]['messages'] = st.session_state.messages
                                save_sessions(sessions)

                    except Exception as e:
                        error_msg = f"Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"⚠️ {error_msg}"
                        })

except ImportError as e:
    st.error(f"Error loading Market Intelligence Agent: {e}")
    st.info("""
    **Troubleshooting:**
    1. Ensure database exists at `market_intelligence_agent/data/events.db`
    2. Check API keys in `.env`
    3. Install dependencies: `cd market_intelligence_agent && pip install -r requirements.txt`
    """)
    import traceback
    with st.expander("Error Details"):
        st.code(traceback.format_exc())
