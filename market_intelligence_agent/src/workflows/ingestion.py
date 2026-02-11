"""
Ingestion Workflow

Orchestrates the flow from source discovery to event storage.

Flow: Source Scout → Content Harvester → Signal Extractor → Store Event

Why LangGraph?
- Handles state management across steps
- Built-in error handling and retries
- Conditional routing (e.g., skip harvesting if source already processed)
- Observable execution (can track progress)

Use cases:
1. Discover and ingest from new source
2. Refresh content from known sources
3. Bulk ingestion from multiple sources
"""

from typing import Dict, Any, List, Optional, TypedDict, Literal
from datetime import datetime
import traceback

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from ..agents import SourceScout, ContentHarvester, SignalExtractor, EnsembleSignalExtractor
from ..storage import EventDatabase, EventVectorStore
from ..llm import LLMProvider
from ..models import MarketSignalEvent


# ============================================================================
# STATE SCHEMA
# ============================================================================

class IngestionState(TypedDict):
    """
    State that flows through the ingestion workflow.

    Each node can read and update the state.
    """
    # Input
    query: str  # Search query or source URL
    provider: str  # Provider name (e.g., "OpenAI")
    source_type: Optional[str]  # "official_blog", "documentation", etc.
    use_ensemble: bool  # Whether to use ensemble extraction

    # Intermediate results
    sources: List[Dict[str, Any]]  # Sources discovered by Scout
    harvested_content: List[Dict[str, Any]]  # Content from Harvester
    events: List[MarketSignalEvent]  # Extracted events

    # Tracking
    errors: List[str]  # Errors encountered
    step: str  # Current step
    status: Literal["pending", "in_progress", "completed", "failed"]

    # Output
    events_stored: int  # Number of events successfully stored
    event_ids: List[str]  # IDs of stored events


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def scout_sources(state: IngestionState, config: RunnableConfig) -> IngestionState:
    """
    Node: Discover sources to monitor.

    Uses Source Scout to find relevant URLs.
    """
    print(f"\n[Scout] Discovering sources for query: {state['query']}")

    try:
        # Get components from config
        scout: SourceScout = config["configurable"]["scout"]

        # Discover sources
        candidates = scout.discover(
            query=state['query'],
            provider=state['provider'],
            limit=5
        )

        # Convert to dicts for state
        sources = []
        for candidate in candidates:
            sources.append({
                'url': candidate.url,
                'source_type': candidate.source_type,
                'relevance_score': candidate.relevance_score,
                'metadata': candidate.metadata
            })

        print(f"[Scout] Found {len(sources)} sources")

        return {
            **state,
            'sources': sources,
            'step': 'scout_complete',
            'status': 'in_progress'
        }

    except Exception as e:
        print(f"[Scout] Error: {e}")
        return {
            **state,
            'errors': state.get('errors', []) + [f"Scout error: {str(e)}"],
            'step': 'scout_failed',
            'status': 'failed'
        }


def harvest_content(state: IngestionState, config: RunnableConfig) -> IngestionState:
    """
    Node: Fetch content from sources.

    Uses Content Harvester to retrieve and clean content.
    """
    print(f"\n[Harvester] Fetching content from {len(state['sources'])} sources")

    try:
        # Get components from config
        harvester: ContentHarvester = config["configurable"]["harvester"]

        harvested = []
        for source in state['sources']:
            print(f"  Fetching: {source['url']}")

            content = harvester.harvest(
                url=source['url'],
                provider=state['provider'],
                source_type=source['source_type']
            )

            if content:
                harvested.append({
                    'url': content.url,
                    'content': content.filtered_content,
                    'published_at': content.published_at,
                    'source_type': content.source_type,
                    'metadata': content.metadata
                })
                print(f"    ✓ Retrieved ({len(content.filtered_content)} chars)")
            else:
                print(f"    ✗ Failed to retrieve")

        print(f"[Harvester] Successfully harvested {len(harvested)}/{len(state['sources'])} sources")

        return {
            **state,
            'harvested_content': harvested,
            'step': 'harvest_complete',
            'status': 'in_progress'
        }

    except Exception as e:
        print(f"[Harvester] Error: {e}")
        return {
            **state,
            'errors': state.get('errors', []) + [f"Harvester error: {str(e)}"],
            'step': 'harvest_failed',
            'status': 'failed'
        }


def extract_signals(state: IngestionState, config: RunnableConfig) -> IngestionState:
    """
    Node: Extract structured events from content.

    Uses Signal Extractor (v1 or v2) to parse content into MarketSignalEvent.
    """
    use_ensemble = state.get('use_ensemble', False)
    extractor_type = "Ensemble (v2)" if use_ensemble else "Standard (v1)"

    print(f"\n[Extractor] Extracting signals using {extractor_type}")

    try:
        # Get components from config
        extractor = config["configurable"]["ensemble_extractor" if use_ensemble else "extractor"]

        events = []
        for content_data in state['harvested_content']:
            print(f"  Processing: {content_data['url']}")

            event = extractor.extract(
                content=content_data['content'],
                provider=state['provider'],
                source_url=content_data['url'],
                source_type=content_data['source_type'],
                published_at=content_data.get('published_at'),
                metadata=content_data.get('metadata', {})
            )

            if event:
                events.append(event)
                print(f"    ✓ Extracted event: {event.event_id}")
            else:
                print(f"    ✗ Extraction failed")

        print(f"[Extractor] Successfully extracted {len(events)}/{len(state['harvested_content'])} events")

        return {
            **state,
            'events': events,
            'step': 'extract_complete',
            'status': 'in_progress'
        }

    except Exception as e:
        print(f"[Extractor] Error: {e}")
        traceback.print_exc()
        return {
            **state,
            'errors': state.get('errors', []) + [f"Extractor error: {str(e)}"],
            'step': 'extract_failed',
            'status': 'failed'
        }


def store_events(state: IngestionState, config: RunnableConfig) -> IngestionState:
    """
    Node: Store events in database and vector store.

    Persists events for later retrieval and analysis.
    """
    print(f"\n[Storage] Storing {len(state['events'])} events")

    try:
        # Get components from config
        db: EventDatabase = config["configurable"]["database"]
        vector_store: EventVectorStore = config["configurable"]["vector_store"]

        stored_count = 0
        event_ids = []

        for event in state['events']:
            try:
                # Store in database
                db.create_event(event)

                # Store in vector store
                vector_store.add_event(event)

                stored_count += 1
                event_ids.append(event.event_id)
                print(f"  ✓ Stored: {event.event_id}")

            except Exception as e:
                print(f"  ✗ Failed to store {event.event_id}: {e}")

        print(f"[Storage] Successfully stored {stored_count}/{len(state['events'])} events")

        return {
            **state,
            'events_stored': stored_count,
            'event_ids': event_ids,
            'step': 'storage_complete',
            'status': 'completed'
        }

    except Exception as e:
        print(f"[Storage] Error: {e}")
        return {
            **state,
            'errors': state.get('errors', []) + [f"Storage error: {str(e)}"],
            'step': 'storage_failed',
            'status': 'failed'
        }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def should_continue_after_scout(state: IngestionState) -> str:
    """Decide whether to continue after scouting."""
    if state['status'] == 'failed':
        return END
    if not state.get('sources'):
        print("[Router] No sources found, ending workflow")
        return END
    return "harvest"


def should_continue_after_harvest(state: IngestionState) -> str:
    """Decide whether to continue after harvesting."""
    if state['status'] == 'failed':
        return END
    if not state.get('harvested_content'):
        print("[Router] No content harvested, ending workflow")
        return END
    return "extract"


def should_continue_after_extract(state: IngestionState) -> str:
    """Decide whether to continue after extraction."""
    if state['status'] == 'failed':
        return END
    if not state.get('events'):
        print("[Router] No events extracted, ending workflow")
        return END
    return "store"


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

def create_ingestion_workflow(
    llm_provider: LLMProvider,
    database: EventDatabase,
    vector_store: EventVectorStore,
    config: Dict[str, Any]
) -> StateGraph:
    """
    Create the ingestion workflow graph.

    Args:
        llm_provider: LLM provider instance
        database: Event database
        vector_store: Vector store
        config: Configuration dict

    Returns:
        Compiled StateGraph ready to run

    Example:
        workflow = create_ingestion_workflow(llm, db, vector_store, config)
        result = workflow.invoke({
            'query': 'OpenAI GPT-4 updates',
            'provider': 'OpenAI',
            'source_type': 'official_blog',
            'use_ensemble': False
        })
    """
    # Initialize agents
    scout = SourceScout(llm_provider, database, config)
    harvester = ContentHarvester(llm_provider, database, config)
    extractor = SignalExtractor(llm_provider, database, config)
    ensemble_extractor = EnsembleSignalExtractor(llm_provider, database, config)

    # Create graph
    workflow = StateGraph(IngestionState)

    # Add nodes
    workflow.add_node("scout", scout_sources)
    workflow.add_node("harvest", harvest_content)
    workflow.add_node("extract", extract_signals)
    workflow.add_node("store", store_events)

    # Add edges
    workflow.set_entry_point("scout")

    workflow.add_conditional_edges(
        "scout",
        should_continue_after_scout,
        {
            "harvest": "harvest",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "harvest",
        should_continue_after_harvest,
        {
            "extract": "extract",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "extract",
        should_continue_after_extract,
        {
            "store": "store",
            END: END
        }
    )

    workflow.add_edge("store", END)

    # Compile with config
    compiled = workflow.compile()

    # Store components in config for nodes to access
    compiled.config = {
        "configurable": {
            "scout": scout,
            "harvester": harvester,
            "extractor": extractor,
            "ensemble_extractor": ensemble_extractor,
            "database": database,
            "vector_store": vector_store
        }
    }

    return compiled


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def ingest_from_query(
    query: str,
    provider: str,
    llm_provider: LLMProvider,
    database: EventDatabase,
    vector_store: EventVectorStore,
    config: Dict[str, Any],
    source_type: Optional[str] = None,
    use_ensemble: bool = False
) -> Dict[str, Any]:
    """
    One-liner to run the full ingestion workflow.

    Args:
        query: Search query or source URL
        provider: Provider name
        llm_provider: LLM provider
        database: Event database
        vector_store: Vector store
        config: Configuration
        source_type: Optional source type
        use_ensemble: Whether to use ensemble extraction

    Returns:
        Final state with events_stored and event_ids

    Example:
        result = ingest_from_query(
            query="OpenAI GPT-4 Turbo context window",
            provider="OpenAI",
            llm_provider=llm,
            database=db,
            vector_store=vector_store,
            config=config,
            use_ensemble=False
        )
        print(f"Stored {result['events_stored']} events")
    """
    # Create workflow
    workflow = create_ingestion_workflow(
        llm_provider, database, vector_store, config
    )

    # Initial state
    initial_state: IngestionState = {
        'query': query,
        'provider': provider,
        'source_type': source_type,
        'use_ensemble': use_ensemble,
        'sources': [],
        'harvested_content': [],
        'events': [],
        'errors': [],
        'step': 'started',
        'status': 'pending',
        'events_stored': 0,
        'event_ids': []
    }

    # Run workflow
    final_state = workflow.invoke(initial_state, workflow.config)

    return final_state


def ingest_from_url(
    url: str,
    provider: str,
    source_type: str,
    llm_provider: LLMProvider,
    database: EventDatabase,
    vector_store: EventVectorStore,
    config: Dict[str, Any],
    use_ensemble: bool = False
) -> Dict[str, Any]:
    """
    Ingest directly from a known URL (skip scouting).

    Faster when you already know the source URL.

    Args:
        url: Source URL
        provider: Provider name
        source_type: Source type
        llm_provider: LLM provider
        database: Event database
        vector_store: Vector store
        config: Configuration
        use_ensemble: Whether to use ensemble extraction

    Returns:
        Final state with events_stored and event_ids
    """
    # Create workflow
    workflow = create_ingestion_workflow(
        llm_provider, database, vector_store, config
    )

    # Initial state with pre-populated source (skip scouting)
    initial_state: IngestionState = {
        'query': url,
        'provider': provider,
        'source_type': source_type,
        'use_ensemble': use_ensemble,
        'sources': [{
            'url': url,
            'source_type': source_type,
            'relevance_score': 1.0,
            'metadata': {}
        }],
        'harvested_content': [],
        'events': [],
        'errors': [],
        'step': 'scout_skipped',
        'status': 'in_progress',
        'events_stored': 0,
        'event_ids': []
    }

    # Run workflow starting from harvest (skip scout)
    # We'll manually invoke the nodes
    harvester = ContentHarvester(llm_provider, database, config)
    extractor = SignalExtractor(llm_provider, database, config) if not use_ensemble else EnsembleSignalExtractor(llm_provider, database, config)

    config_dict = {
        "configurable": {
            "harvester": harvester,
            "extractor": extractor if not use_ensemble else None,
            "ensemble_extractor": extractor if use_ensemble else None,
            "database": database,
            "vector_store": vector_store
        }
    }

    # Run steps manually
    state = harvest_content(initial_state, config_dict)
    if state['status'] != 'failed' and state.get('harvested_content'):
        state = extract_signals(state, config_dict)
        if state['status'] != 'failed' and state.get('events'):
            state = store_events(state, config_dict)

    return state
