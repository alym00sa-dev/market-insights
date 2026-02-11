"""
Analysis Workflow

Orchestrates competitive analysis queries from user input to formatted output.

Flow: User Query → Parse Intent → Retrieve Events → Reasoning → Format Response

Why LangGraph?
- Handles complex routing (4 query types)
- State management for multi-turn conversations
- Observable execution for debugging
- Error recovery and fallbacks

Use cases:
1. Ad-hoc competitive intelligence queries
2. Multi-turn conversational analysis
3. Batch analysis of multiple queries
"""

from typing import Dict, Any, List, Optional, TypedDict, Literal
from datetime import datetime
import traceback

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from ..agents import CompetitiveReasoning, AnalystCopilot
from ..storage import EventDatabase, EventVectorStore
from ..llm import LLMProvider


# ============================================================================
# STATE SCHEMA
# ============================================================================

class AnalysisState(TypedDict):
    """
    State that flows through the analysis workflow.

    Each node can read and update the state.
    """
    # Input
    user_query: str  # Natural language question
    conversation_history: List[Dict[str, str]]  # Previous turns

    # Parsed intent
    query_type: Optional[str]  # "event_impact", "provider_comparison", etc.
    parameters: Dict[str, Any]  # Extracted parameters

    # Retrieved data
    relevant_events: List[Dict[str, Any]]  # Events retrieved for analysis
    context: Dict[str, Any]  # Additional context (provider profiles, etc.)

    # Analysis results
    analysis: Dict[str, Any]  # Raw analysis from Competitive Reasoning
    formatted_response: str  # User-friendly formatted response

    # Tracking
    errors: List[str]  # Errors encountered
    step: str  # Current step
    status: Literal["pending", "in_progress", "completed", "failed"]

    # Output
    confidence: float  # Analysis confidence
    sources: List[str]  # Event IDs cited


# ============================================================================
# WORKFLOW NODES
# ============================================================================

def parse_user_intent(state: AnalysisState, config: RunnableConfig) -> AnalysisState:
    """
    Node: Parse user query to determine intent and extract parameters.

    Uses Analyst Copilot's intent parser.
    """
    print(f"\n[Parser] Parsing query: {state['user_query']}")

    try:
        # Get components from config
        copilot: AnalystCopilot = config["configurable"]["copilot"]

        # Parse intent
        intent = copilot._parse_intent(state['user_query'])

        if 'error' in intent:
            print(f"[Parser] Failed to parse intent: {intent['error']}")
            return {
                **state,
                'errors': state.get('errors', []) + [intent['error']],
                'step': 'parse_failed',
                'status': 'failed'
            }

        print(f"[Parser] Query type: {intent['query_type']}")
        print(f"[Parser] Parameters: {intent['parameters']}")

        return {
            **state,
            'query_type': intent['query_type'],
            'parameters': intent['parameters'],
            'step': 'parse_complete',
            'status': 'in_progress'
        }

    except Exception as e:
        print(f"[Parser] Error: {e}")
        traceback.print_exc()
        return {
            **state,
            'errors': state.get('errors', []) + [f"Parser error: {str(e)}"],
            'step': 'parse_failed',
            'status': 'failed'
        }


def retrieve_relevant_events(state: AnalysisState, config: RunnableConfig) -> AnalysisState:
    """
    Node: Retrieve events relevant to the query.

    Uses database queries and semantic search based on query type.
    """
    print(f"\n[Retriever] Retrieving events for {state['query_type']}")

    try:
        # Get components from config
        db: EventDatabase = config["configurable"]["database"]
        vector_store: EventVectorStore = config["configurable"]["vector_store"]

        query_type = state['query_type']
        params = state['parameters']

        relevant_events = []

        # Different retrieval strategies per query type
        if query_type == "event_impact":
            # Retrieve specific event
            event_id = params.get('event_id')
            if event_id:
                event = db.get_event(event_id)
                if event:
                    relevant_events.append({
                        'event_id': event.event_id,
                        'provider': event.provider,
                        'what_changed': event.what_changed,
                        'published_at': event.published_at.isoformat()
                    })

        elif query_type == "provider_comparison":
            # Retrieve events for specified providers
            providers = params.get('providers', [])
            for provider in providers:
                events = db.search_events(provider=provider, limit=10)
                relevant_events.extend([{
                    'event_id': e.event_id,
                    'provider': e.provider,
                    'what_changed': e.what_changed,
                    'published_at': e.published_at.isoformat()
                } for e in events])

        elif query_type in ["leadership_ranking", "timeline"]:
            # Retrieve events for pillar (handled by reasoning agent)
            # Just log that we'll let reasoning agent handle retrieval
            print(f"  Retrieval will be handled by reasoning agent for {query_type}")

        print(f"[Retriever] Retrieved {len(relevant_events)} events")

        return {
            **state,
            'relevant_events': relevant_events,
            'step': 'retrieve_complete',
            'status': 'in_progress'
        }

    except Exception as e:
        print(f"[Retriever] Error: {e}")
        return {
            **state,
            'errors': state.get('errors', []) + [f"Retriever error: {str(e)}"],
            'step': 'retrieve_failed',
            'status': 'failed'
        }


def analyze_with_reasoning(state: AnalysisState, config: RunnableConfig) -> AnalysisState:
    """
    Node: Perform competitive analysis using Competitive Reasoning agent.

    Routes to appropriate analysis method based on query type.
    """
    print(f"\n[Reasoning] Analyzing with query type: {state['query_type']}")

    try:
        # Get components from config
        copilot: AnalystCopilot = config["configurable"]["copilot"]

        # Execute analysis via copilot (it handles routing)
        analysis = copilot._execute_analysis({
            'query_type': state['query_type'],
            'parameters': state['parameters']
        })

        if 'error' in analysis:
            print(f"[Reasoning] Analysis failed: {analysis['error']}")
            return {
                **state,
                'errors': state.get('errors', []) + [analysis['error']],
                'step': 'reasoning_failed',
                'status': 'failed'
            }

        print(f"[Reasoning] Analysis complete (confidence: {analysis.get('confidence', 0):.2f})")

        return {
            **state,
            'analysis': analysis,
            'confidence': analysis.get('confidence', 0.0),
            'step': 'reasoning_complete',
            'status': 'in_progress'
        }

    except Exception as e:
        print(f"[Reasoning] Error: {e}")
        traceback.print_exc()
        return {
            **state,
            'errors': state.get('errors', []) + [f"Reasoning error: {str(e)}"],
            'step': 'reasoning_failed',
            'status': 'failed'
        }


def format_response(state: AnalysisState, config: RunnableConfig) -> AnalysisState:
    """
    Node: Format analysis into user-friendly response.

    Converts structured analysis to readable markdown.
    """
    print(f"\n[Formatter] Formatting response")

    try:
        # Get components from config
        copilot: AnalystCopilot = config["configurable"]["copilot"]

        # Format response
        formatted = copilot._format_response(
            state['analysis'],
            {'query_type': state['query_type']}
        )

        # Extract sources
        sources = copilot._extract_sources(state['analysis'])

        print(f"[Formatter] Response formatted ({len(formatted)} chars, {len(sources)} sources)")

        return {
            **state,
            'formatted_response': formatted,
            'sources': sources,
            'step': 'format_complete',
            'status': 'completed'
        }

    except Exception as e:
        print(f"[Formatter] Error: {e}")
        return {
            **state,
            'errors': state.get('errors', []) + [f"Formatter error: {str(e)}"],
            'step': 'format_failed',
            'status': 'failed'
        }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def should_continue_after_parse(state: AnalysisState) -> str:
    """Decide whether to continue after parsing."""
    if state['status'] == 'failed':
        return END
    if not state.get('query_type'):
        print("[Router] No query type identified, ending workflow")
        return END
    return "retrieve"


def should_continue_after_retrieve(state: AnalysisState) -> str:
    """Decide whether to continue after retrieval."""
    if state['status'] == 'failed':
        return END
    # Always continue to reasoning (it handles its own retrieval if needed)
    return "reason"


def should_continue_after_reason(state: AnalysisState) -> str:
    """Decide whether to continue after reasoning."""
    if state['status'] == 'failed':
        return END
    if not state.get('analysis'):
        print("[Router] No analysis generated, ending workflow")
        return END
    return "format"


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

def create_analysis_workflow(
    llm_provider: LLMProvider,
    database: EventDatabase,
    vector_store: EventVectorStore,
    config: Dict[str, Any]
) -> StateGraph:
    """
    Create the analysis workflow graph.

    Args:
        llm_provider: LLM provider instance
        database: Event database
        vector_store: Vector store
        config: Configuration dict

    Returns:
        Compiled StateGraph ready to run

    Example:
        workflow = create_analysis_workflow(llm, db, vector_store, config)
        result = workflow.invoke({
            'user_query': 'How do OpenAI and Anthropic differ on alignment?',
            'conversation_history': []
        })
    """
    # Initialize agents
    reasoning = CompetitiveReasoning(llm_provider, database, vector_store, config)
    copilot = AnalystCopilot(llm_provider, database, vector_store, reasoning, config)

    # Create graph
    workflow = StateGraph(AnalysisState)

    # Add nodes
    workflow.add_node("parse", parse_user_intent)
    workflow.add_node("retrieve", retrieve_relevant_events)
    workflow.add_node("reason", analyze_with_reasoning)
    workflow.add_node("format", format_response)

    # Add edges
    workflow.set_entry_point("parse")

    workflow.add_conditional_edges(
        "parse",
        should_continue_after_parse,
        {
            "retrieve": "retrieve",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "retrieve",
        should_continue_after_retrieve,
        {
            "reason": "reason",
            END: END
        }
    )

    workflow.add_conditional_edges(
        "reason",
        should_continue_after_reason,
        {
            "format": "format",
            END: END
        }
    )

    workflow.add_edge("format", END)

    # Compile with config
    compiled = workflow.compile()

    # Store components in config for nodes to access
    compiled.config = {
        "configurable": {
            "copilot": copilot,
            "reasoning": reasoning,
            "database": database,
            "vector_store": vector_store
        }
    }

    return compiled


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def analyze_query(
    user_query: str,
    llm_provider: LLMProvider,
    database: EventDatabase,
    vector_store: EventVectorStore,
    config: Dict[str, Any],
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """
    One-liner to run the full analysis workflow.

    Args:
        user_query: Natural language question
        llm_provider: LLM provider
        database: Event database
        vector_store: Vector store
        config: Configuration
        conversation_history: Optional previous conversation turns

    Returns:
        Final state with formatted_response, confidence, and sources

    Example:
        result = analyze_query(
            user_query="Who is leading on data pipelines?",
            llm_provider=llm,
            database=db,
            vector_store=vector_store,
            config=config
        )
        print(result['formatted_response'])
    """
    # Create workflow
    workflow = create_analysis_workflow(
        llm_provider, database, vector_store, config
    )

    # Initial state
    initial_state: AnalysisState = {
        'user_query': user_query,
        'conversation_history': conversation_history or [],
        'query_type': None,
        'parameters': {},
        'relevant_events': [],
        'context': {},
        'analysis': {},
        'formatted_response': '',
        'errors': [],
        'step': 'started',
        'status': 'pending',
        'confidence': 0.0,
        'sources': []
    }

    # Run workflow
    final_state = workflow.invoke(initial_state, workflow.config)

    return final_state


def batch_analyze(
    queries: List[str],
    llm_provider: LLMProvider,
    database: EventDatabase,
    vector_store: EventVectorStore,
    config: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Run analysis workflow on multiple queries in batch.

    Useful for evaluating system performance or generating reports.

    Args:
        queries: List of natural language questions
        llm_provider: LLM provider
        database: Event database
        vector_store: Vector store
        config: Configuration

    Returns:
        List of final states for each query

    Example:
        queries = [
            "Who leads on technical capabilities?",
            "How did alignment evolve this year?",
            "Compare OpenAI and Anthropic"
        ]
        results = batch_analyze(queries, llm, db, vector_store, config)
        for i, result in enumerate(results):
            print(f"\nQuery {i+1}: {queries[i]}")
            print(f"Confidence: {result['confidence']:.0%}")
    """
    results = []

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"BATCH QUERY {i}/{len(queries)}")
        print(f"{'='*80}")

        try:
            result = analyze_query(
                user_query=query,
                llm_provider=llm_provider,
                database=database,
                vector_store=vector_store,
                config=config
            )
            results.append(result)

        except Exception as e:
            print(f"Query failed: {e}")
            results.append({
                'user_query': query,
                'status': 'failed',
                'errors': [str(e)],
                'formatted_response': f"Analysis failed: {str(e)}",
                'confidence': 0.0,
                'sources': []
            })

    return results
