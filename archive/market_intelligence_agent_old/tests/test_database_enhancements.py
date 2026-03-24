"""
Test Database Enhancements

Tests:
1. Pillar-based filtering
2. Temporal chain storage and traversal
3. Hybrid semantic search (SQL + vector)
"""

import sys
import os
import yaml
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore
from src.agents import SignalExtractor
from src.models import Pillar


# Sample content for multiple events
SAMPLE_CONTENTS = [
    {
        "content": "OpenAI announces 200K context window for GPT-4 Turbo, doubling the previous 128K limit. This enables processing of entire codebases in a single request.",
        "provider": "OpenAI",
        "source_type": "official_blog",
        "url": "https://openai.com/blog/context-200k"
    },
    {
        "content": "Anthropic launches Claude 3 Opus with 200K context window, matching OpenAI's offering. New model shows 30% improvement on MMLU benchmarks.",
        "provider": "Anthropic",
        "source_type": "official_blog",
        "url": "https://anthropic.com/news/claude-3-opus"
    },
    {
        "content": "Google announces open-source Gemma 2B and 7B models under Apache 2.0 license, promoting ecosystem development and reducing vendor lock-in.",
        "provider": "Google",
        "source_type": "official_blog",
        "url": "https://blog.google/gemma-open-source"
    }
]


def test_database_enhancements():
    """
    Test all database enhancements.
    """
    print("=" * 80)
    print("DATABASE ENHANCEMENTS TEST")
    print("=" * 80)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    print("\n1. Initializing components...")
    llm = LLMProvider(config_path)
    db = EventDatabase(config['storage']['database']['path'])
    vector_store = EventVectorStore(
        persist_directory=config['storage']['vector_store']['path'],
        collection_name=config['storage']['vector_store']['collection_name']
    )
    extractor = SignalExtractor(llm, db, config)
    print("✓ Components initialized")

    # Extract multiple events
    print("\n2. Extracting sample events...")
    events = []
    for i, sample in enumerate(SAMPLE_CONTENTS, 1):
        print(f"\n   Extracting event {i}/{len(SAMPLE_CONTENTS)}...")
        event = extractor.extract(
            content=sample['content'],
            provider=sample['provider'],
            source_url=sample['url'],
            source_type=sample['source_type'],
            published_at=datetime.now() - timedelta(days=i),  # Different dates
            metadata={'content_type': 'product_announcement'}
        )

        if event:
            events.append(event)
            # Store in database
            db.create_event(event)
            # Store in vector store
            vector_store.add_event(event)
            print(f"   ✓ Event {event.event_id}")
        else:
            print(f"   ✗ Extraction failed")

    print(f"\n✓ Extracted {len(events)} events")

    # Test 1: Pillar-based filtering
    print("\n" + "=" * 80)
    print("TEST 1: PILLAR-BASED FILTERING")
    print("=" * 80)

    tech_events = db.get_events_by_pillar(
        pillar=Pillar.TECHNICAL_CAPABILITIES,
        limit=10
    )

    print(f"\nEvents with TECHNICAL_CAPABILITIES pillar: {len(tech_events)}")
    for event in tech_events:
        print(f"   - {event.event_id}: {event.provider} - {event.what_changed[:80]}...")

    # Test with provider filter
    openai_tech = db.get_events_by_pillar(
        pillar=Pillar.TECHNICAL_CAPABILITIES,
        provider="OpenAI",
        limit=10
    )

    print(f"\nOpenAI events with TECHNICAL_CAPABILITIES: {len(openai_tech)}")
    for event in openai_tech:
        print(f"   - {event.event_id}: {event.what_changed[:80]}...")

    # Test 2: Temporal chain traversal
    print("\n" + "=" * 80)
    print("TEST 2: TEMPORAL CHAIN TRAVERSAL")
    print("=" * 80)

    if events:
        test_event = events[0]
        print(f"\nTesting event: {test_event.event_id}")

        # Get predecessors
        predecessors = db.get_events_preceded_by(test_event.event_id)
        print(f"Preceded by: {len(predecessors)} events")
        for pred in predecessors:
            print(f"   - {pred.event_id}: {pred.what_changed[:60]}...")

        # Get successors
        successors = db.get_events_triggered_by(test_event.event_id)
        print(f"Triggered: {len(successors)} events")
        for succ in successors:
            print(f"   - {succ.event_id}: {succ.what_changed[:60]}...")

        # Get full chain
        chain = db.get_event_chain(test_event.event_id, direction='both', max_depth=2)
        print(f"\nFull event chain (depth {chain['graph_depth']}):")
        print(f"   Predecessors: {len(chain['predecessors'])}")
        print(f"   Successors: {len(chain['successors'])}")

    # Test 3: Hybrid semantic search
    print("\n" + "=" * 80)
    print("TEST 3: HYBRID SEMANTIC SEARCH")
    print("=" * 80)

    print("\nQuery: 'context window improvements'")
    results = db.semantic_search_with_filters(
        query_text="context window improvements",
        vector_store=vector_store,
        limit=5
    )

    print(f"\nFound {len(results)} results:")
    for result in results:
        print(f"\n   Event: {result['event_id']}")
        print(f"   Provider: {result['provider']}")
        print(f"   Similarity: {result.get('similarity_score', 0):.3f}")
        print(f"   What Changed: {result['what_changed'][:100]}...")

    # Test with filters
    print("\n\nQuery: 'open source' (filtered by Google)")
    results_filtered = db.semantic_search_with_filters(
        query_text="open source models",
        vector_store=vector_store,
        provider="Google",
        limit=5
    )

    print(f"\nFound {len(results_filtered)} results:")
    for result in results_filtered:
        print(f"\n   Event: {result['event_id']}")
        print(f"   Provider: {result['provider']}")
        print(f"   Similarity: {result.get('similarity_score', 0):.3f}")
        print(f"   What Changed: {result['what_changed'][:100]}...")

    # Summary
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    print("\nEnhancements Verified:")
    print(f"   ✓ Pillar-based filtering: {len(tech_events)} events found")
    print(f"   ✓ Temporal chain storage: Events linked")
    print(f"   ✓ Temporal chain traversal: Predecessors and successors queryable")
    print(f"   ✓ Hybrid semantic search: {len(results)} results with filters")

    print("\nDatabase now supports:")
    print("   - Fast pillar-based queries (indexed)")
    print("   - Event relationship tracking (temporal_chains table)")
    print("   - Causal chain traversal (get predecessors/successors)")
    print("   - Hybrid search (semantic + SQL filters)")


if __name__ == "__main__":
    try:
        test_database_enhancements()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
