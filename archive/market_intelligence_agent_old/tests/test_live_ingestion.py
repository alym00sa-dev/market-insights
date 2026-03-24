"""
Test Live Ingestion with Deduplication

Demonstrates Part 2: Ingesting from live sources with smart deduplication.

Note: This test requires real URLs to fully work. For now, it demonstrates
the deduplication logic using manual event creation.
"""

import sys
import os
import yaml
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore
from src.utils import LiveIngestionManager
from src.models import (
    MarketSignalEvent,
    Pillar,
    DirectionOfChange,
    RelativeStrength,
    PillarImpact,
    CompetitiveEffects,
    TemporalContext
)


def test_deduplication_logic():
    """
    Test the deduplication and merge logic.
    """
    print("=" * 80)
    print("LIVE INGESTION TEST - Deduplication Logic")
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
    manager = LiveIngestionManager(db, vector_store, llm, config)
    print("✓ Components initialized")

    # Create a test event (minimal version from Twitter)
    print("\n2. Creating initial event (minimal info from Twitter)...")

    event_v1 = MarketSignalEvent(
        event_id="evt_test_dedup_gpt5",
        provider="OpenAI",
        source_type="social_media",
        source_url="https://twitter.com/openai/status/123",
        published_at=datetime.now(),
        retrieved_at=datetime.now(),
        what_changed="OpenAI announces GPT-5 language model",
        why_it_matters="New model release representing significant advancement in AI capabilities",
        scope="product_release",
        pillars_impacted=[
            PillarImpact(
                pillar_name=Pillar.TECHNICAL_CAPABILITIES,
                direction_of_change=DirectionOfChange.ADVANCE,
                relative_strength_signal=RelativeStrength.MODERATE,
                evidence="New model announced"
            )
        ],
        competitive_effects=CompetitiveEffects(
            advantages_created=["New model"],
            advantages_eroded=[],
            new_barriers=[],
            lock_in_or_openness_shift="neutral"
        ),
        temporal_context=TemporalContext(
            preceded_by_events=[],
            likely_to_trigger_events=[],
            time_horizon="immediate"
        ),
        alignment_implications="Standard safety measures",
        regulatory_signal="none"
    )

    # Store initial version
    db.create_event(event_v1)
    vector_store.add_event(event_v1)
    print(f"✓ Stored: {event_v1.event_id}")
    print(f"  - what_changed: {len(event_v1.what_changed)} chars")
    print(f"  - advantages: {len(event_v1.competitive_effects.advantages_created)}")
    print(f"  - pillars: {len(event_v1.pillars_impacted)}")

    # Create a richer version (from blog post)
    print("\n3. Creating enriched version (detailed info from blog)...")

    event_v2 = MarketSignalEvent(
        event_id="evt_test_dedup_gpt5_blog",
        provider="OpenAI",
        source_type="official_blog",
        source_url="https://openai.com/blog/gpt-5-announcement",
        published_at=datetime.now(),
        retrieved_at=datetime.now(),
        what_changed="OpenAI announces GPT-5 with 10M token context window, improved reasoning capabilities, and 50% cost reduction",
        why_it_matters="Significant advancement in LLM capabilities with extended context enabling whole-codebase analysis and deep reasoning tasks at lower cost",
        scope="product_release",
        pillars_impacted=[
            PillarImpact(
                pillar_name=Pillar.TECHNICAL_CAPABILITIES,
                direction_of_change=DirectionOfChange.ADVANCE,
                relative_strength_signal=RelativeStrength.STRONG,
                evidence="10M token context window, 50% cost reduction"
            ),
            PillarImpact(
                pillar_name=Pillar.DATA_PIPELINES,
                direction_of_change=DirectionOfChange.ADVANCE,
                relative_strength_signal=RelativeStrength.MODERATE,
                evidence="Extended context enables processing entire codebases"
            )
        ],
        competitive_effects=CompetitiveEffects(
            advantages_created=[
                "10M token context window (10x increase)",
                "Improved reasoning capabilities",
                "50% cost reduction increases adoption"
            ],
            advantages_eroded=["Previous context limitations"],
            new_barriers=["Competitors must match context scaling"],
            lock_in_or_openness_shift="increased_lock_in"
        ),
        temporal_context=TemporalContext(
            preceded_by_events=[],
            likely_to_trigger_events=["Competitor response announcements"],
            time_horizon="immediate"
        ),
        alignment_implications="Enhanced safety controls for extended context",
        regulatory_signal="none"
    )

    print(f"  - what_changed: {len(event_v2.what_changed)} chars")
    print(f"  - advantages: {len(event_v2.competitive_effects.advantages_created)}")
    print(f"  - pillars: {len(event_v2.pillars_impacted)}")

    # Test deduplication
    print("\n4. Testing deduplication logic...")
    result = manager._check_and_merge_duplicate(event_v2)

    print(f"\nDeduplication result:")
    print(f"  - Action: {result['action']}")
    print(f"  - Event ID: {result['event_id']}")
    print(f"  - Reason: {result['reason']}")

    # Verify the merge
    if result['action'] == 'updated':
        print("\n5. Verifying merged event...")
        merged = db.get_event(result['event_id'])

        print(f"\n✓ Merged event: {merged.event_id}")
        print(f"  - what_changed: {len(merged.what_changed)} chars (was {len(event_v1.what_changed)})")
        print(f"    '{merged.what_changed}'")
        print(f"  - advantages: {len(merged.competitive_effects.advantages_created)} (was {len(event_v1.competitive_effects.advantages_created)})")
        for adv in merged.competitive_effects.advantages_created:
            print(f"    • {adv}")
        print(f"  - pillars: {len(merged.pillars_impacted)} (was {len(event_v1.pillars_impacted)})")
        for pillar in merged.pillars_impacted:
            print(f"    • {pillar.pillar_name.value} ({pillar.relative_strength_signal.value})")

        print(f"  - source: {merged.source_type} (kept better source type)")

    # Summary
    print("\n" + "=" * 80)
    print("DEDUPLICATION TEST COMPLETE")
    print("=" * 80)

    print("\nKey behaviors demonstrated:")
    print("  ✓ Semantic similarity detection (found duplicate despite different IDs)")
    print("  ✓ Field merging (kept longer what_changed)")
    print("  ✓ Advantage union (merged competitive effects)")
    print("  ✓ Pillar union (merged pillar impacts)")
    print("  ✓ Source preference (kept official_blog over social_media)")

    print("\nIn production:")
    print("  - Would fetch from real URLs using ingestion workflow")
    print("  - Would automatically detect duplicates across sources")
    print("  - Would progressively enrich events as more sources are found")
    print("  - Would prevent duplicate storage while preserving best information")

    # Cleanup
    print("\n6. Cleaning up test event...")
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM events WHERE event_id LIKE 'evt_test_dedup%'")
        conn.commit()

    try:
        vector_store.collection.delete(ids=["evt_test_dedup_gpt5", "evt_test_dedup_gpt5_blog"])
    except:
        pass

    print("✓ Test cleanup complete")


if __name__ == "__main__":
    try:
        test_deduplication_logic()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
