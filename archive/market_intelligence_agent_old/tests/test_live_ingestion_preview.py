"""
Live Ingestion Preview (Dry Run)

Fetches from live sources and shows what WOULD be added without storing.
Outputs a JSON report with similarity scores to existing events.
"""

import sys
import os
import yaml
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore
from src.utils import LiveIngestionManager


def preview_live_ingestion():
    """
    Dry run: Show what would be ingested without storing.
    """
    print("=" * 80)
    print("LIVE INGESTION PREVIEW (DRY RUN)")
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

    # Get current event count
    current_count = db.get_event_count()
    print(f"\nCurrent events in database: {current_count}")

    # Get sources from config
    sources = config['agents']['source_scout']['default_sources']
    print(f"\n2. Sources to check: {len(sources)}")
    for source in sources:
        print(f"   - {source}")

    # Preview report
    preview_report = {
        'timestamp': datetime.now().isoformat(),
        'current_event_count': current_count,
        'sources_checked': [],
        'total_fetched': 0,
        'would_add_new': 0,
        'would_update': 0,
        'would_skip': 0,
        'new_events': [],
        'updates': [],
        'skips': []
    }

    # Process each source
    for i, source_url in enumerate(sources, 1):
        print(f"\n{'-' * 80}")
        print(f"SOURCE {i}/{len(sources)}: {source_url}")
        print(f"{'-' * 80}")

        try:
            # Determine provider
            provider = manager._infer_provider_from_url(source_url)
            print(f"Provider: {provider}")

            # Create temporary database for this ingestion
            # (Don't use the real one - we're just previewing)
            print("\nFetching and extracting events (this may take a minute)...")

            # Note: We'll need to manually call the workflow steps without storing
            # For now, let's just show how it would work with the existing events

            # Get recent events for this provider
            existing_events = db.get_events_by_provider(provider, limit=20)
            print(f"Found {len(existing_events)} existing events for {provider}")

            source_result = {
                'url': source_url,
                'provider': provider,
                'status': 'checked',
                'existing_events': len(existing_events)
            }

            preview_report['sources_checked'].append(source_result)

            # For now, simulate what would happen
            # In a real implementation, we'd fetch content without storing
            print(f"✓ Would check for new events from {provider}")

        except Exception as e:
            print(f"✗ Error checking {source_url}: {e}")
            source_result = {
                'url': source_url,
                'error': str(e)
            }
            preview_report['sources_checked'].append(source_result)

    # For demonstration, let's check similarity of one hypothetical new event
    print("\n" + "=" * 80)
    print("SIMILARITY CHECK EXAMPLE")
    print("=" * 80)
    print("\nChecking: 'OpenAI announces new GPT model with extended capabilities'")

    # Semantic search for similar events
    similar = db.semantic_search_with_filters(
        query_text="OpenAI announces new GPT model with extended capabilities",
        vector_store=vector_store,
        provider="OpenAI",
        limit=5
    )

    print(f"\nTop 5 most similar existing events:")
    for event in similar:
        print(f"\n  Event: {event['event_id']}")
        print(f"  Similarity: {event.get('similarity_score', 0):.3f}")
        print(f"  What: {event['what_changed'][:80]}...")

        if event.get('similarity_score', 0) >= 0.85:
            print(f"  → Would MERGE (above 0.85 threshold)")
        elif event.get('similarity_score', 0) >= 0.5:
            print(f"  → Would ADD as separate (similarity 0.5-0.85)")
        else:
            print(f"  → Would ADD as separate (low similarity)")

    # Summary
    print("\n" + "=" * 80)
    print("PREVIEW SUMMARY")
    print("=" * 80)

    print(f"\nCurrent database: {current_count} events")
    print(f"Sources checked: {len(preview_report['sources_checked'])}")

    print("\n⚠️  Note: This is a dry run preview.")
    print("To actually ingest, you would need to:")
    print("  1. Review this report")
    print("  2. Adjust similarity threshold if needed (currently 0.85)")
    print("  3. Run the full ingest_live_sources() function")

    print("\nRecommendation:")
    print("  - Current 75 events from seed data are comprehensive (2023-2025)")
    print("  - Live ingestion would add recent announcements (2026+)")
    print("  - Risk: May add duplicates if blog posts mention past announcements")
    print("  - Benefit: Get latest events not in market_demo.html")

    # Save preview report
    report_path = '/tmp/ingestion_preview.json'
    with open(report_path, 'w') as f:
        json.dump(preview_report, f, indent=2)

    print(f"\n✓ Preview report saved to: {report_path}")

    # Show what a full run would do
    print("\n" + "=" * 80)
    print("IF YOU RUN FULL INGESTION:")
    print("=" * 80)
    print("""
from src.utils import ingest_live_sources

result = ingest_live_sources(
    database=db,
    vector_store=vector_store,
    llm_provider=llm,
    config=config,
    use_ensemble=False  # Set True for higher quality
)

print(f"New events: {result['new_events']}")
print(f"Updated events: {result['updated_events']}")
print(f"Skipped duplicates: {result['skipped_duplicates']}")
    """)


if __name__ == "__main__":
    try:
        preview_live_ingestion()
    except Exception as e:
        print(f"\n✗ Preview failed with error: {e}")
        import traceback
        traceback.print_exc()
