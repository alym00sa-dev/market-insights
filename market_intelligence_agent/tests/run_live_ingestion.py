"""
Run Live Ingestion

Fetches from all configured sources and stores events with deduplication.
Generates a detailed report of what was added.
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
from src.utils import ingest_live_sources


def run_live_ingestion():
    """
    Run full live ingestion from all configured sources.
    """
    print("=" * 80)
    print("LIVE SOURCE INGESTION")
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
    print("✓ Components initialized")

    # Get current state
    before_count = db.get_event_count()
    before_providers = db.get_all_providers()

    print(f"\n2. Current database state:")
    print(f"   Total events: {before_count}")
    print(f"   Providers: {', '.join(before_providers)}")

    # Show breakdown by provider
    print("\n   Events by provider:")
    for provider in before_providers:
        count = len(db.get_events_by_provider(provider, limit=1000))
        print(f"     - {provider}: {count}")

    # Run ingestion
    print("\n3. Starting live ingestion...")
    print("   (This will fetch from blogs and may take 2-5 minutes)")
    print()

    start_time = datetime.now()

    result = ingest_live_sources(
        database=db,
        vector_store=vector_store,
        llm_provider=llm,
        config=config,
        use_ensemble=False  # Use v1 for speed, set True for quality
    )

    duration = (datetime.now() - start_time).total_seconds()

    # Get new state
    after_count = db.get_event_count()
    after_providers = db.get_all_providers()

    # Summary
    print("\n" + "=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)

    print(f"\nDuration: {duration:.1f} seconds")
    print(f"\nResults:")
    print(f"   Total fetched: {result['total_fetched']}")
    print(f"   New events added: {result['new_events']}")
    print(f"   Existing events updated: {result['updated_events']}")
    print(f"   Duplicates skipped: {result['skipped_duplicates']}")

    if result['errors']:
        print(f"\n   Errors: {len(result['errors'])}")
        for error in result['errors']:
            print(f"     - {error}")

    print(f"\nDatabase state:")
    print(f"   Before: {before_count} events")
    print(f"   After: {after_count} events")
    print(f"   Net change: +{after_count - before_count} events")

    # Show what was added by provider
    if result['new_events'] > 0 or result['updated_events'] > 0:
        print("\n" + "-" * 80)
        print("NEW/UPDATED EVENTS BY PROVIDER")
        print("-" * 80)

        after_providers = db.get_all_providers()
        for provider in after_providers:
            after_provider_count = len(db.get_events_by_provider(provider, limit=1000))
            before_provider_count = len([e for e in db.get_events_by_provider(provider, limit=1000)
                                        if e.retrieved_at < start_time])

            if after_provider_count != before_provider_count:
                print(f"\n{provider}:")
                print(f"   Before: {before_provider_count} events")
                print(f"   After: {after_provider_count} events")
                print(f"   Change: +{after_provider_count - before_provider_count}")

                # Show sample new events
                recent_events = db.search_events(
                    provider=provider,
                    start_date=start_time,
                    limit=5
                )
                if recent_events:
                    print(f"   Sample new events:")
                    for event in recent_events[:3]:
                        print(f"     • {event.event_id}")
                        print(f"       {event.what_changed[:80]}...")

    # Save detailed report
    report = {
        'timestamp': datetime.now().isoformat(),
        'duration_seconds': duration,
        'before': {
            'total_events': before_count,
            'providers': before_providers
        },
        'after': {
            'total_events': after_count,
            'providers': after_providers
        },
        'ingestion_results': result
    }

    report_path = '/tmp/live_ingestion_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n✓ Detailed report saved to: {report_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if result['new_events'] > 0:
        print(f"\n✅ Added {result['new_events']} new events from live sources")

    if result['updated_events'] > 0:
        print(f"✅ Enriched {result['updated_events']} existing events with additional info")

    if result['skipped_duplicates'] > 0:
        print(f"⊙ Skipped {result['skipped_duplicates']} duplicates (already in database)")

    if result['new_events'] == 0 and result['updated_events'] == 0:
        print("\n⊙ No new events found - database already up to date!")
        print("   (This is expected if seed data is comprehensive)")

    print(f"\n📊 Total events now: {after_count}")
    print(f"📊 Providers covered: {len(after_providers)}")

    print("\n✨ Live ingestion complete! Your database is now up to date.")


if __name__ == "__main__":
    try:
        run_live_ingestion()
    except KeyboardInterrupt:
        print("\n\n⚠️  Ingestion cancelled by user")
    except Exception as e:
        print(f"\n✗ Ingestion failed with error: {e}")
        import traceback
        traceback.print_exc()
