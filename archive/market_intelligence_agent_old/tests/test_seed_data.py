"""
Test Seed Data Loader

Loads events from market_demo.html into the database.
"""

import sys
import os
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.storage import EventDatabase, EventVectorStore
from src.utils import load_seed_data


def test_seed_data():
    """
    Test loading seed data from HTML.
    """
    print("=" * 80)
    print("SEED DATA LOADER TEST")
    print("=" * 80)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize storage
    print("\n1. Initializing storage...")
    db = EventDatabase(config['storage']['database']['path'])
    vector_store = EventVectorStore(
        persist_directory=config['storage']['vector_store']['path'],
        collection_name=config['storage']['vector_store']['collection_name']
    )
    print("✓ Storage initialized")

    # Check current event count
    before_count = db.get_event_count()
    print(f"\nEvents in database before loading: {before_count}")

    # Load seed data
    html_path = os.path.join(os.path.dirname(__file__), '..', 'market_demo.html')

    if not os.path.exists(html_path):
        print(f"\n✗ market_demo.html not found at: {html_path}")
        print("Please ensure market_demo.html is in the project root.")
        return

    print(f"\n2. Loading seed data from: {html_path}")

    result = load_seed_data(
        html_path=html_path,
        database=db,
        vector_store=vector_store,
        use_llm_enrichment=False  # Fast mode for testing
    )

    # Summary
    print("\n" + "=" * 80)
    print("SEED DATA LOADING COMPLETE")
    print("=" * 80)

    after_count = db.get_event_count()

    print(f"\nEvents before: {before_count}")
    print(f"Events after: {after_count}")
    print(f"Events created: {result['events_created']}")

    if result['errors']:
        print(f"\nErrors encountered: {len(result['errors'])}")
        for error in result['errors'][:5]:
            print(f"  - {error}")

    # Show sample events
    print("\n" + "-" * 80)
    print("SAMPLE EVENTS")
    print("-" * 80)

    providers = db.get_all_providers()
    print(f"\nProviders: {', '.join(providers)}")

    for provider in providers[:3]:
        events = db.get_events_by_provider(provider, limit=2)
        print(f"\n{provider} ({len(events)} sample events):")
        for event in events:
            print(f"  - {event.event_id}")
            print(f"    {event.what_changed[:80]}...")

    # Verify MVP criteria
    print("\n" + "-" * 80)
    print("MVP CRITERIA CHECK")
    print("-" * 80)

    total_events = db.get_event_count()
    num_providers = len(db.get_all_providers())

    print(f"\n✓ Total events: {total_events} (requirement: ≥30)")
    print(f"✓ Providers: {num_providers} (requirement: ≥3)")

    if total_events >= 30 and num_providers >= 3:
        print("\n🎉 MVP data requirements MET!")
    else:
        print("\n⚠️  MVP data requirements not yet met")

    # Test queries
    print("\n" + "-" * 80)
    print("DATA VALIDATION")
    print("-" * 80)

    # Check pillar distribution
    from src.models import Pillar

    print("\nEvents by Pillar:")
    for pillar in Pillar:
        count = len(db.get_events_by_pillar(pillar, limit=1000))
        print(f"  - {pillar.value}: {count} events")

    # Check temporal distribution
    print("\nEvents by Year:")
    for year in [2023, 2024, 2025]:
        from datetime import datetime
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)
        count = len(db.search_events(start_date=start, end_date=end, limit=1000))
        print(f"  - {year}: {count} events")


if __name__ == "__main__":
    try:
        test_seed_data()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
