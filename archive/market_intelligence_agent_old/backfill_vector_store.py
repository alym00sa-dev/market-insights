"""
Backfill Vector Store

Adds all events from the database to the vector store.
Run this after fixing the ingestion pipeline to populate vector store with existing events.

Usage:
    python backfill_vector_store.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.storage import EventDatabase, EventVectorStore
import yaml


def backfill_vector_store():
    """
    Read all events from database and add them to vector store.
    """
    print("\n" + "="*80)
    print("BACKFILL VECTOR STORE")
    print("="*80)

    # Load config
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    print("\nInitializing...")
    db = EventDatabase(config['storage']['database']['path'])
    vector_store = EventVectorStore(config['storage']['vector_store']['path'])

    # Get all events from database
    print("Fetching all events from database...")
    all_events = db.search_events(limit=10000)  # Get all events

    print(f"Found {len(all_events)} events in database")

    # Check current vector store count
    current_count = vector_store.count_events()
    print(f"Current vector store count: {current_count}")

    # Add events to vector store
    print(f"\nAdding {len(all_events)} events to vector store...")

    added = 0
    skipped = 0
    errors = 0

    for i, event in enumerate(all_events, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(all_events)} ({added} added, {skipped} skipped, {errors} errors)")

        try:
            # Try to add event
            vector_store.add_event(event)
            added += 1
        except Exception as e:
            # Check if event already exists
            if "already exists" in str(e).lower() or "unique" in str(e).lower():
                skipped += 1
            else:
                print(f"  Error adding {event.event_id}: {e}")
                errors += 1

    print(f"\n✓ Backfill complete!")
    print(f"  Added: {added}")
    print(f"  Skipped (already exist): {skipped}")
    print(f"  Errors: {errors}")

    # Final count
    final_count = vector_store.count_events()
    print(f"\nFinal vector store count: {final_count}")
    print("="*80)


if __name__ == "__main__":
    try:
        backfill_vector_store()
    except KeyboardInterrupt:
        print("\n\n⚠ Backfill interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
