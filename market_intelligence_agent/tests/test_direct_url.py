"""
Test with a known good URL to verify the extraction pipeline works end-to-end.
"""

import sys
import os
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore
from src.utils import ingest_live_sources


def test_direct_url():
    """Test with a known good Anthropic announcement URL."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    llm = LLMProvider(config_path)
    db = EventDatabase(config['storage']['database']['path'])
    vector_store = EventVectorStore(
        persist_directory=config['storage']['vector_store']['path'],
        collection_name=config['storage']['vector_store']['collection_name']
    )

    # Test with a specific known URL (from our earlier web search)
    test_url = 'https://ai.meta.com/blog/llama-4-multimodal-intelligence/'  # Llama 4 announcement

    print(f"Testing direct URL ingestion: {test_url}")
    print("=" * 80)

    result = ingest_live_sources(
        database=db,
        vector_store=vector_store,
        llm_provider=llm,
        config=config,
        sources=[test_url],  # Provide URL directly, skip discovery
        use_ensemble=False
    )

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Total fetched: {result['total_fetched']}")
    print(f"New events: {result['new_events']}")
    print(f"Updated events: {result['updated_events']}")
    print(f"Duplicates skipped: {result['skipped_duplicates']}")

    if result['errors']:
        print(f"\nErrors: {len(result['errors'])}")
        for error in result['errors']:
            print(f"  - {error}")

    if result['new_events'] > 0 or result['updated_events'] > 0:
        print("\n✅ SUCCESS! Extraction pipeline works end-to-end")
    else:
        print("\n⊙ No events extracted (might already be in database)")


if __name__ == "__main__":
    try:
        test_direct_url()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
