"""
Test Multi-Query Discovery with Anthropic
"""

import sys
import os
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore
from src.utils import LiveIngestionManager


def test_anthropic():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    llm = LLMProvider(config_path)
    db = EventDatabase(config['storage']['database']['path'])
    vector_store = EventVectorStore(
        persist_directory=config['storage']['vector_store']['path'],
        collection_name=config['storage']['vector_store']['collection_name']
    )
    manager = LiveIngestionManager(db, vector_store, llm, config)

    print("Testing with Anthropic...")
    validated_sources = manager.discover_and_validate_sources('Anthropic')

    print(f"\n{'=' * 80}")
    print(f"FINAL RESULT: {len(validated_sources)} validated sources")
    if validated_sources:
        for i, source in enumerate(validated_sources[:3], 1):
            print(f"\n{i}. {source['title']}")
            print(f"   Score: {source['combined_score']:.1f}/10")
            print(f"   URL: {source['url']}")


if __name__ == "__main__":
    try:
        test_anthropic()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
