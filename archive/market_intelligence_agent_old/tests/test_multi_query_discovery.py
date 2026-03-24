"""
Test Multi-Query Discovery with Scoring

Tests the new multi-query Tavily approach:
1. Run 5+ queries per provider
2. Aggregate results (deduplicate, track multi-query hits)
3. Score with Harvester (relevance + recency + significance + multi-query bonus)
4. Compare against existing DB
5. Return validated, high-scoring URLs
"""

import sys
import os
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore
from src.utils import LiveIngestionManager


def test_multi_query_discovery():
    """
    Test multi-query discovery for one provider.
    """
    print("=" * 80)
    print("TEST: MULTI-QUERY DISCOVERY + AGGREGATION + SCORING")
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

    # Test with one provider
    test_provider = 'OpenAI'

    print(f"\n{'=' * 80}")
    print(f"Testing with: {test_provider}")
    print(f"{'=' * 80}")

    if not manager.tavily_client:
        print("\n⚠️  Tavily API key not found!")
        print("Set TAVILY_API_KEY in your .env file to test discovery")
        return

    # Run discovery
    validated_sources = manager.discover_and_validate_sources(test_provider)

    # Final summary
    print(f"\n{'=' * 80}")
    print(f"FINAL RESULT")
    print(f"{'=' * 80}")

    if validated_sources:
        print(f"\n✅ Validated {len(validated_sources)} sources ready for extraction:")
        for i, source in enumerate(validated_sources, 1):
            print(f"\n{i}. {source['title']}")
            print(f"   URL: {source['url']}")
            print(f"   Score: {source['combined_score']:.1f}/10")
            print(f"   Components:")
            print(f"     - Tavily relevance: {source['tavily_score']:.1f}")
            print(f"     - Recency: {source['recency_score']:.1f}")
            print(f"     - Significance: {source['significance_score']:.1f}")
            print(f"     - Multi-query bonus: {source['multi_query_bonus']:.1f}")
            print(f"   Found by: {source['num_queries']} queries")
            print(f"   Categories: {', '.join(set(source['categories']))}")
    else:
        print("\n⊙ No new sources found")
        print("   This means either:")
        print("   - No recent blog posts in the last 30 days")
        print("   - All discovered content is already in the database")
        print("   - Scores were below threshold (< 5.0/10)")

    print("\n" + "=" * 80)
    print("SYSTEM OVERVIEW")
    print("=" * 80)
    print("\nMulti-Query Discovery Pipeline:")
    print("  1. Check RSS feed freshness (validation baseline)")
    print("  2. Run 5+ Tavily queries per provider")
    print("     - Model releases")
    print("     - API/platform updates")
    print("     - Partnerships")
    print("     - Safety/alignment")
    print("     - Infrastructure")
    print("  3. Aggregate URLs (deduplicate, track multi-query hits)")
    print("  4. Score each URL (0-10 points):")
    print("     - Tavily relevance (0-3)")
    print("     - Recency (0-3)")
    print("     - Harvester significance (0-3)")
    print("     - Multi-query bonus (+1 if found by 2+ queries)")
    print("  5. Filter by threshold (min 5.0/10)")
    print("  6. Check against existing DB (skip if already covered)")
    print("  7. Return validated, high-scoring sources")

    print("\nAdvantages:")
    print("  ✓ Multi-angle coverage (5+ queries per provider)")
    print("  ✓ Quality filtering (only high-scoring content)")
    print("  ✓ Smart deduplication (compare before extracting)")
    print("  ✓ Confidence signals (multi-query hits = high confidence)")
    print("  ✓ Reduced noise (harvester pre-filters significance)")


if __name__ == "__main__":
    try:
        test_multi_query_discovery()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
