"""
Test Full Pipeline: Source Scout → Content Harvester → Signal Extractor

This tests the three-agent pipeline working together.
"""

import sys
import os
import yaml
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore, ProviderMemory
from src.agents import SourceScout, ContentHarvester, SignalExtractor


def test_full_pipeline():
    """
    Test full pipeline from source discovery to event extraction.
    """
    print("=" * 80)
    print("FULL PIPELINE TEST")
    print("=" * 80)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    print("\n1. Initializing components...")
    llm = LLMProvider(config_path)  # LLM takes path
    db = EventDatabase(config['storage']['database']['path'])  # Database takes path string

    # Initialize agents (they all take llm, db, config dict)
    scout = SourceScout(llm, db, config)
    harvester = ContentHarvester(llm, db, config)
    extractor = SignalExtractor(llm, db, config)

    print("✓ Components initialized")

    # Step 1: Discover sources
    print("\n2. Discovering sources for OpenAI...")
    sources = scout.discover_sources(
        provider="OpenAI",
        mode="curated",  # Use curated for faster test
        limit=3
    )

    print(f"✓ Found {len(sources)} sources:")
    for i, source in enumerate(sources, 1):
        print(f"   {i}. {source.url} ({source.source_type}, priority: {source.priority})")

    if not sources:
        print("✗ No sources found. Check config.yaml default_sources.")
        return

    # Step 2: Harvest content from blog source (more likely to have significant content)
    blog_sources = [s for s in sources if s.source_type == 'official_blog']

    if blog_sources:
        print(f"\n3. Harvesting content from blog: {blog_sources[0].url}")
        content = harvester.harvest(
            url=blog_sources[0].url,
            provider="OpenAI",
            source_type=blog_sources[0].source_type
        )
    else:
        print(f"\n3. Harvesting content from: {sources[0].url}")
        content = harvester.harvest(
            url=sources[0].url,
            provider="OpenAI",
            source_type=sources[0].source_type
        )

    if not content:
        print("✗ Content harvesting failed or content not significant enough")
        print("   This is expected for some sources (e.g., if nothing new published)")
        print("   The blog might not have new content, or the content might be low-significance")
        # Try alternative source
        print("\n   Trying alternative source...")
        for source in sources[1:]:
            if source.source_type != 'official_blog':
                continue
            print(f"   Trying: {source.url}")
            content = harvester.harvest(
                url=source.url,
                provider=source.provider if hasattr(source, 'provider') else "OpenAI",
                source_type=source.source_type
            )
            if content:
                break

    if not content:
        print("✗ No content available for extraction")
        return

    print("✓ Content harvested:")
    print(f"   Significance: {content.significance_score}/10")
    print(f"   Type: {content.content_type}")
    print(f"   Content length: {len(content.filtered_content)} chars")
    print(f"   Preview: {content.filtered_content[:200]}...")

    # Step 3: Extract event
    print(f"\n4. Extracting Market Signal Event...")
    event = extractor.extract(
        content=content.filtered_content,
        provider=content.provider,
        source_url=content.url,
        source_type=content.source_type,
        published_at=content.fetched_at,
        metadata=content.metadata
    )

    if not event:
        print("✗ Event extraction failed")
        return

    print("✓ Event extracted successfully!")
    print(f"\n   Event ID: {event.event_id}")
    print(f"   Provider: {event.provider}")
    print(f"   What Changed: {event.what_changed}")
    print(f"   Why It Matters: {event.why_it_matters}")
    print(f"   Pillars Impacted: {len(event.pillars_impacted)}")
    for pillar in event.pillars_impacted:
        print(f"      - {pillar.pillar_name.value}: {pillar.direction_of_change.value} ({pillar.relative_strength_signal.value})")
    print(f"   Competitive Effects:")
    print(f"      - Advantages Created: {len(event.competitive_effects.advantages_created)}")
    print(f"      - Advantages Eroded: {len(event.competitive_effects.advantages_eroded)}")
    print(f"      - New Barriers: {len(event.competitive_effects.new_barriers)}")

    # Step 4: Store event
    print(f"\n5. Storing event in database...")
    db.create_event(event)
    print("✓ Event stored")

    # Verify retrieval
    retrieved = db.get_event(event.event_id)
    if retrieved and retrieved['event_id'] == event.event_id:
        print("✓ Event retrieval verified")
    else:
        print("✗ Event retrieval failed")

    print("\n" + "=" * 80)
    print("PIPELINE TEST COMPLETE")
    print("=" * 80)
    print("\nSummary:")
    print(f"  Sources discovered: {len(sources)}")
    print(f"  Content harvested: {content.url}")
    print(f"  Event extracted: {event.event_id}")
    print(f"  Event stored: ✓")


if __name__ == "__main__":
    try:
        test_full_pipeline()
    except Exception as e:
        print(f"\n✗ Pipeline test failed with error: {e}")
        import traceback
        traceback.print_exc()
