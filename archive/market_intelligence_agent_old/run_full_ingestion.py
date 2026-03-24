"""
Full Ingestion Pipeline

Runs complete pipeline for the last 6 months:
1. Source Scout → discovers sources
2. Content Harvester V2 → filters noise
3. Ensemble Signal Extractor → extracts events + builds Provider Intelligence
4. Stores everything in database

Usage:
    python run_full_ingestion.py
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents import (
    SourceScout,
    SourceScoutMonitor,
    ContentHarvesterV2,
    EnsembleSignalExtractor,
    CompetitiveReasoning,
    HYPERSCALER_SOURCES,
    CROSS_PROVIDER_SOURCES,
    API_SOURCES
)
from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore
import yaml


def load_config():
    """Load configuration from YAML file."""
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_full_ingestion():
    """
    Run full ingestion pipeline for last 6 months.
    """
    print("\n" + "="*80)
    print("MARKET INTELLIGENCE AGENT - FULL INGESTION")
    print("="*80)
    print(f"\nStarting full ingestion: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Timeframe: Last 6 months")
    print("\n" + "="*80)

    # Load config
    print("\n[1/5] Loading configuration...")
    config_path = str(project_root / 'config' / 'config.yaml')
    config = load_config()
    print("✓ Configuration loaded")

    # Initialize components
    print("\n[2/5] Initializing components...")
    llm = LLMProvider(config_path)
    db = EventDatabase(config['storage']['database']['path'])
    vector_store = EventVectorStore(config['storage']['vector_store']['path'])

    # Initialize Content Harvester V2
    harvester = ContentHarvesterV2(llm, db, config)

    # Initialize Ensemble Signal Extractor (with Provider Intelligence)
    ensemble_extractor = EnsembleSignalExtractor(
        llm_provider=llm,
        database=db,
        config=config,
        enable_huggingface=False  # Disable HuggingFace for speed
    )

    # Initialize Source Scout (for web search)
    source_scout = SourceScout(llm, db, config)

    print("✓ Components initialized")
    print("  - Content Harvester V2 (noise filter)")
    print("  - Ensemble Signal Extractor (multi-LLM + Provider Intelligence)")
    print("  - Source Scout (for web search)")
    print("  - Vector Store (for semantic search)")

    # Get all sources to monitor
    print("\n[3/5] Loading sources...")
    all_sources = []
    for provider, sources in HYPERSCALER_SOURCES.items():
        all_sources.extend(sources)
        print(f"  - {provider}: {len(sources)} sources")

    # Add cross-provider and API sources
    all_sources.extend(CROSS_PROVIDER_SOURCES)
    all_sources.extend(API_SOURCES)
    print(f"  - Cross-provider (news): {len(CROSS_PROVIDER_SOURCES)} sources")
    print(f"  - API sources (Artificial Analysis): {len(API_SOURCES)} endpoints")

    print(f"\n✓ Total sources to monitor: {len(all_sources)}")

    # Statistics
    stats = {
        'sources_checked': 0,
        'content_harvested': 0,
        'events_extracted': 0,
        'noise_filtered': 0,
        'errors': 0
    }

    # Process each source
    print("\n[4/5] Processing sources...")
    print("-" * 80)

    for i, source in enumerate(all_sources, 1):
        print(f"\n[{i}/{len(all_sources)}] {source.provider} - {source.source_type}")
        print(f"URL: {source.url}")

        stats['sources_checked'] += 1

        try:
            # Handle different source types
            urls_to_harvest = []

            if source.needs_link_extraction:
                # Extract article links from homepage
                print("  Extracting article links...")
                from src.agents.source_scout import LinkExtractor
                extractor = LinkExtractor()

                article_links = extractor.extract_article_links(
                    source.url,
                    limit=10  # Get last 10 articles
                )

                if article_links:
                    print(f"  ✓ Found {len(article_links)} articles")
                    urls_to_harvest = article_links
                else:
                    print("  ⚠ No articles found")

            elif source.source_type == 'rss_feed':
                # Parse RSS feed
                print("  Parsing RSS feed...")
                import feedparser

                feed = feedparser.parse(source.url)
                if feed.entries:
                    urls_to_harvest = [entry.link for entry in feed.entries[:10]]
                    print(f"  ✓ Found {len(urls_to_harvest)} RSS entries")
                else:
                    print("  ⚠ No RSS entries found")
            else:
                # Direct URL
                urls_to_harvest = [source.url]

            # Harvest and extract from each URL
            for url in urls_to_harvest:
                print(f"\n  Processing: {url}")

                # Stage 1: Content Harvester V2 (noise filter)
                print("    [Harvester V2] Fetching and filtering...")
                content = harvester.harvest(
                    url=url,
                    provider=source.provider,
                    source_type=source.source_type
                )

                if not content:
                    print("    ✗ Filtered as noise or unchanged")
                    stats['noise_filtered'] += 1
                    continue

                print(f"    ✓ Non-noise content (category: {content.content_category})")
                stats['content_harvested'] += 1

                # Stage 2: Ensemble Signal Extractor (with Provider Intelligence)
                print("    [Ensemble Extractor] Extracting event...")
                event = ensemble_extractor.extract(
                    content=content.raw_text,
                    provider=source.provider,
                    source_url=url,
                    source_type=source.source_type,
                    published_at=content.published_at,
                    metadata=content.metadata,
                    parallel=False  # Sequential for stability
                )

                if event:
                    print(f"    ✓ Event extracted: {event.event_id}")

                    # Store event in database
                    success = db.create_event(event)
                    if success:
                        print("    ✓ Event stored in database")

                        # Add to vector store for semantic search
                        try:
                            vector_store.add_event(event)
                            print("    ✓ Event added to vector store")
                        except Exception as vs_error:
                            print(f"    ⚠ Vector store error: {vs_error}")

                        stats['events_extracted'] += 1
                    else:
                        print("    ⚠ Event already exists in database")
                else:
                    print("    ✗ Extraction failed")
                    stats['errors'] += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            stats['errors'] += 1
            continue

    # Web search for recent events (this past week)
    print("\n" + "="*80)
    print("WEB SEARCH FOR RECENT EVENTS (LAST 7 DAYS)")
    print("="*80)

    providers_to_search = ["OpenAI", "Anthropic", "Google", "Microsoft", "Meta"]

    for provider in providers_to_search:
        print(f"\n[Web Search] {provider}...")

        try:
            # Discover recent sources using web search
            discovered = source_scout.discover_sources(
                provider=provider,
                mode="automated",
                use_web_search=True,
                limit=5  # Top 5 recent results
            )

            if discovered:
                print(f"  ✓ Found {len(discovered)} recent sources")

                for candidate in discovered:
                    print(f"\n  Processing: {candidate.url}")

                    # Harvest content
                    print("    [Harvester V2] Fetching...")
                    content = harvester.harvest(
                        url=candidate.url,
                        provider=provider,
                        source_type=candidate.source_type
                    )

                    if not content:
                        print("    ✗ Filtered as noise or unchanged")
                        stats['noise_filtered'] += 1
                        continue

                    print(f"    ✓ Non-noise content (category: {content.content_category})")
                    stats['content_harvested'] += 1

                    # Extract event
                    print("    [Ensemble Extractor] Extracting...")
                    event = ensemble_extractor.extract(
                        content=content.raw_text,
                        provider=provider,
                        source_url=candidate.url,
                        source_type=candidate.source_type,
                        published_at=content.published_at,
                        metadata=content.metadata,
                        parallel=False
                    )

                    if event:
                        print(f"    ✓ Event extracted: {event.event_id}")
                        success = db.create_event(event)
                        if success:
                            print("    ✓ Event stored")

                            # Add to vector store
                            try:
                                vector_store.add_event(event)
                                print("    ✓ Event added to vector store")
                            except Exception as vs_error:
                                print(f"    ⚠ Vector store error: {vs_error}")

                            stats['events_extracted'] += 1
                        else:
                            print("    ⚠ Event already exists")
                    else:
                        print("    ✗ Extraction failed")
            else:
                print("  ⚠ No recent sources found")

        except Exception as e:
            print(f"  ✗ Web search error: {e}")
            stats['errors'] += 1
            continue

    # Print summary
    print("\n" + "="*80)
    print("INGESTION COMPLETE")
    print("="*80)
    print(f"\nStatistics:")
    print(f"  Sources checked: {stats['sources_checked']}")
    print(f"  Content harvested: {stats['content_harvested']}")
    print(f"  Events extracted: {stats['events_extracted']}")
    print(f"  Noise filtered: {stats['noise_filtered']}")
    print(f"  Errors: {stats['errors']}")

    # Provider Intelligence summary
    print(f"\n[5/5] Provider Intelligence Summary:")
    print("-" * 80)

    profiles_summary = ensemble_extractor.provider_intelligence.get_all_profiles_summary()

    if profiles_summary:
        for provider, profile_data in profiles_summary.items():
            print(f"\n{provider}:")
            print(f"  Total events: {profile_data['total_events']}")
            print(f"  Themes: {', '.join(profile_data['primary_themes'][:3])}")
            print(f"  Momentum: {profile_data['momentum']}")
            print(f"  Confidence: {profile_data['confidence']:.2f}")
    else:
        print("  No provider profiles built yet (need minimum 5 events per provider)")

    # Generate Key Findings Summary
    print("\n" + "="*80)
    print("KEY FINDINGS FROM LAST 6 MONTHS")
    print("="*80)

    # Initialize vector store and competitive reasoning for analysis
    try:
        vector_store = EventVectorStore(config['storage']['vector_store']['path'])
    except:
        vector_store = None
        print("\n⚠ Vector store not available, using database only for summary")

    reasoning = CompetitiveReasoning(
        llm_provider=llm,
        database=db,
        vector_store=vector_store,
        config=config,
        provider_intelligence=ensemble_extractor.provider_intelligence
    )

    # Get events from last 6 months and last 7 days
    six_months_ago = datetime.now() - timedelta(days=180)
    seven_days_ago = datetime.now() - timedelta(days=7)

    for provider in providers_to_search:
        print(f"\n{'='*80}")
        print(f"{provider.upper()}")
        print(f"{'='*80}")

        # Get all events for this provider in last 6 months
        all_events = db.search_events(
            provider=provider,
            start_date=six_months_ago,
            limit=100
        )

        # Get recent events (last 7 days)
        recent_events = [e for e in all_events if e.published_at >= seven_days_ago]

        if not all_events:
            print(f"  No events found for {provider}")
            continue

        print(f"\n📊 Overall Activity (6 months):")
        print(f"  Total events: {len(all_events)}")

        # Get provider profile
        profile = ensemble_extractor.provider_intelligence.get_profile(provider)
        if profile:
            print(f"\n🎯 Strategic Profile:")
            print(f"  Themes: {', '.join(profile.primary_themes[:3]) if profile.primary_themes else 'N/A'}")
            print(f"  Direction: {profile.strategic_direction[:100]}..." if len(profile.strategic_direction) > 100 else f"  Direction: {profile.strategic_direction}")
            print(f"  Momentum: {profile.momentum}")

            print(f"\n💪 Pillar Strengths:")
            sorted_pillars = sorted(
                profile.pillar_strengths.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            for pillar, strength in sorted_pillars:
                print(f"  {pillar}: {strength:.1f}/10")

        # Show recent events (this past week)
        if recent_events:
            print(f"\n🔥 This Past Week ({len(recent_events)} events):")
            for event in recent_events[:5]:  # Show top 5
                print(f"\n  📅 {event.published_at.strftime('%Y-%m-%d')}:")
                print(f"     {event.what_changed[:150]}...")
                if event.pillars_impacted:
                    pillars = ', '.join([p.pillar_name.value for p in event.pillars_impacted[:3]])
                    print(f"     Pillars: {pillars}")
        else:
            print(f"\n  No events from this past week")

        # Show key events (6 months)
        print(f"\n📌 Key Events (6 months):")
        # Get events with strong pillar impacts
        strong_events = [
            e for e in all_events
            if any(p.relative_strength_signal.value == "STRONG" for p in e.pillars_impacted)
        ][:5]

        if strong_events:
            for event in strong_events[:3]:
                print(f"\n  📅 {event.published_at.strftime('%Y-%m-%d')}:")
                print(f"     {event.what_changed[:150]}...")
        else:
            # Show most recent events instead
            for event in all_events[:3]:
                print(f"\n  📅 {event.published_at.strftime('%Y-%m-%d')}:")
                print(f"     {event.what_changed[:150]}...")

    print("\n" + "="*80)
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print("\n✓ Ready to test with Streamlit app!")
    print("  Run: streamlit run app.py")


if __name__ == "__main__":
    try:
        run_full_ingestion()
    except KeyboardInterrupt:
        print("\n\n⚠ Ingestion interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
