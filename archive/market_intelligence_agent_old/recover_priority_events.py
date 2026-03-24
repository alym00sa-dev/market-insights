"""
Recover Priority Events

Simple script: Use SourceScout's Tavily search to find news articles,
then extract the most important recent events.

Usage:
    python recover_priority_events.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents import SourceScout, ContentHarvesterV2, EnsembleSignalExtractor
from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore
import yaml


# Most critical recent events to recover
PRIORITY_EVENTS = [
    {'provider': 'Anthropic', 'search': 'Anthropic MCP Model Context Protocol November 2025'},
    {'provider': 'Anthropic', 'search': 'Anthropic Computer Use Claude October 2024'},
    {'provider': 'Google', 'search': 'Google Vertex Agent Engine GA December 2025'},
    {'provider': 'Google', 'search': 'Google Deep Research December 2025'},
    {'provider': 'Microsoft', 'search': 'Microsoft Ignite 2025 AI announcements'},
    {'provider': 'Meta', 'search': 'Meta Code Llama August 2023'},
]


def recover_priority():
    """
    Recover priority events using SourceScout's Tavily search.
    """
    print("\n" + "="*80)
    print("RECOVERING PRIORITY EVENTS")
    print("="*80)
    print(f"\nRecovering {len(PRIORITY_EVENTS)} critical events via Tavily\n")

    # Load config
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize
    print("Initializing...")
    llm = LLMProvider(str(config_path))
    db = EventDatabase(config['storage']['database']['path'])
    vector_store = EventVectorStore(config['storage']['vector_store']['path'])

    source_scout = SourceScout(llm, db, config)
    harvester = ContentHarvesterV2(llm, db, config)
    extractor = EnsembleSignalExtractor(llm, db, config, enable_huggingface=False)

    if not source_scout.tavily_client:
        print("✗ Tavily API key not found! Set TAVILY_API_KEY environment variable.")
        return

    print("✓ Initialized (Tavily enabled)\n")

    stats = {'recovered': 0, 'not_found': 0, 'errors': 0}

    for i, event_info in enumerate(PRIORITY_EVENTS, 1):
        provider = event_info['provider']
        search_query = event_info['search']

        print(f"[{i}/{len(PRIORITY_EVENTS)}] {provider}")
        print(f"  Searching: {search_query}")

        try:
            # Use Tavily to search
            results = source_scout.tavily_client.search(
                query=search_query,
                max_results=3,
                include_domains=["techcrunch.com", "venturebeat.com", "theverge.com"]
            )

            if not results or not results.get('results'):
                print(f"  ✗ No results")
                stats['not_found'] += 1
                continue

            print(f"  ✓ Found {len(results['results'])} articles")

            # Try each result
            extracted = False
            for result in results['results']:
                url = result.get('url', '')
                print(f"    Trying: {url}")

                # Harvest
                content = harvester.harvest(url=url, provider=provider, source_type='news_article')
                if not content:
                    print(f"      ✗ Harvest failed")
                    continue

                print(f"      ✓ Harvested")

                # Extract
                event = extractor.extract(
                    content=content.raw_text,
                    provider=provider,
                    source_url=url,
                    source_type='news_article',
                    published_at=content.published_at,
                    metadata=content.metadata,
                    parallel=False
                )

                if event:
                    print(f"      ✓ Extracted: {event.event_id}")
                    success = db.create_event(event)
                    if success:
                        print(f"      ✓ Stored")
                        try:
                            vector_store.add_event(event)
                            print(f"      ✓ Vector store")
                        except:
                            pass
                        stats['recovered'] += 1
                        extracted = True
                        break
                    else:
                        print(f"      ⚠ Already exists")
                        extracted = True
                        break

            if not extracted:
                stats['not_found'] += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            stats['errors'] += 1

        print()

    # Summary
    print("="*80)
    print(f"Recovered: {stats['recovered']}, Not found: {stats['not_found']}, Errors: {stats['errors']}")

    import sqlite3
    conn = sqlite3.connect(config['storage']['database']['path'])
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM events')
    print(f"Total: {cursor.fetchone()[0]} events")
    conn.close()


if __name__ == "__main__":
    try:
        recover_priority()
    except KeyboardInterrupt:
        print("\n⚠ Interrupted")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
