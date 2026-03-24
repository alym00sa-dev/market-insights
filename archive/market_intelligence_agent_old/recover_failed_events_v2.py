"""
Recover Failed Events V2

Simpler approach: Use WebSearch directly to find announcements,
then extract with ensemble.

Usage:
    python recover_failed_events_v2.py
"""

import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents import ContentHarvesterV2, EnsembleSignalExtractor
from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore
import yaml


# Critical events to recover (focusing on most recent/important)
PRIORITY_EVENTS = [
    {'search': 'OpenAI Frontier launch February 2026', 'provider': 'OpenAI'},
    {'search': 'OpenAI GPT-5.3 Codex February 2026', 'provider': 'OpenAI'},
    {'search': 'OpenAI GPT-5.2 Codex January 2026', 'provider': 'OpenAI'},
    {'search': 'Anthropic Agent Teams February 2026', 'provider': 'Anthropic'},
    {'search': 'Anthropic Cowork January 2026', 'provider': 'Anthropic'},
    {'search': 'Anthropic MCP December 2025', 'provider': 'Anthropic'},
    {'search': 'Microsoft Agent 365 February 2026', 'provider': 'Microsoft'},
    {'search': 'Microsoft Ignite 2025 November', 'provider': 'Microsoft'},
    {'search': 'Google Vertex Agent Engine GA December 2025', 'provider': 'Google'},
    {'search': 'Google Deep Research upgrade December 2025', 'provider': 'Google'},
]


def recover_with_web_search():
    """
    Recover events using direct web search.
    """
    print("\n" + "="*80)
    print("RECOVERING PRIORITY EVENTS (V2 - Direct Web Search)")
    print("="*80)
    print(f"\nRecovering {len(PRIORITY_EVENTS)} critical events\n")

    # Load config
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize
    print("Initializing...")
    llm = LLMProvider(str(config_path))
    db = EventDatabase(config['storage']['database']['path'])
    vector_store = EventVectorStore(config['storage']['vector_store']['path'])

    harvester = ContentHarvesterV2(llm, db, config)
    ensemble_extractor = EnsembleSignalExtractor(
        llm_provider=llm,
        database=db,
        config=config,
        enable_huggingface=False
    )
    print("✓ Initialized\n")

    stats = {'recovered': 0, 'not_found': 0, 'errors': 0}

    for i, event_info in enumerate(PRIORITY_EVENTS, 1):
        print(f"[{i}/{len(PRIORITY_EVENTS)}] Searching: {event_info['search']}")

        try:
            # Use LLM to search web and get URL
            search_prompt = f"""Find the official announcement URL for: {event_info['search']}

Return ONLY the official blog post or documentation URL. No explanation.

Example formats:
- https://openai.com/blog/...
- https://www.anthropic.com/news/...
- https://cloud.google.com/blog/...
- https://devblogs.microsoft.com/...

URL:"""

            response = llm.generate(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that finds official announcement URLs."},
                    {"role": "user", "content": search_prompt}
                ],
                task_complexity="simple",
                temperature=0.1
            )

            url = response['content'].strip()

            # Clean URL (remove markdown, extra text, etc.)
            if '```' in url:
                url = url.split('```')[0]
            url = url.strip().split('\n')[0].split()[0]

            if not url.startswith('http'):
                print(f"  ✗ Invalid URL: {url}")
                stats['not_found'] += 1
                continue

            print(f"  Found: {url}")

            # Harvest content
            print(f"  Harvesting...")
            content = harvester.harvest(
                url=url,
                provider=event_info['provider'],
                source_type='official_blog'
            )

            if not content:
                print(f"  ✗ Failed to harvest or filtered as noise")
                stats['not_found'] += 1
                continue

            print(f"  ✓ Content harvested")

            # Extract with ensemble
            print(f"  Extracting...")
            event = ensemble_extractor.extract(
                content=content.raw_text,
                provider=event_info['provider'],
                source_url=url,
                source_type='official_blog',
                published_at=content.published_at,
                metadata=content.metadata,
                parallel=False
            )

            if event:
                print(f"  ✓ Event extracted: {event.event_id}")

                # Store
                success = db.create_event(event)
                if success:
                    print(f"  ✓ Stored in database")

                    try:
                        vector_store.add_event(event)
                        print(f"  ✓ Added to vector store")
                    except Exception as vs_err:
                        print(f"  ⚠ Vector store error: {vs_err}")

                    stats['recovered'] += 1
                else:
                    print(f"  ⚠ Already exists")
                    stats['recovered'] += 1  # Count as recovered
            else:
                print(f"  ✗ Extraction failed")
                stats['errors'] += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            stats['errors'] += 1
            continue

        print()  # Blank line between events

    # Summary
    print("="*80)
    print("RECOVERY COMPLETE")
    print("="*80)
    print(f"\nResults:")
    print(f"  Recovered: {stats['recovered']}")
    print(f"  Not found: {stats['not_found']}")
    print(f"  Errors: {stats['errors']}")

    import sqlite3
    conn = sqlite3.connect(config['storage']['database']['path'])
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM events')
    total = cursor.fetchone()[0]
    conn.close()

    print(f"\nFinal database count: {total} events")
    print(f"Vector store count: {vector_store.count_events()} events")


if __name__ == "__main__":
    try:
        recover_with_web_search()
    except KeyboardInterrupt:
        print("\n\n⚠ Recovery interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
