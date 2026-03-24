"""
Recover Failed Events

Re-extract the 52 events that failed during ensemble aggregation.
These were real events that got stored with empty JSON.

Strategy:
1. Identify the source URLs that would have these events
2. Re-run ensemble extraction on those sources
3. Store properly this time

Usage:
    python recover_failed_events.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents import (
    ContentHarvesterV2,
    EnsembleSignalExtractor,
    SourceScout
)
from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore
import yaml


# The 52 corrupted event IDs that need to be recovered
FAILED_EVENT_IDS = [
    'evt_openai_custom_gpts_20231106',
    'evt_openai_function_calling_20230613',
    'evt_openai_code_interpreter_20230706',
    'evt_anthropic_projects_20240620',
    'evt_anthropic_computer_use_20241022',
    'evt_anthropic_mcp_20241125',
    'evt_anthropic_artifacts_20240620',
    'evt_anthropic_extended_thinking_20250522',
    'evt_microsoft_azure_agent_service_20241119',
    'evt_microsoft_semantic_kernel_20230317',
    'evt_microsoft_autogen_20230925',
    'evt_meta_code_llama_20230824',
    'evt_google_vertex_agent_builder_20240409',
    'evt_openai_frontier_20260205',  # THIS IS THE ONE YOU ASKED ABOUT!
    'evt_openai_gpt53_codex_20260205',
    'evt_openai_gpt52_codex_20260114',
    'evt_openai_responses_api_tools_20251200',
    'evt_openai_rft_custom_tools_20251200',
    'evt_openai_agentkit_20251006',
    'evt_openai_agents_sdk_20251006',
    'evt_openai_agent_builder_20251006',
    'evt_openai_connector_registry_20251006',
    'evt_openai_chatgpt_agent_mode_20250717',
    'evt_anthropic_agent_teams_20260205',
    'evt_anthropic_cowork_plugins_20260130',
    'evt_anthropic_cowork_20260112',
    'evt_anthropic_agent_skills_standard_20251218',
    'evt_anthropic_enterprise_skills_20251218',
    'evt_anthropic_mcp_donation_20251209',
    'evt_anthropic_mcp_updates_20251100',
    'evt_anthropic_mcp_apps_20251121',
    'evt_anthropic_claude_code_web_20251020',
    'evt_anthropic_agent_skills_launch_20251016',
    'evt_google_vertex_agent_engine_ga_20251216',
    'evt_google_deep_research_upgrade_20251211',
    'evt_google_workspace_studio_20251203',
    'evt_google_antigravity_ide_20251118',
    'evt_google_adk_go_support_20251107',
    'evt_google_vertex_agent_designer_20250900',
    'evt_google_cloud_api_registry_20250900',
    'evt_microsoft_agent_365_preview_20260200',
    'evt_microsoft_ignite_2025_20251100',
    'evt_microsoft_foundry_iq_20251100',
    'evt_microsoft_fabric_iq_20251100',
    'evt_microsoft_gpt5_copilot_studio_ga_20251124',
    'evt_microsoft_agent_framework_preview_20251000',
    'evt_microsoft_copilot_pages_20250900',
    'evt_microsoft_multiagent_orchestration_20250519',
    'evt_microsoft_autogen_v04_20250114',
    'evt_meta_llamafirewall_20250429',
    'evt_meta_llama_guard_4_20250429',
    'evt_meta_llama_stack_dist_20250400',
]


def extract_search_terms_from_event_ids():
    """
    Extract search terms from event IDs to find the sources.

    For example:
    - evt_openai_frontier_20260205 → search for "OpenAI Frontier"
    - evt_anthropic_mcp_20241125 → search for "Anthropic MCP"
    """
    search_queries = []

    for event_id in FAILED_EVENT_IDS:
        # Parse event_id: evt_<provider>_<description>_<date>
        parts = event_id.replace('evt_', '').split('_')

        # Extract provider (first part)
        provider = parts[0].title()

        # Extract description (everything except last part which is date)
        description_parts = parts[1:-1]  # Exclude provider and date
        description = ' '.join(description_parts).replace('_', ' ')

        # Create search query
        search_query = f"{provider} {description}"
        search_queries.append({
            'event_id': event_id,
            'provider': provider,
            'description': description,
            'search_query': search_query
        })

    return search_queries


def recover_events():
    """
    Recover the failed events by searching for their sources and re-extracting.
    """
    print("\n" + "="*80)
    print("RECOVERING FAILED EVENTS")
    print("="*80)
    print(f"\nAttempting to recover {len(FAILED_EVENT_IDS)} events that failed during aggregation")

    # Load config
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    print("\nInitializing components...")
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
    source_scout = SourceScout(llm, db, config)

    print("✓ Components initialized")

    # Extract search terms
    search_queries = extract_search_terms_from_event_ids()

    print(f"\n\nGenerated {len(search_queries)} search queries")
    print("\nFirst 10 queries:")
    for i, query in enumerate(search_queries[:10], 1):
        print(f"  {i}. {query['search_query']} ({query['event_id']})")

    print(f"\n{'='*80}")
    print("Starting web search and extraction...")
    print(f"{'='*80}\n")

    stats = {
        'recovered': 0,
        'not_found': 0,
        'errors': 0
    }

    for query_info in search_queries:
        print(f"\n[{stats['recovered'] + stats['not_found'] + stats['errors'] + 1}/{len(search_queries)}] Searching: {query_info['search_query']}")
        print(f"  Original event_id: {query_info['event_id']}")

        try:
            # Use web search to find the source
            discovered = source_scout.discover_sources(
                provider=query_info['provider'],
                mode="automated",
                use_web_search=True,
                search_query=query_info['search_query'],
                limit=3  # Top 3 results
            )

            if not discovered:
                print("  ✗ No sources found")
                stats['not_found'] += 1
                continue

            print(f"  ✓ Found {len(discovered)} potential sources")

            # Try to extract from the first promising source
            for candidate in discovered[:1]:  # Try the top result
                print(f"  Processing: {candidate.url}")

                # Harvest content
                content = harvester.harvest(
                    url=candidate.url,
                    provider=query_info['provider'],
                    source_type=candidate.source_type
                )

                if not content:
                    print("    ✗ Filtered as noise")
                    continue

                print("    ✓ Content harvested")

                # Extract with ensemble (this time it should work!)
                event = ensemble_extractor.extract(
                    content=content.raw_text,
                    provider=query_info['provider'],
                    source_url=candidate.url,
                    source_type=candidate.source_type,
                    published_at=content.published_at,
                    metadata=content.metadata,
                    parallel=False
                )

                if event:
                    print(f"    ✓ Event extracted: {event.event_id}")

                    # Store in database
                    success = db.create_event(event)
                    if success:
                        print("    ✓ Stored in database")

                        # Add to vector store
                        try:
                            vector_store.add_event(event)
                            print("    ✓ Added to vector store")
                        except Exception as vs_error:
                            print(f"    ⚠ Vector store error: {vs_error}")

                        stats['recovered'] += 1
                        break  # Success, move to next event
                    else:
                        print("    ⚠ Event already exists")
                else:
                    print("    ✗ Extraction failed")

            if not event:
                stats['not_found'] += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            stats['errors'] += 1
            continue

    # Print summary
    print(f"\n{'='*80}")
    print("RECOVERY COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults:")
    print(f"  Recovered: {stats['recovered']}")
    print(f"  Not found: {stats['not_found']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Total: {len(FAILED_EVENT_IDS)}")

    # Final counts
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
        recover_events()
    except KeyboardInterrupt:
        print("\n\n⚠ Recovery interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
