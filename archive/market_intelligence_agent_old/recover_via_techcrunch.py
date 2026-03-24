"""
Recover Failed Events via TechCrunch

Strategy: Search TechCrunch for each failed event, extract from their coverage.
TechCrunch is scrapable and covers all major AI announcements.

Usage:
    python recover_via_techcrunch.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.agents import ContentHarvesterV2, EnsembleSignalExtractor
from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore
import yaml


# The 52 failed events
FAILED_EVENTS = [
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
    'evt_openai_frontier_20260205',
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


def parse_event_id(event_id):
    """Extract provider and description from event_id."""
    parts = event_id.replace('evt_', '').split('_')
    provider = parts[0].title()
    description = ' '.join(parts[1:-1]).replace('_', ' ')
    return provider, description


def recover_via_techcrunch():
    """
    Recover events by finding TechCrunch coverage.
    """
    print("\n" + "="*80)
    print("RECOVERING EVENTS VIA TECHCRUNCH")
    print("="*80)
    print(f"\nRecovering {len(FAILED_EVENTS)} events from TechCrunch coverage\n")

    # Load config
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize
    print("Initializing...")
    llm = LLMProvider(str(config_path))
    db = EventDatabase(config['storage']['database']['path'])
    vector_store = EventVectorStore(config['storage']['vector_store']['path'])

    harvester = ContentHarvesterV2(llm, db, config)
    extractor = EnsembleSignalExtractor(llm, db, config, enable_huggingface=False)
    print("✓ Initialized\n")

    stats = {'recovered': 0, 'not_found': 0, 'errors': 0}

    for i, event_id in enumerate(FAILED_EVENTS, 1):
        provider, description = parse_event_id(event_id)
        search_query = f"{provider} {description} TechCrunch"

        print(f"[{i}/{len(FAILED_EVENTS)}] {provider}: {description}")
        print(f"  Searching: {search_query}")

        try:
            # Ask LLM to find TechCrunch URL
            search_prompt = f"""Find the TechCrunch article URL about: {provider} {description}

Return ONLY the TechCrunch URL. Format: https://techcrunch.com/YYYY/MM/DD/article-title/

If no article exists, return: NONE

URL:"""

            response = llm.generate(
                messages=[
                    {"role": "system", "content": "You find TechCrunch article URLs. Return ONLY the URL or NONE."},
                    {"role": "user", "content": search_prompt}
                ],
                task_complexity="simple",
                temperature=0.1
            )

            url = response['content'].strip()

            # Clean URL
            if '```' in url:
                url = url.split('```')[0]
            url = url.strip().split('\n')[0].split()[0]

            if 'NONE' in url.upper() or not url.startswith('http'):
                print(f"  ✗ No TechCrunch article found")
                stats['not_found'] += 1
                continue

            print(f"  Found: {url}")

            # Harvest
            content = harvester.harvest(
                url=url,
                provider=provider,
                source_type='news_article'
            )

            if not content:
                print(f"  ✗ Failed to harvest")
                stats['not_found'] += 1
                continue

            print(f"  ✓ Harvested")

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
                print(f"  ✓ Extracted: {event.event_id}")

                # Store
                success = db.create_event(event)
                if success:
                    print(f"  ✓ Stored")

                    try:
                        vector_store.add_event(event)
                        print(f"  ✓ Added to vector store")
                    except Exception as vs_err:
                        print(f"  ⚠ Vector store: {vs_err}")

                    stats['recovered'] += 1
                else:
                    print(f"  ⚠ Already exists")
                    stats['recovered'] += 1
            else:
                print(f"  ✗ Extraction failed")
                stats['errors'] += 1

        except Exception as e:
            print(f"  ✗ Error: {e}")
            stats['errors'] += 1

        print()  # Blank line

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

    print(f"\nFinal database: {total} events")
    print(f"Vector store: {vector_store.count_events()} events")


if __name__ == "__main__":
    try:
        recover_via_techcrunch()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
