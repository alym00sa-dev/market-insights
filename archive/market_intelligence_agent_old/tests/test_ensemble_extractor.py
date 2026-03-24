"""
Test Ensemble Signal Extractor (v2)

Compares v1 (standard) vs v2 (ensemble) extraction on same content.
"""

import sys
import os
import yaml
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm import LLMProvider
from src.storage import EventDatabase
from src.agents import SignalExtractor, EnsembleSignalExtractor


# Sample content for testing
SAMPLE_CONTENT = """
Anthropic Launches Claude Code: Developer-Focused AI Assistant

San Francisco, January 2025 - Anthropic today announced Claude Code, a new developer-focused AI assistant optimized for software engineering workflows. Built on Claude Sonnet 4.5, Claude Code introduces several capabilities designed specifically for professional developers.

Key Features:

**Agentic Code Editing**
Claude Code can autonomously navigate codebases, read multiple files, make coordinated edits across files, and run tests to verify changes. Unlike traditional code completion tools, it can plan and execute multi-step refactoring operations.

**Terminal Integration**
Direct terminal access allows Claude Code to run build commands, execute tests, install dependencies, and interact with development tools like git, docker, and package managers. This enables end-to-end workflow automation within a single session.

**Context-Aware Assistance**
The assistant maintains awareness of project structure, dependencies, and recent changes across conversation turns. It can reference documentation, stack traces, and test outputs to provide contextual debugging assistance.

**Safety & Sandboxing**
All code execution happens in isolated sandboxed environments with user approval required for commands that modify files or execute privileged operations. This balances automation with developer control.

Pricing & Availability:
- CLI tool: Free and open source (Apache 2.0 license)
- API access: $3/hour for Pro users, $0.50/hour for Free tier
- Team plan: Volume discounts for organizations

Market Context:
This positions Anthropic directly against GitHub Copilot (Microsoft/OpenAI), Cursor (Anysphere), and emerging AI-powered IDEs. The focus on agentic capabilities (multi-file edits, tool use) rather than just autocomplete represents a strategic bet on higher-level AI assistance.

According to Anthropic CEO Dario Amodei: "Claude Code represents our vision for AI as a collaborative coding partner, not just a suggestion engine. By giving Claude real agency over the development environment, we enable it to handle entire features, not just individual functions."

The open-source CLI approach also addresses lock-in concerns that have slowed enterprise adoption of proprietary coding assistants. Developers can inspect, modify, and integrate Claude Code into existing workflows without vendor dependencies.

Early beta users report 40% reduction in time spent on routine refactoring tasks and 60% fewer context switches between editor and documentation. However, the model still requires human oversight for architectural decisions and complex debugging.
"""


def test_ensemble_vs_standard():
    """
    Compare v1 standard extraction vs v2 ensemble extraction.
    """
    print("=" * 80)
    print("V1 vs V2 EXTRACTOR COMPARISON")
    print("=" * 80)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    print("\n1. Initializing components...")
    llm = LLMProvider(config_path)
    db = EventDatabase(config['storage']['database']['path'])

    # Initialize both extractors
    v1_extractor = SignalExtractor(llm, db, config)
    v2_extractor = EnsembleSignalExtractor(llm, db, config, enable_huggingface=False)  # Disable HF for speed
    print("✓ Components initialized")

    # Test metadata
    test_metadata = {
        "title": "Anthropic Launches Claude Code",
        "content_type": "product_announcement"
    }

    # Test v1 extraction
    print("\n" + "=" * 80)
    print("V1 STANDARD EXTRACTION (Single LLM)")
    print("=" * 80)

    v1_start = datetime.now()
    v1_event = v1_extractor.extract(
        content=SAMPLE_CONTENT,
        provider="Anthropic",
        source_url="https://anthropic.com/news/claude-code",
        source_type="official_blog",
        published_at=datetime(2025, 1, 20),
        metadata=test_metadata
    )
    v1_time = (datetime.now() - v1_start).total_seconds()

    if v1_event:
        print(f"\n✓ V1 extraction complete ({v1_time:.2f}s)")
        print(f"\nEvent ID: {v1_event.event_id}")
        print(f"Pillars: {len(v1_event.pillars_impacted)}")
        for p in v1_event.pillars_impacted:
            print(f"   - {p.pillar_name.value}: {p.direction_of_change.value} ({p.relative_strength_signal.value})")
        print(f"Advantages Created: {len(v1_event.competitive_effects.advantages_created)}")
        print(f"Advantages Eroded: {len(v1_event.competitive_effects.advantages_eroded)}")
    else:
        print("✗ V1 extraction failed")

    # Test v2 ensemble extraction
    print("\n" + "=" * 80)
    print("V2 ENSEMBLE EXTRACTION (Multiple LLMs + Aggregation)")
    print("=" * 80)

    v2_start = datetime.now()
    v2_event = v2_extractor.extract(
        content=SAMPLE_CONTENT,
        provider="Anthropic",
        source_url="https://anthropic.com/news/claude-code",
        source_type="official_blog",
        published_at=datetime(2025, 1, 20),
        metadata=test_metadata,
        parallel=False  # Sequential for clearer output
    )
    v2_time = (datetime.now() - v2_start).total_seconds()

    if v2_event:
        print(f"\n✓ V2 extraction complete ({v2_time:.2f}s)")
        print(f"\nEvent ID: {v2_event.event_id}")
        print(f"Pillars: {len(v2_event.pillars_impacted)}")
        for p in v2_event.pillars_impacted:
            print(f"   - {p.pillar_name.value}: {p.direction_of_change.value} ({p.relative_strength_signal.value})")
        print(f"Advantages Created: {len(v2_event.competitive_effects.advantages_created)}")
        print(f"Advantages Eroded: {len(v2_event.competitive_effects.advantages_eroded)}")
    else:
        print("✗ V2 extraction failed")

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    if v1_event and v2_event:
        print(f"\nExtraction Time:")
        print(f"   V1: {v1_time:.2f}s")
        print(f"   V2: {v2_time:.2f}s ({v2_time/v1_time:.1f}x slower)")

        print(f"\nPillars Identified:")
        v1_pillars = set(p.pillar_name.value for p in v1_event.pillars_impacted)
        v2_pillars = set(p.pillar_name.value for p in v2_event.pillars_impacted)
        print(f"   V1: {sorted(v1_pillars)}")
        print(f"   V2: {sorted(v2_pillars)}")
        print(f"   Agreement: {v1_pillars & v2_pillars}")
        print(f"   V1 only: {v1_pillars - v2_pillars}")
        print(f"   V2 only: {v2_pillars - v1_pillars}")

        print(f"\nCompetitive Effects:")
        print(f"   V1 advantages created: {len(v1_event.competitive_effects.advantages_created)}")
        print(f"   V2 advantages created: {len(v2_event.competitive_effects.advantages_created)}")
        print(f"   V1 advantages eroded: {len(v1_event.competitive_effects.advantages_eroded)}")
        print(f"   V2 advantages eroded: {len(v2_event.competitive_effects.advantages_eroded)}")

        print(f"\nWhat Changed (comparison):")
        print(f"   V1 length: {len(v1_event.what_changed)} chars")
        print(f"   V2 length: {len(v2_event.what_changed)} chars")
        print(f"\n   V1: {v1_event.what_changed[:200]}...")
        print(f"\n   V2: {v2_event.what_changed[:200]}...")

        # Metrics
        print(f"\nV2 Ensemble Metrics:")
        metrics = v2_extractor.get_metrics_summary()
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2%}" if value < 2 else f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    # Summary
    print("\nKey Observations:")
    print("- V2 is slower (multiple LLM calls + aggregation) but may produce higher quality")
    print("- V2 provides consensus metrics (useful for research)")
    print("- V1 is more cost-effective for routine extractions")
    print("- V2 is valuable for high-stakes events or validation")


if __name__ == "__main__":
    try:
        test_ensemble_vs_standard()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
