"""
Test Signal Extractor Agent

Tests event extraction from sample content without needing live sources.
"""

import sys
import os
import yaml
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm import LLMProvider
from src.storage import EventDatabase
from src.agents import SignalExtractor


# Sample content for testing
SAMPLE_CONTENT = """
OpenAI Announces Context Caching for API

Today, we're introducing context caching for the OpenAI API. This feature allows developers to cache frequently used context (like system prompts, long documents, or conversation history) for up to 1 hour, reducing latency and costs by up to 90% for cached tokens.

Key features:
- Automatic caching of prompts over 1,024 tokens
- Cached content persists for 1 hour
- 90% cost reduction on cached input tokens
- Available for GPT-4o and GPT-4o-mini models
- No code changes required - automatic detection

Pricing:
- Regular input tokens: $0.0025/1K tokens
- Cached input tokens: $0.00025/1K tokens (90% discount)
- Output tokens: unchanged

This feature is particularly valuable for:
- RAG applications with large knowledge bases
- Agents with extensive system prompts
- Multi-turn conversations with persistent context
- Code assistants with project-wide context

Technical implementation:
- Server-side caching with automatic eviction
- Cache key based on content hash
- Compatible with all existing API parameters
- Gradual rollout over the next 2 weeks

Context caching joins our portfolio of optimization features including batch processing and streaming responses, giving developers more tools to build cost-effective AI applications.
"""


def test_signal_extractor():
    """
    Test Signal Extractor with sample content.
    """
    print("=" * 80)
    print("SIGNAL EXTRACTOR TEST")
    print("=" * 80)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize components
    print("\n1. Initializing components...")
    llm = LLMProvider(config_path)
    db = EventDatabase(config['storage']['database']['path'])
    extractor = SignalExtractor(llm, db, config)
    print("✓ Components initialized")

    # Test extraction
    print("\n2. Extracting event from sample content...")
    print(f"   Content length: {len(SAMPLE_CONTENT)} chars")
    print(f"   Content preview: {SAMPLE_CONTENT[:150]}...")

    event = extractor.extract(
        content=SAMPLE_CONTENT,
        provider="OpenAI",
        source_url="https://openai.com/blog/context-caching",
        source_type="official_blog",
        published_at=datetime(2025, 1, 15),
        metadata={
            "title": "OpenAI Announces Context Caching for API",
            "content_type": "product_announcement"
        }
    )

    if not event:
        print("✗ Event extraction failed")
        print("\n   This could be due to:")
        print("   - LLM API error")
        print("   - Content validation failure")
        print("   - Confidence below threshold")
        return

    print("✓ Event extracted successfully!")

    # Display results
    print(f"\n{'='*80}")
    print("EXTRACTED EVENT")
    print(f"{'='*80}")

    print(f"\nEvent ID: {event.event_id}")
    print(f"Provider: {event.provider}")
    print(f"Source: {event.source_url}")
    print(f"Published: {event.published_at.strftime('%Y-%m-%d')}")

    print(f"\nWhat Changed:")
    print(f"  {event.what_changed}")

    print(f"\nWhy It Matters:")
    print(f"  {event.why_it_matters}")

    print(f"\nPillars Impacted ({len(event.pillars_impacted)}):")
    for pillar in event.pillars_impacted:
        print(f"  - {pillar.pillar_name.value}")
        print(f"    Direction: {pillar.direction_of_change.value}")
        print(f"    Strength: {pillar.relative_strength_signal.value}")
        print(f"    Evidence: {pillar.evidence[:100]}...")

    print(f"\nCompetitive Effects:")
    if event.competitive_effects.advantages_created:
        print(f"  Advantages Created:")
        for adv in event.competitive_effects.advantages_created:
            print(f"    - {adv}")

    if event.competitive_effects.advantages_eroded:
        print(f"  Advantages Eroded:")
        for adv in event.competitive_effects.advantages_eroded:
            print(f"    - {adv}")

    if event.competitive_effects.new_barriers:
        print(f"  New Barriers:")
        for barrier in event.competitive_effects.new_barriers:
            print(f"    - {barrier}")

    print(f"\n  Lock-in/Openness: {event.competitive_effects.lock_in_or_openness_shift}")

    print(f"\nTemporal Context:")
    print(f"  Time Horizon: {event.temporal_context.time_horizon}")
    if event.temporal_context.preceded_by_events:
        print(f"  Preceded by: {len(event.temporal_context.preceded_by_events)} event(s)")
    if event.temporal_context.likely_to_trigger_events:
        print(f"  Likely to trigger: {len(event.temporal_context.likely_to_trigger_events)} event(s)")

    print(f"\nAlignment Implications:")
    print(f"  {event.alignment_implications}")

    print(f"\nRegulatory Signal: {event.regulatory_signal}")

    # Test storage
    print(f"\n{'='*80}")
    print("STORAGE TEST")
    print(f"{'='*80}")

    print("\n3. Storing event in database...")
    db.create_event(event)
    print("✓ Event stored")

    # Verify retrieval
    print("\n4. Retrieving event from database...")
    retrieved = db.get_event(event.event_id)

    if retrieved:
        # retrieved is a MarketSignalEvent object
        if retrieved.event_id == event.event_id:
            print("✓ Event retrieval verified")
            print(f"   Retrieved: {retrieved.what_changed[:100]}...")
        else:
            print("✗ Event retrieval failed - wrong event returned")
    else:
        print("✗ Event retrieval failed - no event found")

    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        test_signal_extractor()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
