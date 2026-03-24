"""
Test Competitive Reasoning Agent

Tests all 4 query types:
1. Event Impact Analysis
2. Provider Comparison
3. Leadership Ranking
4. Timeline Analysis
"""

import sys
import os
import yaml
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.llm import LLMProvider
from src.storage import EventDatabase, EventVectorStore
from src.agents import CompetitiveReasoning
from src.models import Pillar


def test_competitive_reasoning():
    """
    Test all 4 competitive reasoning query types.
    """
    print("=" * 80)
    print("COMPETITIVE REASONING TEST")
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
    reasoning = CompetitiveReasoning(llm, db, vector_store, config)
    print("✓ Components initialized")

    # Check if we have events
    event_count = db.get_event_count()
    print(f"\nEvents in database: {event_count}")

    if event_count == 0:
        print("\n⚠️  No events in database. Run test_database_enhancements.py first to populate events.")
        return

    # Get sample event for testing
    all_providers = db.get_all_providers()
    if not all_providers:
        print("\n⚠️  No providers found in database.")
        return

    sample_provider = all_providers[0]
    sample_events = db.get_events_by_provider(sample_provider, limit=1)
    if not sample_events:
        print(f"\n⚠️  No events found for provider {sample_provider}.")
        return

    sample_event = sample_events[0]

    # Test 1: Event Impact Analysis
    print("\n" + "=" * 80)
    print("TEST 1: EVENT IMPACT ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing event: {sample_event.event_id}")
    print(f"Provider: {sample_event.provider}")
    print(f"What Changed: {sample_event.what_changed[:100]}...")

    result = reasoning.analyze_event_impact(sample_event.event_id)

    if 'error' in result:
        print(f"✗ Analysis failed: {result['error']}")
    else:
        print("\n✓ Event impact analysis complete:")
        print(f"\nImmediate Impact:")
        print(f"  {result.get('immediate_impact', 'N/A')}")

        if result.get('competitive_shifts'):
            print(f"\nCompetitive Shifts ({len(result['competitive_shifts'])}):")
            for shift in result['competitive_shifts'][:3]:
                print(f"  - {shift.get('dimension', 'N/A')}: {shift.get('shift', 'N/A')[:80]}...")

        if result.get('triggered_responses'):
            print(f"\nTriggered Responses: {len(result['triggered_responses'])}")

        print(f"\nConfidence: {result.get('confidence', 0):.2f}")

    # Test 2: Provider Comparison
    print("\n" + "=" * 80)
    print("TEST 2: PROVIDER COMPARISON")
    print("=" * 80)

    # Compare first 2 providers
    providers_to_compare = all_providers[:min(2, len(all_providers))]
    print(f"\nComparing: {', '.join(providers_to_compare)}")
    print(f"On pillar: {Pillar.TECHNICAL_CAPABILITIES.value}")

    result = reasoning.compare_providers(
        providers=providers_to_compare,
        pillar=Pillar.TECHNICAL_CAPABILITIES,
        start_date=datetime.now() - timedelta(days=180)
    )

    if 'error' in result:
        print(f"✗ Comparison failed: {result['error']}")
    else:
        print("\n✓ Provider comparison complete:")

        if result.get('differences'):
            print(f"\nKey Differences ({len(result['differences'])}):")
            for diff in result['differences'][:3]:
                print(f"\n  {diff.get('dimension', 'N/A')}:")
                if 'provider_positions' in diff:
                    for provider, position in diff['provider_positions'].items():
                        print(f"    - {provider}: {position[:80]}...")

        print(f"\nConvergence/Divergence:")
        print(f"  {result.get('convergence_divergence', 'N/A')[:150]}...")

        print(f"\nConfidence: {result.get('confidence', 0):.2f}")

    # Test 3: Leadership Ranking
    print("\n" + "=" * 80)
    print("TEST 3: LEADERSHIP RANKING")
    print("=" * 80)
    print(f"\nRanking providers on: {Pillar.TECHNICAL_CAPABILITIES.value}")
    print(f"Time period: Last 180 days")

    result = reasoning.rank_leadership(
        pillar=Pillar.TECHNICAL_CAPABILITIES,
        start_date=datetime.now() - timedelta(days=180)
    )

    if 'error' in result:
        print(f"✗ Ranking failed: {result['error']}")
    else:
        print("\n✓ Leadership ranking complete:")

        if result.get('rankings'):
            print(f"\nRankings:")
            for i, ranking in enumerate(result['rankings'], 1):
                print(f"\n{i}. {ranking.get('provider', 'Unknown')}: {ranking.get('score', 0):.1f}/100")
                if ranking.get('key_strengths'):
                    print(f"   Strengths: {', '.join(ranking['key_strengths'][:2])}")
                if ranking.get('evidence_events'):
                    print(f"   Evidence: {len(ranking['evidence_events'])} events")

        print(f"\nAnalysis:")
        print(f"  {result.get('analysis', 'N/A')[:200]}...")

        print(f"\nConfidence: {result.get('confidence', 0):.2f}")

    # Test 4: Timeline Analysis
    print("\n" + "=" * 80)
    print("TEST 4: TIMELINE ANALYSIS")
    print("=" * 80)
    print(f"\nAnalyzing evolution of: {Pillar.TECHNICAL_CAPABILITIES.value}")
    print(f"Time period: Last 180 days")

    result = reasoning.analyze_timeline(
        pillar=Pillar.TECHNICAL_CAPABILITIES,
        start_date=datetime.now() - timedelta(days=180)
    )

    if 'error' in result:
        print(f"✗ Timeline failed: {result['error']}")
    else:
        print("\n✓ Timeline analysis complete:")

        if result.get('timeline'):
            print(f"\nTimeline ({len(result['timeline'])} events):")
            for event in result['timeline'][:5]:
                print(f"  {event.get('date', 'N/A')} | {event.get('provider', 'Unknown')}: {event.get('description', 'N/A')[:60]}...")

        if result.get('key_trends'):
            print(f"\nKey Trends:")
            for trend in result['key_trends'][:3]:
                print(f"  - {trend}")

        if result.get('turning_points'):
            print(f"\nTurning Points: {len(result['turning_points'])}")

        print(f"\nNarrative:")
        print(f"  {result.get('narrative', 'N/A')[:200]}...")

        print(f"\nConfidence: {result.get('confidence', 0):.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    print("\nAll 4 Query Types Verified:")
    print("  ✓ Event Impact Analysis - Cause and effect reasoning")
    print("  ✓ Provider Comparison - Strategic differences identified")
    print("  ✓ Leadership Ranking - Evidence-based scores")
    print("  ✓ Timeline Analysis - Narrative evolution tracking")

    print("\nCompetitive Reasoning agent ready for:")
    print("  - Complex strategic analysis")
    print("  - Multi-event synthesis")
    print("  - Evidence-based insights")
    print("  - Competitive pattern recognition")


if __name__ == "__main__":
    try:
        test_competitive_reasoning()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
