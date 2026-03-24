"""
Ingest model data from Artificial Analysis API into the database.

Uses the Market Intelligence Glossary to add proper context to benchmarks
and technical specifications.
"""

import json
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage import EventDatabase, EventVectorStore
from src.models import (MarketSignalEvent, PillarImpact, CompetitiveEffects, Pillar,
                        DirectionOfChange, RelativeStrength, TemporalContext)


class GlossaryContextualizer:
    """Load and use glossary to add context to benchmarks and metrics."""

    def __init__(self, glossary_path: str):
        """Load glossary from YAML file."""
        with open(glossary_path) as f:
            self.glossary = yaml.safe_load(f)

    def get_benchmark_context(self, benchmark_name: str, score: float) -> str:
        """Get contextualized description of a benchmark score."""
        benchmarks = self.glossary.get('benchmarks', {})

        if benchmark_name not in benchmarks:
            return f"{benchmark_name}: {score}"

        bench = benchmarks[benchmark_name]

        # Determine performance level
        good_looks_like = bench.get('what_good_looks_like', '')

        # Extract thresholds (simplified parsing)
        if 'excellent' in good_looks_like.lower() and 'strong' in good_looks_like.lower():
            if '>0.' in good_looks_like or '>' in good_looks_like:
                # Try to extract numeric thresholds
                import re
                numbers = re.findall(r'>(\d+\.?\d*)', good_looks_like)
                if len(numbers) >= 2:
                    excellent_threshold = float(numbers[0])
                    strong_threshold = float(numbers[1])

                    if score >= excellent_threshold:
                        level = "excellent"
                    elif score >= strong_threshold:
                        level = "strong"
                    else:
                        level = "needs improvement"
                else:
                    level = "measured"
            else:
                level = "measured"
        else:
            level = "measured"

        full_name = bench.get('full_name', benchmark_name)
        why_matters = bench.get('why_it_matters', '')

        return f"{full_name}: {score:.3f} ({level}) - {why_matters}"

    def get_performance_context(self, metric_name: str, value: float) -> str:
        """Get contextualized description of a performance metric."""
        metrics = self.glossary.get('performance_metrics', {})

        if metric_name not in metrics:
            return f"{metric_name}: {value}"

        metric = metrics[metric_name]
        description = metric.get('description', '')
        why_matters = metric.get('why_it_matters', '')

        return f"{description} Value: {value}. {why_matters}"


def filter_to_major_models(models: List[Dict], max_per_provider: int = 5) -> List[Dict]:
    """Filter to most important models from major providers."""

    # Major providers we care about
    major_providers = {
        'openai': 'OpenAI',
        'anthropic': 'Anthropic',
        'google': 'Google',
        'meta': 'Meta',
        'microsoft': 'Microsoft'
    }

    # Group by provider
    by_provider = {}
    for model in models:
        provider_slug = model.get('model_creator', {}).get('slug', '').lower()

        if provider_slug in major_providers:
            if provider_slug not in by_provider:
                by_provider[provider_slug] = []
            by_provider[provider_slug].append(model)

    # Take top N models per provider (by intelligence index)
    filtered = []
    for provider_slug, provider_models in by_provider.items():
        # Sort by intelligence index
        sorted_models = sorted(
            provider_models,
            key=lambda m: m.get('evaluations', {}).get('artificial_analysis_intelligence_index', 0) or 0,
            reverse=True
        )

        # Take top N
        filtered.extend(sorted_models[:max_per_provider])

    return filtered


def create_model_event(model_data: Dict, contextualizer: GlossaryContextualizer) -> MarketSignalEvent:
    """Create a MarketSignalEvent for a model with contextualized benchmarks."""

    # Extract model info
    model_name = model_data.get('name', 'Unknown')
    provider_name = model_data.get('model_creator', {}).get('name', 'Unknown')
    release_date_str = model_data.get('release_date')

    # Parse release date
    try:
        release_date = datetime.strptime(release_date_str, '%Y-%m-%d')
    except:
        release_date = datetime.now()

    # Extract evaluations/benchmarks
    evals = model_data.get('evaluations', {})
    pricing = model_data.get('pricing', {})

    # Build contextualized benchmark descriptions
    benchmark_descriptions = []
    key_benchmarks = [
        'artificial_analysis_intelligence_index',
        'artificial_analysis_coding_index',
        'artificial_analysis_math_index',
        'mmlu_pro',
        'gpqa',
        'livecodebench',
        'aime_25'
    ]

    for bench_name in key_benchmarks:
        score = evals.get(bench_name)
        if score is not None:
            context = contextualizer.get_benchmark_context(bench_name, score)
            benchmark_descriptions.append(context)

    # Performance metrics
    output_speed = model_data.get('median_output_tokens_per_second')
    ttft = model_data.get('median_time_to_first_token_seconds')

    performance_desc = []
    if output_speed:
        perf_context = contextualizer.get_performance_context('output_tokens_per_second', output_speed)
        performance_desc.append(perf_context)
    if ttft:
        ttft_context = contextualizer.get_performance_context('time_to_first_token', ttft)
        performance_desc.append(ttft_context)

    # Pricing
    price_input = pricing.get('price_1m_input_tokens')
    price_output = pricing.get('price_1m_output_tokens')
    pricing_desc = f"Pricing: ${price_input}/1M input tokens, ${price_output}/1M output tokens" if price_input and price_output else ""

    # Build what_changed
    what_changed = f"{provider_name} model '{model_name}' benchmarked with the following capabilities:\n\n"
    what_changed += "**Key Benchmarks:**\n"
    what_changed += "\n".join(f"- {desc}" for desc in benchmark_descriptions[:5])  # Top 5

    if performance_desc:
        what_changed += "\n\n**Performance:**\n"
        what_changed += "\n".join(f"- {desc}" for desc in performance_desc)

    if pricing_desc:
        what_changed += f"\n\n**{pricing_desc}**"

    # Build why_it_matters
    intelligence_idx = evals.get('artificial_analysis_intelligence_index', 0)
    coding_idx = evals.get('artificial_analysis_coding_index', 0)

    why_it_matters = f"This model represents {provider_name}'s current capabilities in the competitive landscape. "

    if intelligence_idx and intelligence_idx > 30:
        why_it_matters += f"With an Intelligence Index of {intelligence_idx}, this represents top-tier performance. "
    elif intelligence_idx and intelligence_idx > 20:
        why_it_matters += f"With an Intelligence Index of {intelligence_idx}, this represents strong competitive performance. "

    if coding_idx and coding_idx > 25:
        why_it_matters += f"Coding Index of {coding_idx} indicates excellent developer tool capabilities. "

    if output_speed and output_speed > 200:
        why_it_matters += f"High output speed ({output_speed:.0f} tokens/sec) enables real-time applications. "

    why_it_matters += "Benchmark data enables competitive comparison across providers."

    # Create event ID
    event_id = f"evt_{provider_name.lower()}_{model_name.replace(' ', '_').replace('(', '').replace(')', '').lower()}_benchmark_{release_date.strftime('%Y%m%d')}"

    # Create PillarImpact
    pillar_impacts = [
        PillarImpact(
            pillar_name=Pillar.TECHNICAL_CAPABILITIES,
            direction_of_change=DirectionOfChange.ADVANCE,
            relative_strength_signal=RelativeStrength.STRONG if intelligence_idx > 30 else RelativeStrength.MODERATE,
            evidence=f"Benchmark data shows {provider_name} capabilities across intelligence, coding, and specialized tasks. Intelligence Index: {intelligence_idx}, Coding Index: {coding_idx}.",
            impact_score=int(intelligence_idx * 2.5) if intelligence_idx else 70
        )
    ]

    # Create CompetitiveEffects
    competitive_effects = CompetitiveEffects(
        advantages_created=[
            f"Quantified performance across {len([k for k, v in evals.items() if v is not None])} benchmarks",
            "Enables data-driven model selection",
            "Transparent capability comparison"
        ],
        disadvantages_created=[],
        market_gaps_filled=["Model capability transparency", "Competitive benchmarking"],
        lock_in_or_openness_shift="Neutral - benchmark data increases transparency and enables informed switching decisions."
    )

    # Create TemporalContext
    temporal_context = TemporalContext(
        preceded_by_events=[],
        likely_to_trigger_events=[
            "Competitors may release updated models to improve benchmark standings",
            "Pricing adjustments based on performance/cost ratios"
        ],
        time_horizon="immediate"
    )

    # Create MarketSignalEvent
    event = MarketSignalEvent(
        event_id=event_id,
        provider=provider_name,
        source_type="api_benchmark",
        source_url="https://artificialanalysis.ai/",
        published_at=release_date,
        retrieved_at=datetime.now(),
        what_changed=what_changed,
        why_it_matters=why_it_matters,
        scope=f"Global availability. Benchmark data for model capability assessment and competitive analysis.",
        pillars_impacted=pillar_impacts,
        competitive_effects=competitive_effects,
        temporal_context=temporal_context,
        alignment_implications="Benchmark-based evaluation promotes transparency and informed decision-making.",
        extraction_confidence=0.90
    )

    return event


def ingest_api_models():
    """Main ingestion function."""

    # Paths
    base_path = Path(__file__).parent
    glossary_path = base_path / "glossary.yaml"
    api_data_path = base_path / ".." / "aa_api_pull" / "api_data_20260202_150741.json"
    db_path = base_path / "data" / "events.db"
    vector_path = base_path / "data" / "vector_store"

    # Initialize
    print("Loading glossary...")
    contextualizer = GlossaryContextualizer(str(glossary_path))

    print("Loading API data...")
    with open(api_data_path) as f:
        api_data = json.load(f)

    models = api_data['llms']['data']['data']
    print(f"Found {len(models)} total models in API data")

    # Filter to major models
    print("\nFiltering to major provider models...")
    major_models = filter_to_major_models(models, max_per_provider=5)
    print(f"Selected {len(major_models)} models from major providers (top 5 per provider)")

    # Initialize database
    print("\nInitializing database...")
    db = EventDatabase(str(db_path))
    vector_store = EventVectorStore(str(vector_path))

    # Ingest models
    print("\nIngesting model benchmark events...")
    success_count = 0
    skip_count = 0

    for model_data in major_models:
        model_name = model_data.get('name', 'Unknown')
        provider = model_data.get('model_creator', {}).get('name', 'Unknown')

        try:
            # Create event with contextualized benchmarks
            event = create_model_event(model_data, contextualizer)

            # Add to database
            db.create_event(event)
            vector_store.add_event(event)

            print(f"✅ {provider}: {model_name}")
            success_count += 1

        except Exception as e:
            print(f"⚠️  Skipped {provider} {model_name}: {e}")
            skip_count += 1

    print(f"\n✅ Ingestion complete!")
    print(f"   Successfully added: {success_count} models")
    print(f"   Skipped: {skip_count} models")
    print(f"\n📊 Database now includes benchmark data from Artificial Analysis API")
    print(f"   Models can be compared across {len(major_models)} benchmarked systems")


if __name__ == "__main__":
    ingest_api_models()
