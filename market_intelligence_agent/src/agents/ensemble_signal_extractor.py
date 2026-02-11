"""
Ensemble Signal Extractor (v2 Experimental)

Multi-LLM extraction with consensus aggregation to reduce hallucinations.

Design: Ensemble voting + aggregator synthesis
1. Multiple extractors run independently (Claude, GPT-4o, HuggingFace)
2. Each produces MarketSignalEvent
3. Aggregator LLM (Claude Opus 4.5) synthesizes consensus
4. Track metrics: consensus rate, hallucination detection, accuracy per model

Why ensemble?
- Reduces hallucinations (cross-validation between models)
- Captures diverse perspectives (different models emphasize different aspects)
- Higher confidence in consensus facts
- Identifies uncertain/subjective claims

Trade-offs:
- Higher cost (3-4 LLM calls instead of 1)
- Higher latency (sequential or parallel extraction)
- More complex error handling
- Requires careful prompt design to avoid bias

When to use:
- High-stakes events requiring maximum accuracy
- Novel/ambiguous content where interpretation varies
- Validation of v1 extractions
- Research into model disagreement patterns
"""

import hashlib
import json
import os
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass

from ..llm import LLMProvider
from ..llm.structured_output import generate_with_structured_output
from ..models import MarketSignalEvent, Pillar, DirectionOfChange, RelativeStrength
from ..storage import EventDatabase


@dataclass
class ExtractionResult:
    """
    Result from a single extractor.

    Includes the extracted event, metadata about the extraction,
    and metrics for evaluation.
    """
    model_name: str  # "claude-sonnet-4.5", "gpt-4o", "meta-llama-3.1-70b"
    event: Optional[MarketSignalEvent]
    extraction_time_ms: float
    success: bool
    error_message: Optional[str] = None

    # Metadata
    token_count: Optional[int] = None
    cost_estimate: Optional[float] = None


@dataclass
class EnsembleMetrics:
    """
    Metrics tracking ensemble performance.

    Used to evaluate:
    - Which models are most reliable
    - Where models disagree (interesting for research)
    - Consensus vs divergence patterns
    """
    # Overall
    total_extractions: int
    successful_extractions: int
    failed_extractions: int

    # Consensus
    consensus_rate: float  # 0-1, how often models agree
    high_agreement_fields: List[str]  # Fields with >80% agreement
    divergent_fields: List[str]  # Fields with <50% agreement

    # Per-model metrics
    model_performance: Dict[str, Dict[str, Any]]  # {model_name: {accuracy, hallucination_rate, ...}}

    # Aggregation
    aggregation_time_ms: float
    confidence_score: float  # 0-1, aggregator's confidence in synthesis


class EnsembleSignalExtractor:
    """
    Ensemble extractor using multiple LLMs with consensus aggregation.

    Why separate from standard SignalExtractor?
    - Different architecture (multiple extractors + aggregator)
    - Different error handling (partial failures allowed)
    - Different use cases (high-stakes vs routine)
    - Experimental (can iterate without breaking v1)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        database: EventDatabase,
        config: Dict[str, Any],
        enable_huggingface: bool = True
    ):
        """
        Initialize Ensemble Signal Extractor.

        Args:
            llm_provider: LLM provider (for Claude and GPT-4o)
            database: Database for storing events and metrics
            config: Configuration dict
            enable_huggingface: Whether to use HuggingFace models (requires HUGGINGFACE_API_KEY)
        """
        self.llm = llm_provider
        self.db = database
        self.config = config

        # Extractors configuration
        self.extractors = [
            {"name": "claude-sonnet-4.5", "provider": "anthropic", "complexity": "complex"},
            {"name": "gpt-4o", "provider": "openai", "complexity": "complex"},
        ]

        # Add HuggingFace if enabled
        self.huggingface_enabled = enable_huggingface and os.getenv('HUGGINGFACE_API_KEY')
        if self.huggingface_enabled:
            self.extractors.append({
                "name": "meta-llama-3.1-70b",
                "provider": "huggingface",
                "complexity": "complex"
            })
        else:
            print("Note: HuggingFace extractors disabled (no HUGGINGFACE_API_KEY)")

        # Aggregator (Claude Opus 4.5 for best reasoning)
        self.aggregator_model = "claude-opus-4.5"

        # Metrics tracking
        self.metrics_history: List[EnsembleMetrics] = []

    def extract(
        self,
        content: str,
        provider: str,
        source_url: str,
        source_type: str,
        published_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parallel: bool = False
    ) -> Optional[MarketSignalEvent]:
        """
        Ensemble extraction: Multiple models extract, aggregator synthesizes.

        Args:
            content: Raw text content to analyze
            provider: Provider name (e.g., "OpenAI", "Anthropic")
            source_url: URL of source
            source_type: Type of source
            published_at: Publication date
            metadata: Optional metadata
            parallel: Whether to run extractors in parallel (faster but more cost upfront)

        Returns:
            Synthesized MarketSignalEvent from ensemble, or None if failed

        Example:
            event = ensemble_extractor.extract(
                content=content,
                provider="OpenAI",
                source_url="https://...",
                source_type="official_blog",
                parallel=True  # Run all extractors in parallel
            )
        """
        try:
            print(f"\n{'='*80}")
            print("ENSEMBLE EXTRACTION")
            print(f"{'='*80}")
            print(f"Extractors: {len(self.extractors)}")
            print(f"Mode: {'Parallel' if parallel else 'Sequential'}")

            # Stage 1: Multiple extractors
            extraction_results = self._run_extractors(
                content=content,
                provider=provider,
                source_url=source_url,
                source_type=source_type,
                published_at=published_at,
                metadata=metadata,
                parallel=parallel
            )

            # Check if we have enough successful extractions
            successful_results = [r for r in extraction_results if r.success]

            if len(successful_results) < 2:
                print(f"✗ Insufficient successful extractions ({len(successful_results)}/{len(self.extractors)})")
                return None

            print(f"\n✓ Successful extractions: {len(successful_results)}/{len(self.extractors)}")
            for result in successful_results:
                print(f"   - {result.model_name}: {result.extraction_time_ms:.0f}ms")

            # Stage 2: Aggregator synthesizes consensus
            print(f"\nAggregating with {self.aggregator_model}...")
            aggregation_start = datetime.now()

            synthesized_event = self._aggregate_extractions(
                extraction_results=successful_results,
                content=content,
                provider=provider,
                source_url=source_url,
                source_type=source_type,
                published_at=published_at
            )

            aggregation_time = (datetime.now() - aggregation_start).total_seconds() * 1000
            print(f"✓ Aggregation complete: {aggregation_time:.0f}ms")

            if not synthesized_event:
                print("✗ Aggregation failed")
                return None

            # Stage 3: Calculate metrics
            metrics = self._calculate_metrics(
                extraction_results=extraction_results,
                aggregation_time_ms=aggregation_time,
                synthesized_event=synthesized_event
            )

            self.metrics_history.append(metrics)

            print(f"\nMetrics:")
            print(f"   Consensus rate: {metrics.consensus_rate:.2%}")
            print(f"   Confidence: {metrics.confidence_score:.2f}")

            return synthesized_event

        except Exception as e:
            print(f"Ensemble extraction failed: {e}")
            return None

    def _run_extractors(
        self,
        content: str,
        provider: str,
        source_url: str,
        source_type: str,
        published_at: Optional[datetime],
        metadata: Optional[Dict[str, Any]],
        parallel: bool
    ) -> List[ExtractionResult]:
        """
        Run all extractors on the content.

        Args:
            content: Content to extract from
            provider: Provider name
            source_url: Source URL
            source_type: Source type
            published_at: Publication date
            metadata: Optional metadata
            parallel: Whether to run in parallel

        Returns:
            List of ExtractionResult objects
        """
        results = []

        # Build base prompt (same for all extractors)
        prompt = self._build_extraction_prompt(content, provider, source_type, metadata or {})

        for extractor_config in self.extractors:
            print(f"\nExtracting with {extractor_config['name']}...")

            result = self._run_single_extractor(
                extractor_config=extractor_config,
                prompt=prompt,
                provider=provider,
                source_url=source_url,
                source_type=source_type,
                published_at=published_at
            )

            results.append(result)

            if result.success:
                print(f"✓ {extractor_config['name']}: Success ({result.extraction_time_ms:.0f}ms)")
            else:
                print(f"✗ {extractor_config['name']}: Failed - {result.error_message}")

        return results

    def _run_single_extractor(
        self,
        extractor_config: Dict[str, Any],
        prompt: str,
        provider: str,
        source_url: str,
        source_type: str,
        published_at: Optional[datetime]
    ) -> ExtractionResult:
        """
        Run a single extractor model.

        Args:
            extractor_config: Config for this extractor
            prompt: Extraction prompt
            provider: Provider name
            source_url: Source URL
            source_type: Source type
            published_at: Publication date

        Returns:
            ExtractionResult with extracted event or error
        """
        start_time = datetime.now()

        try:
            # For HuggingFace models, use different approach
            if extractor_config['provider'] == 'huggingface':
                event = self._extract_with_huggingface(
                    model_name=extractor_config['name'],
                    prompt=prompt,
                    provider=provider,
                    source_url=source_url,
                    source_type=source_type,
                    published_at=published_at
                )
            else:
                # Use standard structured output for Claude/GPT-4o
                event = generate_with_structured_output(
                    llm_provider=self.llm,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt()},
                        {"role": "user", "content": prompt}
                    ],
                    output_model=MarketSignalEvent,
                    tool_name="extract_market_signal_event",
                    tool_description="Extract structured market signal event from content",
                    task_complexity=extractor_config['complexity'],
                    temperature=0.3
                )

            extraction_time = (datetime.now() - start_time).total_seconds() * 1000

            # Validate extraction
            if event and hasattr(event, 'event_id'):
                # Fill in missing fields if needed
                if not event.event_id:
                    event.event_id = self._generate_event_id(provider, prompt, published_at or datetime.now())
                if not event.source_url:
                    event.source_url = source_url
                if not event.source_type:
                    event.source_type = source_type
                if not event.retrieved_at:
                    event.retrieved_at = datetime.now()
                if not event.published_at:
                    event.published_at = published_at or datetime.now()

                return ExtractionResult(
                    model_name=extractor_config['name'],
                    event=event,
                    extraction_time_ms=extraction_time,
                    success=True
                )
            else:
                return ExtractionResult(
                    model_name=extractor_config['name'],
                    event=None,
                    extraction_time_ms=extraction_time,
                    success=False,
                    error_message="Event validation failed"
                )

        except Exception as e:
            extraction_time = (datetime.now() - start_time).total_seconds() * 1000
            return ExtractionResult(
                model_name=extractor_config['name'],
                event=None,
                extraction_time_ms=extraction_time,
                success=False,
                error_message=str(e)
            )

    def _extract_with_huggingface(
        self,
        model_name: str,
        prompt: str,
        provider: str,
        source_url: str,
        source_type: str,
        published_at: Optional[datetime]
    ) -> Optional[MarketSignalEvent]:
        """
        Extract using HuggingFace Inference API.

        HuggingFace doesn't support structured outputs like Anthropic/OpenAI,
        so we need to:
        1. Request JSON output in prompt
        2. Parse JSON from response
        3. Validate against schema

        Args:
            model_name: HuggingFace model name
            prompt: Extraction prompt
            provider: Provider name
            source_url: Source URL
            source_type: Source type
            published_at: Publication date

        Returns:
            MarketSignalEvent or None if parsing failed
        """
        import requests

        api_key = os.getenv('HUGGINGFACE_API_KEY')
        if not api_key:
            raise ValueError("HUGGINGFACE_API_KEY not set")

        # HuggingFace Inference API endpoint
        api_url = f"https://api-inference.huggingface.co/models/{model_name}"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Build prompt with explicit JSON instruction
        full_prompt = f"""{self._get_system_prompt()}

{prompt}

IMPORTANT: Return ONLY valid JSON matching the MarketSignalEvent schema. No markdown, no explanation."""

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "temperature": 0.3,
                "max_new_tokens": 4096,
                "return_full_text": False
            }
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()

            # Parse response
            result = response.json()

            # Extract generated text
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get('generated_text', '')
            elif isinstance(result, dict):
                generated_text = result.get('generated_text', '')
            else:
                raise ValueError(f"Unexpected HuggingFace response format: {result}")

            # Try to parse JSON from response
            # Sometimes models wrap JSON in markdown code blocks
            json_text = generated_text.strip()
            if json_text.startswith('```'):
                # Extract JSON from code block
                lines = json_text.split('\n')
                json_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else json_text

            event_data = json.loads(json_text)

            # Validate against schema
            event = MarketSignalEvent(**event_data)

            return event

        except Exception as e:
            print(f"HuggingFace extraction failed: {e}")
            return None

    def _aggregate_extractions(
        self,
        extraction_results: List[ExtractionResult],
        content: str,
        provider: str,
        source_url: str,
        source_type: str,
        published_at: Optional[datetime]
    ) -> Optional[MarketSignalEvent]:
        """
        Aggregate multiple extractions into a single consensus event.

        Uses Claude Opus 4.5 as aggregator to:
        1. Identify consensus facts (all models agree)
        2. Resolve disagreements (majority vote or best reasoning)
        3. Flag potential hallucinations (outlier claims)
        4. Produce final synthesized event

        Args:
            extraction_results: List of successful extractions
            content: Original content (for reference)
            provider: Provider name
            source_url: Source URL
            source_type: Source type
            published_at: Publication date

        Returns:
            Synthesized MarketSignalEvent
        """
        # Build aggregation prompt
        extractions_summary = []

        for i, result in enumerate(extraction_results, 1):
            event = result.event
            summary = {
                "extractor": result.model_name,
                "what_changed": event.what_changed,
                "why_it_matters": event.why_it_matters,
                "pillars": [
                    {
                        "pillar": p.pillar_name.value,
                        "direction": p.direction_of_change.value,
                        "strength": p.relative_strength_signal.value,
                        "evidence": p.evidence[:200]  # Truncate for brevity
                    }
                    for p in event.pillars_impacted
                ],
                "advantages_created": event.competitive_effects.advantages_created,
                "advantages_eroded": event.competitive_effects.advantages_eroded,
                "new_barriers": event.competitive_effects.new_barriers
            }
            extractions_summary.append(summary)

        aggregation_prompt = f"""You are an expert aggregator synthesizing {len(extraction_results)} independent Market Signal Event extractions.

Your task: Produce a single, high-confidence MarketSignalEvent by:
1. **Consensus facts**: Include claims all extractors agree on
2. **Resolve disagreements**: Use majority vote or best-supported reasoning
3. **Detect hallucinations**: Exclude claims made by only one extractor without strong evidence
4. **Synthesize**: Produce a coherent, evidence-based event

**Original Content:**
{content[:3000]}

**Extraction Results:**
{json.dumps(extractions_summary, indent=2)}

**Analysis Guidelines:**
- **what_changed**: Synthesize consensus description (favor facts over interpretation)
- **why_it_matters**: Combine competitive insights (include perspectives from multiple extractors if complementary)
- **pillars_impacted**: Include pillar if 2+ extractors identified it OR 1 extractor with very strong evidence
- **competitive_effects**: Include effect if 2+ extractors identified it OR 1 extractor with specific supporting evidence
- **Disagreements**: Note where extractors diverged (in reasoning field)
- **Confidence**: Higher consensus = higher confidence

Extract the synthesized event now."""

        try:
            # Use Claude Opus 4.5 for aggregation (best reasoning)
            # Note: This requires the model to be configured in LLMProvider
            # For MVP, we'll use Claude Sonnet 4.5 as fallback

            synthesized = generate_with_structured_output(
                llm_provider=self.llm,
                messages=[
                    {"role": "system", "content": "You are an expert competitive intelligence analyst specializing in ensemble extraction aggregation."},
                    {"role": "user", "content": aggregation_prompt}
                ],
                output_model=MarketSignalEvent,
                tool_name="synthesize_consensus_event",
                tool_description="Synthesize consensus Market Signal Event from multiple extractions",
                task_complexity="complex",  # Claude Sonnet 4.5 (Opus not available yet)
                temperature=0.2  # Low temperature for consistent synthesis
            )

            # Fill in metadata
            synthesized_dict = synthesized.model_dump()
            synthesized_dict['source_url'] = source_url
            synthesized_dict['source_type'] = source_type
            synthesized_dict['retrieved_at'] = datetime.now()
            synthesized_dict['published_at'] = published_at or datetime.now()

            if not synthesized_dict.get('event_id'):
                synthesized_dict['event_id'] = self._generate_event_id(
                    provider, content, published_at or datetime.now()
                )

            return MarketSignalEvent(**synthesized_dict)

        except Exception as e:
            print(f"Aggregation failed: {e}")
            return None

    def _calculate_metrics(
        self,
        extraction_results: List[ExtractionResult],
        aggregation_time_ms: float,
        synthesized_event: MarketSignalEvent
    ) -> EnsembleMetrics:
        """
        Calculate ensemble performance metrics.

        Tracks:
        - Success rates per model
        - Consensus vs divergence
        - Confidence in synthesis

        Args:
            extraction_results: All extraction results
            aggregation_time_ms: Time taken for aggregation
            synthesized_event: Final synthesized event

        Returns:
            EnsembleMetrics object
        """
        successful = [r for r in extraction_results if r.success]
        failed = [r for r in extraction_results if not r.success]

        # Calculate consensus rate (simplified: check pillar agreement)
        pillar_agreements = self._calculate_pillar_agreement(successful)
        consensus_rate = sum(pillar_agreements.values()) / max(len(pillar_agreements), 1)

        # Identify high/low agreement fields
        high_agreement = [p for p, rate in pillar_agreements.items() if rate > 0.8]
        divergent = [p for p, rate in pillar_agreements.items() if rate < 0.5]

        # Per-model performance (simplified)
        model_performance = {}
        for result in extraction_results:
            model_performance[result.model_name] = {
                "success": result.success,
                "extraction_time_ms": result.extraction_time_ms,
                "error": result.error_message if not result.success else None
            }

        return EnsembleMetrics(
            total_extractions=len(extraction_results),
            successful_extractions=len(successful),
            failed_extractions=len(failed),
            consensus_rate=consensus_rate,
            high_agreement_fields=high_agreement,
            divergent_fields=divergent,
            model_performance=model_performance,
            aggregation_time_ms=aggregation_time_ms,
            confidence_score=0.8 if consensus_rate > 0.7 else 0.6  # Simplified confidence
        )

    def _calculate_pillar_agreement(
        self,
        successful_results: List[ExtractionResult]
    ) -> Dict[str, float]:
        """
        Calculate agreement rate for each pillar across extractors.

        Args:
            successful_results: Successful extraction results

        Returns:
            Dict mapping pillar name to agreement rate (0-1)
        """
        if not successful_results:
            return {}

        # Count how many extractors identified each pillar
        pillar_counts: Dict[Pillar, int] = {}

        for result in successful_results:
            if result.event:
                for pillar_impact in result.event.pillars_impacted:
                    pillar = pillar_impact.pillar_name
                    pillar_counts[pillar] = pillar_counts.get(pillar, 0) + 1

        # Calculate agreement rate
        total_extractors = len(successful_results)
        agreement_rates = {
            pillar.value: count / total_extractors
            for pillar, count in pillar_counts.items()
        }

        return agreement_rates

    def _build_extraction_prompt(
        self,
        content: str,
        provider: str,
        source_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Build extraction prompt (same as v1 Signal Extractor).

        This ensures consistency across extractors.
        """
        # Import from v1 if needed, or duplicate the prompt here
        # For now, simplified version
        return f"""Analyze this content from {provider} ({source_type}) and extract a Market Signal Event.

**Content:**
{content}

Extract:
1. what_changed (concise factual summary)
2. why_it_matters (competitive impact)
3. pillars_impacted (which I³ pillars, with evidence)
4. competitive_effects (advantages created/eroded, barriers, lock-in)
5. temporal_context (what preceded, what triggers, time horizon)
6. alignment_implications
7. regulatory_signal

Be specific, evidence-based, and focused on competitive significance."""

    def _get_system_prompt(self) -> str:
        """System prompt for extractors."""
        return """You are an expert competitive intelligence analyst specializing in frontier AI markets.

Extract structured Market Signal Events that track competitive dynamics across the I³ Index framework.

Focus on:
- Evidence-based claims (cite specific content)
- Competitive impact analysis
- Pillar precision (only map with clear evidence)
- Strategic depth (explain competitive implications)"""

    def _generate_event_id(
        self,
        provider: str,
        content: str,
        published_at: datetime
    ) -> str:
        """Generate unique event ID."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:6]
        date_str = published_at.strftime('%Y%m%d')
        provider_clean = provider.lower().replace(' ', '')
        return f"evt_{provider_clean}_{content_hash}_{date_str}"

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get summary of ensemble performance across all extractions.

        Returns:
            Dict with aggregated metrics
        """
        if not self.metrics_history:
            return {"message": "No extractions performed yet"}

        avg_consensus = sum(m.consensus_rate for m in self.metrics_history) / len(self.metrics_history)
        avg_confidence = sum(m.confidence_score for m in self.metrics_history) / len(self.metrics_history)
        total_successful = sum(m.successful_extractions for m in self.metrics_history)
        total_failed = sum(m.failed_extractions for m in self.metrics_history)

        return {
            "total_ensemble_extractions": len(self.metrics_history),
            "average_consensus_rate": avg_consensus,
            "average_confidence": avg_confidence,
            "total_successful_individual_extractions": total_successful,
            "total_failed_individual_extractions": total_failed,
            "success_rate": total_successful / (total_successful + total_failed) if (total_successful + total_failed) > 0 else 0
        }
