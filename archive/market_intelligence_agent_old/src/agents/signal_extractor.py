"""
Signal Extractor Agent (v1 Standard)

Converts raw content into structured Market Signal Events.

Design: Single-LLM extraction with Claude Sonnet 4.5
- Uses structured output (tool use) for schema compliance
- Validates completeness and quality
- Generates unique event IDs
- Self-assesses confidence

Why Claude Sonnet 4.5?
- Complex reasoning required (map to I³ pillars, identify competitive effects)
- Structured output works well with Anthropic's tool use
- High quality extraction reduces need for post-processing

Responsibilities:
1. Parse raw content and extract Market Signal Event
2. Map to I³ pillars with evidence
3. Identify competitive effects (advantages, barriers, lock-in shifts)
4. Determine temporal context (precedes, triggers)
5. Generate event ID (evt_{provider}_{hash}_{date})
6. Validate extraction quality
"""

import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..llm import LLMProvider
from ..llm.structured_output import generate_with_structured_output
from ..models import MarketSignalEvent, Pillar, DirectionOfChange, RelativeStrength
from ..storage import EventDatabase


class SignalExtractor:
    """
    Extracts structured Market Signal Events from raw content.

    Why separate from Content Harvester?
    - Different concerns: parsing vs structuring
    - Can be run independently (re-extract from stored content)
    - Different error handling (validation vs network)
    - Easier to test extraction logic
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        database: EventDatabase,
        config: Dict[str, Any]
    ):
        """
        Initialize Signal Extractor.

        Args:
            llm_provider: LLM for extraction (Claude Sonnet 4.5)
            database: Database for storing events and querying context
            config: Configuration dict
        """
        self.llm = llm_provider
        self.db = database
        self.config = config

        # Extract config
        extractor_config = config['agents']['signal_extractor']
        self.min_confidence = extractor_config.get('min_confidence_threshold', 0.7)

    def extract(
        self,
        content: str,
        provider: str,
        source_url: str,
        source_type: str,
        published_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[MarketSignalEvent]:
        """
        Main entry point: Extract structured event from raw content.

        Args:
            content: Raw text content to analyze
            provider: Provider name (e.g., "OpenAI", "Anthropic")
            source_url: URL of source
            source_type: Type of source (official_blog, github, etc.)
            published_at: Publication date (if known)
            metadata: Optional metadata from Content Harvester

        Returns:
            MarketSignalEvent if extraction successful and confident, else None

        Example:
            event = extractor.extract(
                content=harvested_content.filtered_content,
                provider=harvested_content.provider,
                source_url=harvested_content.url,
                source_type=harvested_content.source_type,
                published_at=harvested_content.fetched_at,
                metadata=harvested_content.metadata
            )
            if event:
                # Store event and update provider memory
        """
        try:
            # Build extraction prompt
            prompt = self._build_extraction_prompt(
                content=content,
                provider=provider,
                source_type=source_type,
                metadata=metadata or {}
            )

            # Call LLM with structured output
            result = generate_with_structured_output(
                llm_provider=self.llm,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                output_model=MarketSignalEvent,
                tool_name="extract_market_signal_event",
                tool_description="Extract structured market signal event from content",
                task_complexity="complex",  # Claude Sonnet 4.5
                temperature=0.3  # Lower = more consistent extraction
            )

            # Convert Pydantic model to dict for processing
            event_data = result.model_dump()

            # Generate event ID if not provided
            if not event_data.get('event_id'):
                event_data['event_id'] = self._generate_event_id(
                    provider=provider,
                    content=content,
                    published_at=published_at or datetime.now()
                )

            # Fill in missing fields
            event_data['source_url'] = source_url
            event_data['source_type'] = source_type
            event_data['retrieved_at'] = datetime.now()
            if published_at:
                event_data['published_at'] = published_at
            elif not event_data.get('published_at'):
                event_data['published_at'] = datetime.now()

            # Create event object
            event = MarketSignalEvent(**event_data)

            # Validate extraction quality
            validation_result = self._validate_extraction(event)

            if not validation_result['is_valid']:
                print(f"Event extraction failed validation: {validation_result['reason']}")
                return None

            # Check confidence threshold
            confidence = validation_result.get('confidence', 0.0)
            if confidence < self.min_confidence:
                print(f"Event confidence too low ({confidence:.2f} < {self.min_confidence})")
                return None

            print(f"✓ Extracted event: {event.event_id} (confidence: {confidence:.2f})")
            return event

        except Exception as e:
            print(f"Signal extraction failed: {e}")
            return None

    def _build_extraction_prompt(
        self,
        content: str,
        provider: str,
        source_type: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Build prompt for LLM extraction.

        Includes:
        - Content to analyze
        - Provider context
        - I³ pillar definitions
        - Extraction guidelines
        - Example structure
        """
        pillar_definitions = self._get_pillar_definitions()

        # Extract content_type hint from metadata if available
        content_type_hint = ""
        if metadata and 'content_type' in metadata:
            content_type = metadata['content_type']
            # Provide pillar hints based on content type
            pillar_hints = {
                'product_announcement': "Likely pillars: TECHNICAL_CAPABILITIES, DATA_PIPELINES",
                'partnership': "Likely pillars: MARKET_SHAPING, EDUCATION_INFLUENCE",
                'governance_policy': "Likely pillars: ALIGNMENT, MARKET_SHAPING",
                'infrastructure_update': "Likely pillars: TECHNICAL_CAPABILITIES",
                'pricing_change': "Likely pillars: TECHNICAL_CAPABILITIES, MARKET_SHAPING"
            }
            hint = pillar_hints.get(content_type, "")
            if hint:
                content_type_hint = f"\n**Content Type Hint:** This is a {content_type}. {hint}"

        prompt = f"""Analyze this content from {provider} ({source_type}) and extract a Market Signal Event.

**Content to Analyze:**
{content}

**Additional Context:**
{json.dumps(metadata, indent=2) if metadata else "None"}{content_type_hint}

**Your Task:**
Extract a structured Market Signal Event following these guidelines:

1. **What Changed**: Concise summary of the factual change (1-2 sentences)
   - Focus on concrete, observable changes
   - Avoid speculation or editorializing

2. **Why It Matters**: Competitive impact explanation (2-3 sentences)
   - How does this shift competitive dynamics?
   - What strategic advantages/disadvantages emerge?
   - Who benefits, who loses?

3. **I³ Pillar Mapping**: Identify which pillars are impacted
   {pillar_definitions}

   For each impacted pillar:
   - direction_of_change: ADVANCE (strengthens position), NEUTRAL, or CONSTRAIN (weakens position)
   - relative_strength_signal: STRONG (clear major impact), MODERATE (notable impact), WEAK (minor signal)
   - evidence: Specific quote or detail from content supporting this pillar impact

4. **Competitive Effects**: Identify concrete competitive dynamics
   - advantages_created: What new advantages does this create for the provider?
   - advantages_eroded: What advantages do competitors lose or get diminished?
   - new_barriers: What new barriers to competition are created?
   - lock_in_or_openness_shift: How does this affect openness vs lock-in?

5. **Temporal Context**: Place in competitive timeline
   - preceded_by_events: What prior events set this up? (event_ids if known, else descriptions)
   - likely_to_trigger_events: What follow-on moves might this trigger?
   - time_horizon: immediate (<3 months), medium (3-12 months), long (>12 months)

6. **Alignment Implications**: Governance/safety dimensions (1-2 sentences)
   - Does this raise safety concerns?
   - Does this demonstrate responsible AI leadership?
   - Does this set governance precedents?

7. **Regulatory Signal**: none, emerging (possible future regulation), material (clear regulatory implications)

**Important**:
- Be specific and evidence-based (cite exact details from content)
- Focus on competitive significance (not just "what happened")
- Only include pillars with clear evidence
- Distinguish between facts (what changed) and analysis (why it matters)
- Consider both direct and indirect competitive effects

Extract the event now.
"""
        return prompt

    def _get_system_prompt(self) -> str:
        """
        System prompt for extraction task.

        Sets context and extraction standards.
        """
        return """You are an expert competitive intelligence analyst specializing in frontier AI markets.

Your task is to extract structured Market Signal Events from raw content. These events track competitive dynamics across the I³ Index framework (5 pillars: Data Pipelines, Technical Capabilities, Education/Influence, Market Shaping, Alignment/Governance).

**Core Principles**:
1. **Evidence-based**: Every claim must be supported by specific content
2. **Competitive focus**: Analyze how this changes relative positions
3. **Pillar precision**: Only map to pillars with clear evidence
4. **Strategic depth**: Go beyond "what happened" to explain competitive implications
5. **Temporal awareness**: Consider precedents and likely responses

**Quality Standards**:
- what_changed: Factual, concise, no speculation
- why_it_matters: Clear competitive impact analysis
- pillar mapping: Minimum 1 pillar, each with evidence
- competitive_effects: Specific, not generic
- temporal_context: Realistic predictions based on market patterns

You are rigorous, analytical, and focused on competitive significance."""

    def _get_pillar_definitions(self) -> str:
        """
        Get I³ pillar definitions for prompt.

        Helps LLM understand pillar mapping.
        """
        return """
   - DATA_PIPELINES: Control over data formats, interoperability, portability, export standards
   - TECHNICAL_CAPABILITIES: Infrastructure, platforms, compute, models, APIs, agent orchestration
   - EDUCATION_INFLUENCE: Training programs, certifications, learning platforms, thought leadership
   - MARKET_SHAPING: Partnerships, alliances, ecosystem development, industry coalitions
   - ALIGNMENT: Governance, safety measures, responsible AI practices, transparency, auditing
"""

    def _generate_event_id(
        self,
        provider: str,
        content: str,
        published_at: datetime
    ) -> str:
        """
        Generate unique event ID.

        Format: evt_{provider}_{content_hash}_{date}

        Why this format?
        - provider: Easy filtering and grouping
        - content_hash: Prevents duplicates (same content = same ID)
        - date: Human-readable temporal ordering

        Example: evt_openai_a3f2b1_20250115
        """
        # Hash content (first 6 chars of SHA256)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:6]

        # Format date
        date_str = published_at.strftime('%Y%m%d')

        # Clean provider name (lowercase, no spaces)
        provider_clean = provider.lower().replace(' ', '')

        return f"evt_{provider_clean}_{content_hash}_{date_str}"

    def _validate_extraction(self, event: MarketSignalEvent) -> Dict[str, Any]:
        """
        Validate extraction quality.

        Checks:
        1. Completeness (all required fields present)
        2. Pillar mapping quality (at least 1 pillar with evidence)
        3. Competitive effects identified
        4. Temporal context provided
        5. Substantive descriptions (not generic)

        Returns:
            Dict with 'is_valid' (bool), 'confidence' (float 0-1), 'reason' (str)
        """
        issues = []
        confidence_score = 1.0

        # Check what_changed
        if not event.what_changed or len(event.what_changed) < 20:
            issues.append("what_changed too short or missing")
            confidence_score -= 0.3

        # Check why_it_matters
        if not event.why_it_matters or len(event.why_it_matters) < 30:
            issues.append("why_it_matters too short or missing")
            confidence_score -= 0.3

        # Check pillar mapping
        if not event.pillars_impacted or len(event.pillars_impacted) == 0:
            issues.append("No pillars mapped")
            confidence_score -= 0.4
        else:
            # Check evidence quality
            for pillar_impact in event.pillars_impacted:
                if not pillar_impact.evidence or len(pillar_impact.evidence) < 15:
                    issues.append(f"Weak evidence for {pillar_impact.pillar_name.value}")
                    confidence_score -= 0.1

        # Check competitive effects
        if not event.competitive_effects:
            issues.append("No competitive effects")
            confidence_score -= 0.3
        else:
            effects = event.competitive_effects
            if (not effects.advantages_created and
                not effects.advantages_eroded and
                not effects.new_barriers):
                issues.append("Competitive effects empty")
                confidence_score -= 0.2

        # Check temporal context
        if not event.temporal_context:
            issues.append("No temporal context")
            confidence_score -= 0.2

        # Overall validation
        is_valid = confidence_score >= self.min_confidence and len(issues) == 0

        return {
            'is_valid': is_valid,
            'confidence': max(0.0, confidence_score),
            'reason': '; '.join(issues) if issues else 'Valid extraction',
            'issues': issues
        }

    def batch_extract(
        self,
        content_list: List[Dict[str, Any]]
    ) -> List[MarketSignalEvent]:
        """
        Extract events from multiple content pieces.

        Useful for batch processing (e.g., backfill from RSS feed).

        Args:
            content_list: List of dicts with keys:
                - content: str (required)
                - provider: str (required)
                - source_url: str (required)
                - source_type: str (required)
                - published_at: datetime (optional)
                - metadata: dict (optional)

        Returns:
            List of successfully extracted events
        """
        events = []

        for i, content_data in enumerate(content_list):
            print(f"Extracting {i+1}/{len(content_list)}...")

            event = self.extract(
                content=content_data['content'],
                provider=content_data['provider'],
                source_url=content_data['source_url'],
                source_type=content_data['source_type'],
                published_at=content_data.get('published_at'),
                metadata=content_data.get('metadata')
            )

            if event:
                events.append(event)

        print(f"✓ Extracted {len(events)}/{len(content_list)} events")
        return events

    def re_extract(self, event_id: str) -> Optional[MarketSignalEvent]:
        """
        Re-extract event from stored content.

        Useful for:
        - Testing prompt improvements
        - Fixing extraction errors
        - Updating analysis with new context

        Args:
            event_id: ID of existing event to re-extract

        Returns:
            New MarketSignalEvent or None if failed
        """
        # Get original event from database
        original_event = self.db.get_event(event_id)

        if not original_event:
            print(f"Event {event_id} not found")
            return None

        # Re-extract using original content
        # (Would need to store original content in DB for this to work)
        # For MVP, this is a placeholder
        print("Re-extraction requires original content storage (not yet implemented)")
        return None
