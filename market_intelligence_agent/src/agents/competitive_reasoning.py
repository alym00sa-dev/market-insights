"""
Competitive Reasoning Agent

Analyzes competitive dynamics across frontier AI providers using the I³ framework.

Design: 4 query types, each with specialized reasoning
1. Event Impact: How did event X change competitive dynamics?
2. Provider Comparison: How do providers A and B differ on pillar X?
3. Leadership Ranking: Who leads on pillar X over time period Y?
4. Timeline Analysis: How did pillar X evolve over time period Y?

Why Claude Sonnet 4.5?
- Complex reasoning required (synthesizing multiple events)
- Strategic analysis (not just data aggregation)
- Evidence-based argumentation
- Competitive pattern recognition

Uses:
- Database queries (pillar-based, temporal chains)
- Vector store (semantic search)
- LLM reasoning (synthesis, analysis, narrative)
"""

import json
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, timedelta
from collections import defaultdict

from ..llm import LLMProvider
from ..storage import EventDatabase, EventVectorStore
from ..models import Pillar, DirectionOfChange, RelativeStrength, MarketSignalEvent


class CompetitiveReasoning:
    """
    Analyzes competitive dynamics using Market Signal Events.

    Core capability: Transform raw events into strategic insights.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        database: EventDatabase,
        vector_store: EventVectorStore,
        config: Dict[str, Any]
    ):
        """
        Initialize Competitive Reasoning agent.

        Args:
            llm_provider: LLM for complex reasoning
            database: Event database
            vector_store: Vector store for semantic search
            config: Configuration dict
        """
        self.llm = llm_provider
        self.db = database
        self.vector_store = vector_store
        self.config = config

        # Extract config
        reasoning_config = config['agents']['competitive_reasoning']
        self.max_events_per_query = reasoning_config.get('max_events_per_query', 50)

    # ========================================================================
    # QUERY TYPE 1: EVENT IMPACT ANALYSIS
    # ========================================================================

    def analyze_event_impact(
        self,
        event_id: str
    ) -> Dict[str, Any]:
        """
        Analyze how a single event changed competitive dynamics.

        Question: "When X released Y, how did that change the market?"

        Analysis includes:
        - Immediate competitive impact
        - Advantages created/eroded
        - Barriers established
        - Responses triggered (follow-on events)
        - Long-term implications

        Args:
            event_id: Event to analyze

        Returns:
            Dict with:
                - event_summary: Brief description
                - immediate_impact: Short-term effects
                - competitive_shifts: Changes in relative positions
                - triggered_responses: Events that followed
                - long_term_implications: Strategic consequences
                - confidence: Analysis confidence (0-1)

        Example:
            result = reasoning.analyze_event_impact("evt_openai_context_200k")
        """
        # Get the event
        event = self.db.get_event(event_id)
        if not event:
            return {'error': f'Event {event_id} not found'}

        # Get temporal context (what came before/after)
        chain = self.db.get_event_chain(event_id, direction='both', max_depth=2)

        # Build analysis prompt
        prompt = self._build_event_impact_prompt(event, chain)

        # LLM reasoning
        messages = [
            {"role": "system", "content": self._get_system_prompt("event_impact")},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.generate(
            messages=messages,
            task_complexity="complex",  # Claude Sonnet 4.5
            temperature=0.3
        )

        # Parse response
        try:
            content = response['content'].strip()

            # Try to extract JSON if wrapped in markdown
            if content.startswith('```'):
                # Remove markdown code blocks
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            analysis = json.loads(content)

            # Add metadata
            analysis['query_type'] = 'event_impact'
            analysis['event_id'] = event_id
            analysis['event'] = {
                'provider': event.provider,
                'what_changed': event.what_changed,
                'published_at': event.published_at.isoformat()
            }

            return analysis

        except json.JSONDecodeError as e:
            return {
                'error': f'Failed to parse LLM response: {str(e)}',
                'raw_response': response['content'][:500]  # Truncate for debugging
            }

    # ========================================================================
    # QUERY TYPE 2: PROVIDER COMPARISON
    # ========================================================================

    def compare_providers(
        self,
        providers: List[str],
        pillar: Optional[Pillar] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Compare providers on a specific pillar (or overall).

        Question: "How do OpenAI and Anthropic differ on memory portability?"

        Analysis includes:
        - Key differences in approach/strategy
        - Relative strengths/weaknesses
        - Evidence from events
        - Convergence vs divergence trends

        Args:
            providers: List of provider names (2-4 recommended)
            pillar: Optional pillar to focus comparison
            start_date: Optional start date
            end_date: Optional end date

        Returns:
            Dict with:
                - providers: Providers being compared
                - pillar: Pillar being analyzed (if specified)
                - differences: List of key differences with evidence
                - convergence_divergence: Are they converging or diverging?
                - strategic_implications: What this means competitively
                - confidence: Analysis confidence (0-1)

        Example:
            result = reasoning.compare_providers(
                ["OpenAI", "Anthropic"],
                pillar=Pillar.TECHNICAL_CAPABILITIES,
                start_date=datetime(2024, 1, 1)
            )
        """
        # Get events for each provider
        provider_events = {}
        for provider in providers:
            if pillar:
                events = self.db.get_events_by_pillar(
                    pillar=pillar,
                    provider=provider,
                    start_date=start_date,
                    end_date=end_date,
                    limit=self.max_events_per_query
                )
            else:
                # Get all events for provider
                events = self.db.search_events(
                    provider=provider,
                    start_date=start_date,
                    end_date=end_date,
                    limit=self.max_events_per_query
                )
            provider_events[provider] = events

        # Build comparison prompt
        prompt = self._build_comparison_prompt(provider_events, pillar)

        # LLM reasoning
        messages = [
            {"role": "system", "content": self._get_system_prompt("provider_comparison")},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.generate(
            messages=messages,
            task_complexity="complex",
            temperature=0.3
        )

        # Parse response
        try:
            content = response['content'].strip()

            # Try to extract JSON if wrapped in markdown
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            analysis = json.loads(content)

            # Add metadata
            analysis['query_type'] = 'provider_comparison'
            analysis['providers'] = providers
            analysis['pillar'] = pillar.value if pillar else 'overall'
            analysis['time_range'] = {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else None
            }

            return analysis

        except json.JSONDecodeError as e:
            return {
                'error': f'Failed to parse LLM response: {str(e)}',
                'raw_response': response['content'][:500]
            }

    # ========================================================================
    # QUERY TYPE 3: LEADERSHIP RANKING
    # ========================================================================

    def rank_leadership(
        self,
        pillar: Pillar,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        providers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Rank providers by leadership on a pillar over time.

        Question: "Who is leading on infrastructure openness over last 6 months?"

        Scoring approach:
        1. Event quantity (how many significant events?)
        2. Event quality (strong vs weak signals)
        3. Direction (advancing vs constraining)
        4. Momentum (accelerating vs lagging)

        Args:
            pillar: Pillar to analyze leadership on
            start_date: Optional start date (default: 6 months ago)
            end_date: Optional end date (default: now)
            providers: Optional list to limit (default: all providers)

        Returns:
            Dict with:
                - pillar: Pillar being analyzed
                - rankings: List of providers ranked by leadership score
                    Each entry:
                        - provider: Provider name
                        - score: Leadership score (0-100)
                        - evidence_events: Event IDs supporting ranking
                        - key_strengths: What they're doing well
                        - key_weaknesses: Gaps or constraints
                - analysis: Narrative explanation of rankings
                - confidence: Analysis confidence (0-1)

        Example:
            result = reasoning.rank_leadership(
                Pillar.DATA_PIPELINES,
                start_date=datetime.now() - timedelta(days=180)
            )
        """
        # Default time range: last 6 months
        if not start_date:
            start_date = datetime.now() - timedelta(days=180)
        if not end_date:
            end_date = datetime.now()

        # Get all events for this pillar in time range
        events = self.db.get_events_by_pillar(
            pillar=pillar,
            start_date=start_date,
            end_date=end_date,
            limit=self.max_events_per_query * 2  # Get more for comprehensive analysis
        )

        # Filter by providers if specified
        if providers:
            events = [e for e in events if e.provider in providers]

        # Calculate scores per provider
        provider_scores = self._calculate_leadership_scores(events, pillar)

        # Build ranking prompt
        prompt = self._build_ranking_prompt(provider_scores, events, pillar)

        # LLM reasoning
        messages = [
            {"role": "system", "content": self._get_system_prompt("leadership_ranking")},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.generate(
            messages=messages,
            task_complexity="complex",
            temperature=0.3
        )

        # Parse response
        try:
            content = response['content'].strip()

            # Try to extract JSON if wrapped in markdown
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            analysis = json.loads(content)

            # Add metadata
            analysis['query_type'] = 'leadership_ranking'
            analysis['pillar'] = pillar.value
            analysis['time_range'] = {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }

            return analysis

        except json.JSONDecodeError as e:
            return {
                'error': f'Failed to parse LLM response: {str(e)}',
                'raw_response': response['content'][:500]
            }

    # ========================================================================
    # QUERY TYPE 4: TIMELINE ANALYSIS
    # ========================================================================

    def analyze_timeline(
        self,
        pillar: Pillar,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        providers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze how a pillar evolved over time.

        Question: "How did governance expectations evolve this year?"

        Analysis includes:
        - Chronological sequence of significant events
        - Narrative arc (what changed and why)
        - Key trends and patterns
        - Turning points
        - Future trajectory

        Args:
            pillar: Pillar to analyze evolution
            start_date: Optional start date (default: 1 year ago)
            end_date: Optional end date (default: now)
            providers: Optional provider filter

        Returns:
            Dict with:
                - pillar: Pillar being analyzed
                - timeline: List of events chronologically
                    Each entry:
                        - date: Event date
                        - event_id: Event ID
                        - provider: Provider
                        - description: What happened
                        - significance: "high", "medium", "low"
                - narrative: Story of how this pillar evolved
                - key_trends: Major patterns observed
                - turning_points: Critical events that shifted trajectory
                - future_trajectory: Where this is heading
                - confidence: Analysis confidence (0-1)

        Example:
            result = reasoning.analyze_timeline(
                Pillar.ALIGNMENT,
                start_date=datetime(2024, 1, 1)
            )
        """
        # Default time range: last year
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()

        # Get events for this pillar
        events = self.db.get_events_by_pillar(
            pillar=pillar,
            start_date=start_date,
            end_date=end_date,
            limit=self.max_events_per_query
        )

        # Filter by providers if specified
        if providers:
            events = [e for e in events if e.provider in providers]

        # Sort chronologically
        events.sort(key=lambda e: e.published_at)

        # Build timeline prompt
        prompt = self._build_timeline_prompt(events, pillar)

        # LLM reasoning
        messages = [
            {"role": "system", "content": self._get_system_prompt("timeline")},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.generate(
            messages=messages,
            task_complexity="complex",
            temperature=0.3
        )

        # Parse response
        try:
            content = response['content'].strip()

            # Try to extract JSON if wrapped in markdown
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            analysis = json.loads(content)

            # Add metadata
            analysis['query_type'] = 'timeline'
            analysis['pillar'] = pillar.value
            analysis['time_range'] = {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }

            return analysis

        except json.JSONDecodeError as e:
            return {
                'error': f'Failed to parse LLM response: {str(e)}',
                'raw_response': response['content'][:500]
            }

    # ========================================================================
    # HELPER METHODS - PROMPT BUILDING
    # ========================================================================

    def _build_event_impact_prompt(
        self,
        event: MarketSignalEvent,
        chain: Dict[str, Any]
    ) -> str:
        """Build prompt for event impact analysis."""
        predecessors_text = ""
        if chain['predecessors']:
            pred_list = [f"- {e.event_id}: {e.what_changed}" for e in chain['predecessors'][:5]]
            predecessors_text = f"\n**Preceded By:**\n" + "\n".join(pred_list)

        successors_text = ""
        if chain['successors']:
            succ_list = [f"- {e.event_id}: {e.what_changed}" for e in chain['successors'][:5]]
            successors_text = f"\n**Followed By:**\n" + "\n".join(succ_list)

        return f"""Analyze the competitive impact of this event:

**Event:** {event.event_id}
**Provider:** {event.provider}
**Date:** {event.published_at.strftime('%Y-%m-%d')}

**What Changed:**
{event.what_changed}

**Why It Matters:**
{event.why_it_matters}

**Pillars Impacted:**
{self._format_pillars(event.pillars_impacted)}

**Competitive Effects:**
- Advantages Created: {', '.join(event.competitive_effects.advantages_created) if event.competitive_effects.advantages_created else 'None'}
- Advantages Eroded: {', '.join(event.competitive_effects.advantages_eroded) if event.competitive_effects.advantages_eroded else 'None'}
- New Barriers: {', '.join(event.competitive_effects.new_barriers) if event.competitive_effects.new_barriers else 'None'}
- Lock-in Shift: {event.competitive_effects.lock_in_or_openness_shift}{predecessors_text}{successors_text}

**Your Task:**
Respond with ONLY a JSON object (no other text, no markdown). Use this exact structure:

{{
  "immediate_impact": "What changed immediately in competitive dynamics?",
  "competitive_shifts": [
    {{
      "dimension": "e.g., cost competitiveness",
      "shift": "How relative positions changed",
      "evidence": "Specific details from event"
    }}
  ],
  "triggered_responses": [
    {{
      "provider": "Provider that responded",
      "response_event": "Event ID if known, else description",
      "timing": "How quickly they responded"
    }}
  ],
  "long_term_implications": "Strategic consequences over 6-12 months",
  "confidence": 0.85
}}

Focus on:
1. How relative competitive positions changed
2. Strategic advantages/disadvantages created
3. Likely competitive responses
4. Long-term market structure implications

Remember: Return ONLY the JSON object, starting with {{ and ending with }}."""

    def _build_comparison_prompt(
        self,
        provider_events: Dict[str, List[MarketSignalEvent]],
        pillar: Optional[Pillar]
    ) -> str:
        """Build prompt for provider comparison."""
        pillar_text = f" on {pillar.value}" if pillar else ""

        events_summary = []
        for provider, events in provider_events.items():
            event_list = [f"  - {e.published_at.strftime('%Y-%m-%d')}: {e.what_changed[:80]}..." for e in events[:10]]
            events_summary.append(f"\n**{provider}** ({len(events)} events):\n" + "\n".join(event_list))

        return f"""Compare these providers{pillar_text}:

{''.join(events_summary)}

**Your Task:**
Respond with ONLY a JSON object (no other text, no markdown). Use this exact structure:

{{
  "differences": [
    {{
      "dimension": "e.g., strategic approach",
      "provider_positions": {{
        "Provider1": "Their approach/position",
        "Provider2": "Their approach/position"
      }},
      "evidence": ["event_id_1", "event_id_2"]
    }}
  ],
  "convergence_divergence": "Are they converging (similar strategies) or diverging (different approaches)?",
  "strategic_implications": "What this means for competitive dynamics",
  "confidence": 0.85
}}

Focus on:
1. Strategic differences (not just "who did more")
2. Underlying approaches/philosophies
3. Strengths/weaknesses revealed by events
4. Whether they're converging or diverging

Remember: Return ONLY the JSON object, starting with {{ and ending with }}."""

    def _build_ranking_prompt(
        self,
        provider_scores: Dict[str, Dict[str, Any]],
        events: List[MarketSignalEvent],
        pillar: Pillar
    ) -> str:
        """Build prompt for leadership ranking."""
        scores_text = []
        for provider, data in sorted(provider_scores.items(), key=lambda x: x[1]['score'], reverse=True):
            scores_text.append(
                f"- {provider}: {data['score']:.1f} points "
                f"({data['num_events']} events, {data['advances']} advances, {data['constrains']} constraints)"
            )

        scores_formatted = '\n'.join(scores_text)
        return f"""Rank providers by leadership on {pillar.value}:

**Calculated Scores:**
{scores_formatted}

**Recent Events (sample):**
{self._format_events_for_prompt(events[:15])}

**Your Task:**
Respond with ONLY a JSON object (no other text, no markdown). Use this exact structure:

{{
  "rankings": [
    {{
      "provider": "Provider name",
      "score": 85.0,
      "evidence_events": ["evt_1", "evt_2"],
      "key_strengths": ["What they're doing well"],
      "key_weaknesses": ["Gaps or areas behind"]
    }}
  ],
  "analysis": "Narrative explaining rankings and competitive dynamics",
  "confidence": 0.85
}}

Focus on:
1. Not just quantity, but quality and direction
2. Strategic significance of events
3. Momentum (accelerating vs plateauing)
4. Gaps and opportunities

Remember: Return ONLY the JSON object, starting with {{ and ending with }}."""

    def _build_timeline_prompt(
        self,
        events: List[MarketSignalEvent],
        pillar: Pillar
    ) -> str:
        """Build prompt for timeline analysis."""
        return f"""Analyze how {pillar.value} evolved over time:

**Events (chronological):**
{self._format_events_for_prompt(events)}

**Your Task:**
Respond with ONLY a JSON object (no other text, no markdown). Use this exact structure:

{{
  "timeline": [
    {{
      "date": "2024-03-15",
      "event_id": "evt_...",
      "provider": "Provider",
      "description": "What happened",
      "significance": "high"
    }}
  ],
  "narrative": "Story of how this pillar evolved (2-3 paragraphs)",
  "key_trends": ["Major patterns observed"],
  "turning_points": [
    {{
      "event_id": "evt_...",
      "why": "Why this was a turning point"
    }}
  ],
  "future_trajectory": "Where this is likely heading",
  "confidence": 0.85
}}

Focus on:
1. Narrative arc (not just event list)
2. Cause and effect relationships
3. Inflection points that changed trajectory
4. Emerging patterns

Remember: Return ONLY the JSON object, starting with {{ and ending with }}."""

    # ========================================================================
    # HELPER METHODS - SCORING & FORMATTING
    # ========================================================================

    def _calculate_leadership_scores(
        self,
        events: List[MarketSignalEvent],
        pillar: Pillar
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate leadership scores for each provider.

        Score = weighted sum of:
        - Event count (quantity)
        - Signal strength (quality)
        - Direction (advancing vs constraining)
        - Recency (recent events weighted higher)
        """
        provider_data = defaultdict(lambda: {
            'events': [],
            'score': 0.0,
            'num_events': 0,
            'advances': 0,
            'constrains': 0
        })

        for event in events:
            provider = event.provider
            provider_data[provider]['events'].append(event)
            provider_data[provider]['num_events'] += 1

            # Find pillar impact
            pillar_impact = None
            for impact in event.pillars_impacted:
                if impact.pillar_name == pillar:
                    pillar_impact = impact
                    break

            if not pillar_impact:
                continue

            # Score components
            direction_score = {
                DirectionOfChange.ADVANCE: 1.0,
                DirectionOfChange.NEUTRAL: 0.0,
                DirectionOfChange.CONSTRAIN: -0.5
            }[pillar_impact.direction_of_change]

            strength_score = {
                RelativeStrength.STRONG: 1.0,
                RelativeStrength.MODERATE: 0.67,
                RelativeStrength.WEAK: 0.33
            }[pillar_impact.relative_strength_signal]

            # Recency bonus (events in last 30 days get 1.5x)
            days_old = (datetime.now() - event.published_at).days
            recency_multiplier = 1.5 if days_old < 30 else 1.0

            # Calculate score
            event_score = direction_score * strength_score * recency_multiplier * 10

            provider_data[provider]['score'] += event_score

            # Track direction counts
            if pillar_impact.direction_of_change == DirectionOfChange.ADVANCE:
                provider_data[provider]['advances'] += 1
            elif pillar_impact.direction_of_change == DirectionOfChange.CONSTRAIN:
                provider_data[provider]['constrains'] += 1

        return dict(provider_data)

    def _format_pillars(self, pillars_impacted: List) -> str:
        """Format pillar impacts for prompt."""
        lines = []
        for impact in pillars_impacted:
            lines.append(
                f"- {impact.pillar_name.value}: {impact.direction_of_change.value} "
                f"({impact.relative_strength_signal.value})"
            )
        return "\n".join(lines)

    def _format_events_for_prompt(self, events: List[MarketSignalEvent]) -> str:
        """Format events for inclusion in prompt."""
        lines = []
        for event in events:
            lines.append(
                f"- {event.published_at.strftime('%Y-%m-%d')} | {event.provider} | "
                f"{event.event_id}: {event.what_changed[:100]}..."
            )
        return "\n".join(lines)

    def _get_system_prompt(self, query_type: str) -> str:
        """Get system prompt for specific query type."""
        base = """You are an expert competitive intelligence analyst specializing in frontier AI markets.

You analyze Market Signal Events using the I³ Index framework (5 pillars: Data Pipelines, Technical Capabilities, Education/Influence, Market Shaping, Alignment/Governance).

Your analysis is:
- Evidence-based (cite specific events)
- Strategic (focus on competitive dynamics, not just facts)
- Pattern-recognizing (identify trends and implications)
- Actionable (provide insights for strategic decisions)

CRITICAL: You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON object. Start your response with { and end with }. No markdown formatting, no explanation text."""

        additions = {
            "event_impact": "\n\nFocus on cause-and-effect: What changed, why it matters, what it triggered.",
            "provider_comparison": "\n\nFocus on strategic differences: How providers differ in approach, not just activity level.",
            "leadership_ranking": "\n\nFocus on justified rankings: Use evidence, not just scores. Explain momentum and strategy.",
            "timeline": "\n\nFocus on narrative: Tell a story of evolution, identify turning points, predict trajectory."
        }

        return base + additions.get(query_type, "")
