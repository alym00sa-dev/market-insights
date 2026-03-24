"""
Provider Intelligence Layer

Per-hyperscaler signal consolidation and validation.

Responsibilities:
1. Aggregate events per provider to build strategic profile
2. Identify patterns, themes, and strategic direction
3. Track momentum and trajectory over time
4. Validate signal consistency (detect contradictions)
5. Build provider context for competitive reasoning

Design:
- Integrated into Ensemble Extractor (not standalone)
- Updates after each event extraction
- Maintains rolling window of provider activity
- Synthesizes strategic narrative per provider

Why this matters:
- Prevents "comparing apples to oranges" (validates signals before comparison)
- Builds rich provider context for competitive analysis
- Identifies strategic shifts and pivots
- Helps Competitive Reasoning make better comparisons
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from ..llm import LLMProvider
from ..models import MarketSignalEvent, Pillar, DirectionOfChange
from ..storage import EventDatabase


@dataclass
class ProviderProfile:
    """
    Strategic profile for a single hyperscaler.

    Built incrementally as events are processed.
    """
    provider: str
    last_updated: datetime

    # Event statistics
    total_events: int = 0
    events_by_pillar: Dict[str, int] = field(default_factory=dict)
    recent_events: List[str] = field(default_factory=list)  # Event IDs (last 30 days)

    # Strategic themes (identified by LLM)
    primary_themes: List[str] = field(default_factory=list)  # e.g., ["memory lock-in", "agent orchestration"]
    strategic_direction: str = ""  # e.g., "Focusing on enterprise moat-building"
    momentum: str = ""  # e.g., "Accelerating", "Plateauing", "Shifting"

    # Pillar strengths (0-10 scores)
    pillar_strengths: Dict[str, float] = field(default_factory=dict)

    # Competitive positioning
    key_advantages: List[str] = field(default_factory=list)
    key_gaps: List[str] = field(default_factory=list)

    # Confidence (how confident we are in this profile)
    confidence: float = 0.5  # 0-1, increases with more events


class ProviderIntelligence:
    """
    Per-hyperscaler intelligence synthesis.

    Maintains strategic profiles for each provider based on their events.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        database: EventDatabase,
        config: Dict[str, Any]
    ):
        """
        Initialize Provider Intelligence.

        Args:
            llm_provider: LLM for synthesis and analysis
            database: Database for querying historical events
            config: Configuration dict
        """
        self.llm = llm_provider
        self.db = database
        self.config = config

        # Provider profiles (cached in memory)
        self.profiles: Dict[str, ProviderProfile] = {}

        # Configuration
        intel_config = config.get('agents', {}).get('provider_intelligence', {})
        self.profile_window_days = intel_config.get('profile_window_days', 90)  # Look back 90 days
        self.min_events_for_profile = intel_config.get('min_events_for_profile', 5)
        self.update_frequency = intel_config.get('update_frequency', 'per_event')  # or 'daily'

    def update_profile(
        self,
        event: MarketSignalEvent,
        ensemble_metrics: Optional[Any] = None
    ) -> ProviderProfile:
        """
        Update provider profile with new event.

        Called by Ensemble Extractor after each successful extraction.

        Args:
            event: Newly extracted event
            ensemble_metrics: Optional ensemble metrics for quality assessment

        Returns:
            Updated ProviderProfile
        """
        provider = event.provider

        # Get or create profile
        if provider not in self.profiles:
            self.profiles[provider] = ProviderProfile(
                provider=provider,
                last_updated=datetime.now()
            )

        profile = self.profiles[provider]

        # Update statistics
        profile.total_events += 1
        profile.last_updated = datetime.now()

        # Track pillar activity
        for pillar_impact in event.pillars_impacted:
            pillar_name = pillar_impact.pillar_name.value
            profile.events_by_pillar[pillar_name] = profile.events_by_pillar.get(pillar_name, 0) + 1

        # Track recent events
        profile.recent_events.append(event.event_id)
        # Keep only last 50 event IDs
        profile.recent_events = profile.recent_events[-50:]

        # Periodically synthesize strategic profile (every 5 events or when requested)
        if profile.total_events % 5 == 0 or profile.total_events >= self.min_events_for_profile:
            self._synthesize_strategic_profile(profile)

        return profile

    def get_profile(self, provider: str, refresh: bool = False) -> Optional[ProviderProfile]:
        """
        Get provider profile.

        Args:
            provider: Provider name
            refresh: Whether to refresh from database (expensive)

        Returns:
            ProviderProfile or None if not enough data
        """
        if refresh or provider not in self.profiles:
            # Build profile from database
            self._build_profile_from_db(provider)

        profile = self.profiles.get(provider)

        # Only return if we have enough events
        if profile and profile.total_events >= self.min_events_for_profile:
            return profile

        return None

    def compare_profiles(
        self,
        provider_a: str,
        provider_b: str,
        focus_pillar: Optional[Pillar] = None
    ) -> Dict[str, Any]:
        """
        Compare two provider profiles.

        Used by Competitive Reasoning for context before detailed analysis.

        Args:
            provider_a: First provider
            provider_b: Second provider
            focus_pillar: Optional pillar to focus comparison

        Returns:
            Dict with comparison insights
        """
        profile_a = self.get_profile(provider_a, refresh=False)
        profile_b = self.get_profile(provider_b, refresh=False)

        if not profile_a or not profile_b:
            return {
                'error': f'Insufficient data for {provider_a if not profile_a else provider_b}'
            }

        comparison = {
            'providers': [provider_a, provider_b],
            'focus_pillar': focus_pillar.value if focus_pillar else None,
            'strategic_contrasts': self._identify_strategic_contrasts(profile_a, profile_b),
            'momentum_comparison': {
                provider_a: profile_a.momentum,
                provider_b: profile_b.momentum
            },
            'pillar_activity_comparison': self._compare_pillar_activity(
                profile_a, profile_b, focus_pillar
            ),
            'confidence': min(profile_a.confidence, profile_b.confidence)
        }

        return comparison

    def _synthesize_strategic_profile(self, profile: ProviderProfile) -> None:
        """
        Use LLM to synthesize strategic themes and direction.

        Analyzes recent events to identify:
        - Primary strategic themes
        - Overall direction
        - Momentum (accelerating, plateauing, shifting)
        - Competitive positioning

        Args:
            profile: Profile to synthesize
        """
        # Get recent events from database
        events = self.db.search_events(
            provider=profile.provider,
            start_date=datetime.now() - timedelta(days=self.profile_window_days),
            limit=50
        )

        if not events or len(events) < self.min_events_for_profile:
            return

        # Build synthesis prompt
        events_summary = []
        for event in events[-20:]:  # Last 20 events
            events_summary.append({
                'date': event.published_at.strftime('%Y-%m-%d'),
                'what_changed': event.what_changed,
                'pillars': [p.pillar_name.value for p in event.pillars_impacted],
                'advantages_created': event.competitive_effects.advantages_created[:2] if event.competitive_effects.advantages_created else []
            })

        prompt = f"""Analyze this provider's recent activity and synthesize their strategic profile.

**Provider:** {profile.provider}
**Events analyzed:** {len(events)} events over {self.profile_window_days} days
**Pillar distribution:** {json.dumps(profile.events_by_pillar, indent=2)}

**Recent Events:**
{json.dumps(events_summary, indent=2)}

**Your Task:**
Identify strategic patterns and synthesize a profile. Return JSON with:

{{
  "primary_themes": ["2-3 major strategic themes", "e.g., memory lock-in, agent orchestration"],
  "strategic_direction": "1-2 sentence summary of overall strategy",
  "momentum": "Accelerating|Plateauing|Shifting|Lagging",
  "key_advantages": ["Competitive advantages evident from events"],
  "key_gaps": ["Areas where they're behind or absent"],
  "pillar_strengths": {{
    "DATA_PIPELINES": 7.5,
    "TECHNICAL_CAPABILITIES": 8.0,
    // etc (0-10 scores for each pillar based on activity and impact)
  }},
  "confidence": 0.85
}}

Focus on:
1. What are their strategic priorities? (themes)
2. Where are they headed? (direction)
3. Are they accelerating or slowing? (momentum)
4. What advantages have they built? (strengths)
5. Where are they weak/absent? (gaps)

Return ONLY the JSON object."""

        try:
            response = self.llm.generate(
                messages=[
                    {"role": "system", "content": "You are an expert strategic analyst for frontier AI markets."},
                    {"role": "user", "content": prompt}
                ],
                task_complexity="complex",
                temperature=0.3
            )

            # Parse response
            content = response['content'].strip()

            # Strip markdown if present
            if content.startswith('```'):
                lines = content.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                content = '\n'.join(lines).strip()

            synthesis = json.loads(content)

            # Update profile
            profile.primary_themes = synthesis.get('primary_themes', [])
            profile.strategic_direction = synthesis.get('strategic_direction', '')
            profile.momentum = synthesis.get('momentum', '')
            profile.key_advantages = synthesis.get('key_advantages', [])
            profile.key_gaps = synthesis.get('key_gaps', [])
            profile.pillar_strengths = synthesis.get('pillar_strengths', {})
            profile.confidence = synthesis.get('confidence', 0.7)

            print(f"✓ Synthesized profile for {profile.provider} (confidence: {profile.confidence:.2f})")

        except Exception as e:
            print(f"Profile synthesis failed for {profile.provider}: {e}")

    def _build_profile_from_db(self, provider: str) -> None:
        """
        Build profile from historical events in database.

        Args:
            provider: Provider to build profile for
        """
        # Query recent events
        events = self.db.search_events(
            provider=provider,
            start_date=datetime.now() - timedelta(days=self.profile_window_days),
            limit=100
        )

        if not events:
            return

        # Create profile
        profile = ProviderProfile(
            provider=provider,
            last_updated=datetime.now(),
            total_events=len(events)
        )

        # Calculate statistics
        for event in events:
            # Track pillar activity
            for pillar_impact in event.pillars_impacted:
                pillar_name = pillar_impact.pillar_name.value
                profile.events_by_pillar[pillar_name] = profile.events_by_pillar.get(pillar_name, 0) + 1

            # Track recent events (last 30 days)
            days_old = (datetime.now() - event.published_at).days
            if days_old <= 30:
                profile.recent_events.append(event.event_id)

        self.profiles[provider] = profile

        # Synthesize strategic profile if enough events
        if len(events) >= self.min_events_for_profile:
            self._synthesize_strategic_profile(profile)

    def _identify_strategic_contrasts(
        self,
        profile_a: ProviderProfile,
        profile_b: ProviderProfile
    ) -> List[str]:
        """
        Identify key strategic contrasts between two profiles.

        Args:
            profile_a: First provider profile
            profile_b: Second provider profile

        Returns:
            List of contrast strings
        """
        contrasts = []

        # Theme contrasts
        themes_a = set(profile_a.primary_themes)
        themes_b = set(profile_b.primary_themes)
        unique_a = themes_a - themes_b
        unique_b = themes_b - themes_a

        if unique_a:
            contrasts.append(f"{profile_a.provider} focuses on {', '.join(list(unique_a)[:2])}")
        if unique_b:
            contrasts.append(f"{profile_b.provider} focuses on {', '.join(list(unique_b)[:2])}")

        # Momentum contrast
        if profile_a.momentum != profile_b.momentum:
            contrasts.append(f"Momentum: {profile_a.provider} is {profile_a.momentum.lower()}, {profile_b.provider} is {profile_b.momentum.lower()}")

        return contrasts

    def _compare_pillar_activity(
        self,
        profile_a: ProviderProfile,
        profile_b: ProviderProfile,
        focus_pillar: Optional[Pillar]
    ) -> Dict[str, Any]:
        """
        Compare pillar activity between two profiles.

        Args:
            profile_a: First provider profile
            profile_b: Second provider profile
            focus_pillar: Optional pillar to focus on

        Returns:
            Dict with activity comparison
        """
        if focus_pillar:
            pillar_name = focus_pillar.value
            return {
                'pillar': pillar_name,
                profile_a.provider: {
                    'events': profile_a.events_by_pillar.get(pillar_name, 0),
                    'strength': profile_a.pillar_strengths.get(pillar_name, 0.0)
                },
                profile_b.provider: {
                    'events': profile_b.events_by_pillar.get(pillar_name, 0),
                    'strength': profile_b.pillar_strengths.get(pillar_name, 0.0)
                }
            }
        else:
            # Overall comparison
            return {
                'all_pillars': True,
                profile_a.provider: {
                    'total_events': profile_a.total_events,
                    'top_pillars': sorted(
                        profile_a.events_by_pillar.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                },
                profile_b.provider: {
                    'total_events': profile_b.total_events,
                    'top_pillars': sorted(
                        profile_b.events_by_pillar.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                }
            }

    def get_all_profiles_summary(self) -> Dict[str, Any]:
        """
        Get summary of all provider profiles.

        Returns:
            Dict with all profiles
        """
        return {
            provider: {
                'total_events': profile.total_events,
                'primary_themes': profile.primary_themes,
                'momentum': profile.momentum,
                'confidence': profile.confidence,
                'last_updated': profile.last_updated.isoformat()
            }
            for provider, profile in self.profiles.items()
            if profile.total_events >= self.min_events_for_profile
        }
