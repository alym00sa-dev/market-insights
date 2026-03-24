"""
Provider Profile Schema

Provider profiles store persistent memory about hyperscalers:
- Strengths by pillar
- Known strategies
- Historical behavior patterns

This enables the system to provide context in competitive analysis.

PRD Reference: Section 8 - Persistent Memory Requirements
"""

from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .i3_pillars import Pillar


class ProviderProfile(BaseModel):
    """
    Persistent profile for a hyperscaler provider.

    This is the system's "memory" about a provider, built up over time
    from analyzing their Market Signal Events.
    """

    provider_name: str = Field(
        description="Provider name (e.g., 'OpenAI', 'Anthropic', 'Google', 'Microsoft')"
    )

    description: Optional[str] = Field(
        default=None,
        description="Brief description of the provider and their focus"
    )

    strengths_by_pillar: Dict[Pillar, str] = Field(
        default_factory=dict,
        description="For each pillar, a summary of the provider's strengths. "
                    "E.g., DATA_PIPELINES: 'Strong focus on portability and GDPR compliance'"
    )

    known_strategies: List[str] = Field(
        default_factory=list,
        description="Strategic patterns identified from events. "
                    "E.g., 'Enterprise-first approach', 'Walled garden strategy', "
                    "'Public-benefit model prioritizes openness'"
    )

    historical_behavior_patterns: List[str] = Field(
        default_factory=list,
        description="Behavioral patterns observed over time. "
                    "E.g., 'Tends to react quickly to competitor moves', "
                    "'Announces features before full availability', "
                    "'Favors large partnerships over grassroots community'"
    )

    # Metadata
    first_event_date: Optional[datetime] = Field(
        default=None,
        description="Date of first event ingested for this provider"
    )

    last_event_date: Optional[datetime] = Field(
        default=None,
        description="Date of most recent event ingested"
    )

    total_events: int = Field(
        default=0,
        description="Total number of events ingested for this provider"
    )

    last_updated: datetime = Field(
        default_factory=datetime.now,
        description="When this profile was last updated"
    )

    def get_pillar_strength_summary(self, pillar: Pillar) -> str:
        """
        Get strength summary for a specific pillar.

        Returns:
            Strength description or "No data" if pillar not analyzed yet
        """
        return self.strengths_by_pillar.get(
            pillar,
            f"No data yet for {pillar.value}"
        )

    def add_strategy(self, strategy: str) -> None:
        """
        Add a newly identified strategy (if not already present).

        Used by Competitive Reasoning Agent when it identifies patterns.
        """
        if strategy not in self.known_strategies:
            self.known_strategies.append(strategy)
            self.last_updated = datetime.now()

    def add_behavior_pattern(self, pattern: str) -> None:
        """
        Add a newly identified behavioral pattern.

        Used by Competitive Reasoning Agent for longitudinal analysis.
        """
        if pattern not in self.historical_behavior_patterns:
            self.historical_behavior_patterns.append(pattern)
            self.last_updated = datetime.now()

    def update_pillar_strength(self, pillar: Pillar, strength_description: str) -> None:
        """
        Update or set pillar strength description.

        Called when new events change our understanding of provider's strengths.
        """
        self.strengths_by_pillar[pillar] = strength_description
        self.last_updated = datetime.now()

    def increment_event_count(self, event_date: datetime) -> None:
        """
        Increment event count and update date tracking.

        Called whenever a new event is ingested for this provider.
        """
        self.total_events += 1

        if self.first_event_date is None or event_date < self.first_event_date:
            self.first_event_date = event_date

        if self.last_event_date is None or event_date > self.last_event_date:
            self.last_event_date = event_date

        self.last_updated = datetime.now()


# Default provider profiles (seed data)
DEFAULT_PROVIDERS = {
    "OpenAI": ProviderProfile(
        provider_name="OpenAI",
        description="Leading frontier AI company, developer of GPT models and ChatGPT"
    ),
    "Anthropic": ProviderProfile(
        provider_name="Anthropic",
        description="Public benefit AI company, developer of Claude"
    ),
    "Google": ProviderProfile(
        provider_name="Google",
        description="Tech giant with Gemini models, Google Workspace, and education focus"
    ),
    "Microsoft": ProviderProfile(
        provider_name="Microsoft",
        description="Enterprise-focused hyperscaler with Copilot, Azure OpenAI, and M365"
    )
}


def get_or_create_provider(provider_name: str) -> ProviderProfile:
    """
    Get default provider profile or create new one.

    Used when we encounter a provider for the first time.
    """
    return DEFAULT_PROVIDERS.get(
        provider_name,
        ProviderProfile(provider_name=provider_name)
    )
