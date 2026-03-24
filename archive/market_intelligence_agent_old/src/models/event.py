"""
Market Signal Event Schema

The MarketSignalEvent is the atomic unit of the intelligence system.
Each event represents evidence that changes the competitive balance.

PRD Reference: Section 5 - Event Schema
"""

from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator

from .i3_pillars import Pillar, DirectionOfChange, RelativeStrength


class PillarImpact(BaseModel):
    """
    How a single event impacts one I³ pillar.

    An event can impact multiple pillars (e.g., a memory API affects both
    DATA_PIPELINES and TECHNICAL_CAPABILITIES).
    """

    pillar_name: Pillar = Field(
        description="Which I³ pillar is impacted"
    )

    direction_of_change: DirectionOfChange = Field(
        description="Does this advance, constrain, or neutrally affect the pillar?"
    )

    relative_strength_signal: RelativeStrength = Field(
        description="How strong is this signal? (STRONG, MODERATE, WEAK)"
    )

    evidence: str = Field(
        min_length=10,
        description="Specific evidence from source content supporting this classification. "
                    "Should quote or paraphrase key text."
    )

    def calculate_impact_score(self) -> float:
        """Calculate numeric impact score for this pillar."""
        from .i3_pillars import calculate_pillar_impact_score
        return calculate_pillar_impact_score(
            self.direction_of_change,
            self.relative_strength_signal
        )


class CompetitiveEffects(BaseModel):
    """
    Competitive advantages, disadvantages, and barriers created by an event.

    This is MVP-critical (PRD Section 5.4) - explains WHO benefits and WHO is pressured.
    """

    advantages_created: List[str] = Field(
        default_factory=list,
        description="What advantages does this create for the provider? "
                    "What can they now do that competitors can't?"
    )

    advantages_eroded: List[str] = Field(
        default_factory=list,
        description="Which competitors' advantages are weakened? "
                    "Whose competitive moats are reduced? Be specific about WHO."
    )

    new_barriers: List[str] = Field(
        default_factory=list,
        description="What barriers to entry or switching are created?"
    )

    lock_in_or_openness_shift: str = Field(
        description="Does this increase vendor lock-in or improve openness/portability? "
                    "How does it affect customer switching costs?"
    )


class TemporalContext(BaseModel):
    """
    Temporal relationships and causal chains.

    This is MVP-critical (PRD Section 5.5) - enables timeline analysis and
    distinguishes first-move (innovation) from reactive (response) events.
    """

    preceded_by_events: List[str] = Field(
        default_factory=list,
        description="Event IDs that led to this event. "
                    "Use to build causal chains and identify reactions."
    )

    likely_to_trigger_events: List[str] = Field(
        default_factory=list,
        description="Predicted future events this will cause. "
                    "E.g., 'Anthropic will likely announce context window increase' "
                    "(store as event description, not ID since it hasn't happened yet)"
    )

    time_horizon: Literal["immediate", "medium", "long"] = Field(
        description="How quickly will this impact the market? "
                    "immediate (days-weeks), medium (months), long (year+)"
    )


class MarketSignalEvent(BaseModel):
    """
    Complete Market Signal Event structure.

    This is the core data structure of the intelligence system.

    PRD Quote: "A Market Signal Event is evidence that changes the
    competitive balance along one or more I³ pillars."
    """

    # =========================================================================
    # Event Metadata (PRD Section 5.1)
    # =========================================================================

    event_id: str = Field(
        description="Unique identifier. Format: evt_{provider}_{slug}_{date}"
    )

    provider: str = Field(
        description="Provider name (e.g., 'OpenAI', 'Anthropic', 'Google', 'Microsoft')"
    )

    source_type: str = Field(
        description="Type of source (e.g., 'official_blog', 'github', 'documentation', 'pdf')"
    )

    source_url: str = Field(
        description="URL where this signal was found"
    )

    published_at: datetime = Field(
        description="When the provider published this (from source metadata)"
    )

    retrieved_at: datetime = Field(
        default_factory=datetime.now,
        description="When our system retrieved this content"
    )

    # =========================================================================
    # Event Description (PRD Section 5.2)
    # =========================================================================

    what_changed: str = Field(
        min_length=20,
        description="Concise, factual description of what happened. "
                    "E.g., 'OpenAI increased GPT-4 context window from 128K to 200K tokens'"
    )

    why_it_matters: str = Field(
        min_length=30,
        description="Mechanism of competitive impact, not opinion. "
                    "Explain HOW this changes the competitive landscape."
    )

    scope: str = Field(
        min_length=10,
        description="Who is affected? (regions, users, APIs, partners, etc.)"
    )

    # =========================================================================
    # I³ Classification (PRD Section 5.3)
    # =========================================================================

    pillars_impacted: List[PillarImpact] = Field(
        min_length=1,
        description="At least one pillar must be impacted for this to be a valid market signal"
    )

    # =========================================================================
    # Competitive Impact (PRD Section 5.4 - MVP Critical)
    # =========================================================================

    competitive_effects: CompetitiveEffects = Field(
        description="Competitive advantages, disadvantages, and barriers created"
    )

    # =========================================================================
    # Temporal Context (PRD Section 5.5 - MVP Critical)
    # =========================================================================

    temporal_context: TemporalContext = Field(
        description="Causal relationships with other events"
    )

    # =========================================================================
    # Governance & Risk Lens (PRD Section 5.6)
    # =========================================================================

    alignment_implications: str = Field(
        description="How does this affect AI safety, ethics, or governance?"
    )

    regulatory_signal: Literal["none", "emerging", "material"] = Field(
        default="none",
        description="Is there regulatory relevance? "
                    "none: no regulatory implications, "
                    "emerging: potential future regulatory attention, "
                    "material: significant regulatory implications"
    )

    # =========================================================================
    # Metadata for Quality Assurance
    # =========================================================================

    extraction_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence score from Signal Extractor (0.0-1.0). "
                    "Values < 0.7 may need human review."
    )

    extraction_notes: Optional[str] = Field(
        default=None,
        description="Notes from Signal Extractor about ambiguities or assumptions"
    )

    @field_validator('pillars_impacted')
    @classmethod
    def validate_pillars(cls, v: List[PillarImpact]) -> List[PillarImpact]:
        """Ensure at least one pillar is impacted."""
        if not v:
            raise ValueError("Event must impact at least one pillar")
        return v

    @field_validator('event_id')
    @classmethod
    def validate_event_id(cls, v: str) -> str:
        """Ensure event_id follows naming convention."""
        if not v.startswith('evt_'):
            raise ValueError("event_id must start with 'evt_'")
        return v

    def get_primary_pillar(self) -> Pillar:
        """
        Get the pillar with strongest impact.

        Used when we need to categorize event by single pillar.
        """
        return max(
            self.pillars_impacted,
            key=lambda p: p.calculate_impact_score()
        ).pillar_name

    def is_first_move(self) -> bool:
        """
        Is this a first-move (innovation) or reactive (response) event?

        First-move: No preceding events (or only external market forces)
        Reactive: Has preceding events from competitors
        """
        return len(self.temporal_context.preceded_by_events) == 0

    def total_impact_score(self) -> float:
        """
        Calculate total competitive impact across all pillars.

        Used in leadership ranking calculations.
        """
        return sum(impact.calculate_impact_score() for impact in self.pillars_impacted)
