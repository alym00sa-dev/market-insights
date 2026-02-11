"""
I³ Index (Hyperscaler Innovation Influence Index) Pillar Definitions

The I³ Index examines hyperscaler influence across 5 core dimensions.
Each Market Signal Event must be mapped to at least one pillar.
"""

from enum import Enum
from typing import Dict


class Pillar(str, Enum):
    """
    The 5 pillars of the I³ Index.

    Why str, Enum?
    - str: Allows JSON serialization directly
    - Enum: Type safety, prevents typos
    """

    DATA_PIPELINES = "DATA_PIPELINES"
    """
    Control over data resources, interoperability, and standards.

    Examples:
    - Memory export APIs (portability)
    - Open data schemas (Ed-Fi, LIF, CTDL)
    - Data governance policies
    - API interoperability
    """

    TECHNICAL_CAPABILITIES = "TECHNICAL_CAPABILITIES"
    """
    AI infrastructure, compute power, model orchestration, and innovation tooling.

    Examples:
    - Context window increases
    - New model releases
    - Agent orchestration frameworks
    - Infrastructure (TPUs, accelerators)
    - Platform features (streaming, multimodal)
    """

    EDUCATION_INFLUENCE = "EDUCATION_INFLUENCE"
    """
    Breadth and impact of education, skills, and advisory programs.

    Examples:
    - AI training partnerships
    - Teacher academies
    - Certification programs
    - Educational grants/funding
    - Public literacy initiatives
    """

    MARKET_SHAPING = "MARKET_SHAPING"
    """
    Cross-sector ecosystem orchestration, partnerships, and alliances.

    Examples:
    - Strategic partnerships (e.g., OpenAI + Microsoft)
    - Standards body participation (Frontier Model Forum)
    - Startup funding/acquisitions
    - Open source community building
    """

    ALIGNMENT = "ALIGNMENT"
    """
    Responsible AI practices, ethics, governance, and regulatory engagement.

    Examples:
    - Safety disclosures (system cards, red team reports)
    - Governance frameworks
    - Compliance certifications (SOC 2, GDPR)
    - Transparency reports
    - Content moderation policies
    """


class DirectionOfChange(str, Enum):
    """
    Direction of competitive positioning change for a pillar.

    Used to assess whether an event strengthens or weakens a provider's position.
    """

    ADVANCE = "ADVANCE"
    """
    Strengthens competitive position.

    Examples:
    - Launching new capability competitors lack
    - Opening up previously closed systems
    - Investing in ecosystem growth
    """

    NEUTRAL = "NEUTRAL"
    """
    Maintains current position, no significant shift.

    Examples:
    - Minor updates or patches
    - Clarifications of existing policy
    - Routine maintenance announcements
    """

    CONSTRAIN = "CONSTRAIN"
    """
    Weakens competitive position.

    Examples:
    - Removing previously available features
    - Increasing lock-in (reducing portability)
    - Restricting API access
    - Controversial policy changes
    """


class RelativeStrength(str, Enum):
    """
    Magnitude of the signal's impact on competitive dynamics.

    Used to weight events when calculating leadership scores.
    """

    STRONG = "STRONG"
    """
    Major competitive move, industry-leading.

    Characteristics:
    - First-to-market innovation
    - Significant technical advancement (e.g., 2x improvement)
    - Forces competitors to respond
    - Changes customer expectations

    Examples:
    - GPT-4 200K context (industry-leading)
    - First memory export API (GDPR compliance)
    - Major partnership (Microsoft + OpenAI scale)
    """

    MODERATE = "MODERATE"
    """
    Notable move, but not industry-leading.

    Characteristics:
    - Follows competitor precedent
    - Incremental improvement
    - Affects specific use cases, not universal

    Examples:
    - Memory TTL controls (useful but not revolutionary)
    - Education partnership with single institution
    - Minor API additions
    """

    WEAK = "WEAK"
    """
    Minor signal, limited competitive impact.

    Characteristics:
    - Small incremental change
    - Catch-up to industry standard
    - Limited scope or reach

    Examples:
    - Bug fixes
    - Documentation updates
    - Routine policy clarifications
    """


# Display names for UI/reporting
PILLAR_DISPLAY_NAMES: Dict[Pillar, str] = {
    Pillar.DATA_PIPELINES: "Data Pipelines & Standards",
    Pillar.TECHNICAL_CAPABILITIES: "Technical Capabilities & Platforms",
    Pillar.EDUCATION_INFLUENCE: "Education & Advisory Influence",
    Pillar.MARKET_SHAPING: "Market Shaping & Partnerships",
    Pillar.ALIGNMENT: "Alignment / Governance"
}


# Numeric scores for calculations (leadership ranking)
STRENGTH_SCORES: Dict[RelativeStrength, float] = {
    RelativeStrength.STRONG: 1.0,
    RelativeStrength.MODERATE: 0.67,
    RelativeStrength.WEAK: 0.33
}

DIRECTION_SCORES: Dict[DirectionOfChange, float] = {
    DirectionOfChange.ADVANCE: 1.0,
    DirectionOfChange.NEUTRAL: 0.0,
    DirectionOfChange.CONSTRAIN: -1.0
}


def get_pillar_display_name(pillar: Pillar) -> str:
    """Get human-readable pillar name for UI display."""
    return PILLAR_DISPLAY_NAMES[pillar]


def calculate_pillar_impact_score(
    direction: DirectionOfChange,
    strength: RelativeStrength
) -> float:
    """
    Calculate numeric score for a pillar impact.

    Used in leadership ranking calculations.

    Returns:
        Score between -1.0 and 1.0
        - Positive: Advances position
        - Negative: Constrains position
        - Zero: Neutral
    """
    return DIRECTION_SCORES[direction] * STRENGTH_SCORES[strength]
