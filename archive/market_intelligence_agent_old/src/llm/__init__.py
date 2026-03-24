"""
LLM Layer

Multi-provider wrapper, structured outputs, and prompt templates.
"""

from .providers import LLMProvider
from .structured_output import (
    pydantic_to_anthropic_tool,
    pydantic_to_openai_function,
    extract_structured_output,
    create_tool_choice,
    generate_with_structured_output
)
from .prompts import (
    # Signal Extractor
    SIGNAL_EXTRACTOR_SYSTEM_PROMPT,
    SIGNAL_EXTRACTOR_USER_PROMPT,

    # Content Harvester
    CONTENT_SIGNIFICANCE_SYSTEM_PROMPT,
    CONTENT_SIGNIFICANCE_USER_PROMPT,

    # Competitive Reasoning
    LEADERSHIP_RANKING_SYSTEM_PROMPT,
    LEADERSHIP_RANKING_USER_PROMPT,
    COMPARE_PROVIDERS_SYSTEM_PROMPT,
    COMPARE_PROVIDERS_USER_PROMPT,
    EVENT_IMPACT_ANALYSIS_SYSTEM_PROMPT,
    EVENT_IMPACT_ANALYSIS_USER_PROMPT,
    TIMELINE_CONSTRUCTION_SYSTEM_PROMPT,
    TIMELINE_CONSTRUCTION_USER_PROMPT,

    # Analyst Copilot
    COPILOT_INTENT_CLASSIFICATION_SYSTEM_PROMPT,
    COPILOT_INTENT_CLASSIFICATION_USER_PROMPT,
    COPILOT_RESPONSE_FORMATTING_SYSTEM_PROMPT,
    COPILOT_RESPONSE_FORMATTING_USER_PROMPT,

    # Source Scout
    SOURCE_SCOUT_SYSTEM_PROMPT,
    SOURCE_SCOUT_USER_PROMPT,

    # Helpers
    format_events_for_prompt
)

__all__ = [
    # Provider
    "LLMProvider",

    # Structured output
    "pydantic_to_anthropic_tool",
    "pydantic_to_openai_function",
    "extract_structured_output",
    "create_tool_choice",
    "generate_with_structured_output",

    # Prompts
    "SIGNAL_EXTRACTOR_SYSTEM_PROMPT",
    "SIGNAL_EXTRACTOR_USER_PROMPT",
    "CONTENT_SIGNIFICANCE_SYSTEM_PROMPT",
    "CONTENT_SIGNIFICANCE_USER_PROMPT",
    "LEADERSHIP_RANKING_SYSTEM_PROMPT",
    "LEADERSHIP_RANKING_USER_PROMPT",
    "COMPARE_PROVIDERS_SYSTEM_PROMPT",
    "COMPARE_PROVIDERS_USER_PROMPT",
    "EVENT_IMPACT_ANALYSIS_SYSTEM_PROMPT",
    "EVENT_IMPACT_ANALYSIS_USER_PROMPT",
    "TIMELINE_CONSTRUCTION_SYSTEM_PROMPT",
    "TIMELINE_CONSTRUCTION_USER_PROMPT",
    "COPILOT_INTENT_CLASSIFICATION_SYSTEM_PROMPT",
    "COPILOT_INTENT_CLASSIFICATION_USER_PROMPT",
    "COPILOT_RESPONSE_FORMATTING_SYSTEM_PROMPT",
    "COPILOT_RESPONSE_FORMATTING_USER_PROMPT",
    "SOURCE_SCOUT_SYSTEM_PROMPT",
    "SOURCE_SCOUT_USER_PROMPT",
    "format_events_for_prompt",
]
