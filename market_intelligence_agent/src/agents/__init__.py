"""
Market Intelligence Agents

Five specialized agents for competitive intelligence pipeline:
1. Source Scout: Discovers sources to monitor ✓
2. Content Harvester: Fetches and filters content ✓
3. Signal Extractor (v1): Standard single-LLM extraction ✓
4. Signal Extractor (v2): Ensemble multi-LLM extraction with consensus ✓
5. Competitive Reasoning: Analyzes competitive dynamics ✓
6. Analyst Copilot: Chat interface for queries ✓
"""

from .source_scout import SourceScout, SourceCandidate
from .content_harvester import ContentHarvester, HarvestedContent
from .signal_extractor import SignalExtractor
from .ensemble_signal_extractor import EnsembleSignalExtractor, ExtractionResult, EnsembleMetrics
from .competitive_reasoning import CompetitiveReasoning
from .analyst_copilot import AnalystCopilot

__all__ = [
    'SourceScout',
    'SourceCandidate',
    'ContentHarvester',
    'HarvestedContent',
    'SignalExtractor',
    'EnsembleSignalExtractor',
    'ExtractionResult',
    'EnsembleMetrics',
    'CompetitiveReasoning',
    'AnalystCopilot',
]
