"""
Source Scout package - discovers and monitors sources for market signals.
"""

from .source_scout import SourceScout, SourceCandidate
from .monitored_source import MonitoredSource, HYPERSCALER_SOURCES, CROSS_PROVIDER_SOURCES, API_SOURCES
from .source_scout_monitor import SourceScoutMonitor
from .link_extractor import LinkExtractor

__all__ = [
    'SourceScout',
    'SourceCandidate',
    'MonitoredSource',
    'HYPERSCALER_SOURCES',
    'CROSS_PROVIDER_SOURCES',
    'API_SOURCES',
    'SourceScoutMonitor',
    'LinkExtractor'
]
