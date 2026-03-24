"""
LangGraph Workflows

Orchestrates multi-agent pipelines for ingestion and analysis.

Two main workflows:
1. Ingestion: Source Scout → Content Harvester → Signal Extractor → Storage
2. Analysis: User Query → Parse Intent → Retrieve Events → Reasoning → Format

Why workflows?
- State management across multiple agents
- Error handling and recovery
- Observable execution
- Conditional routing
"""

from .ingestion import (
    create_ingestion_workflow,
    ingest_from_query,
    ingest_from_url,
    IngestionState
)

from .analysis import (
    create_analysis_workflow,
    analyze_query,
    batch_analyze,
    AnalysisState
)

__all__ = [
    # Ingestion workflow
    'create_ingestion_workflow',
    'ingest_from_query',
    'ingest_from_url',
    'IngestionState',

    # Analysis workflow
    'create_analysis_workflow',
    'analyze_query',
    'batch_analyze',
    'AnalysisState',
]
