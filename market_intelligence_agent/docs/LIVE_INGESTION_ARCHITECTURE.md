# Live Ingestion Architecture

## Overview

The live ingestion system uses a sophisticated multi-query discovery approach with aggregation, scoring, and validation to ensure only high-quality, novel market signal events are extracted from provider blogs.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    FOR EACH PROVIDER                          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  PHASE 1: MULTI-QUERY DISCOVERY                              │
│                                                               │
│  1. Check RSS Feed Freshness (validation baseline)           │
│     - Get last update timestamp                              │
│     - Used to validate Tavily results                        │
│                                                               │
│  2. Run 5+ Tavily Queries Per Provider                       │
│     OpenAI example:                                           │
│       - "OpenAI GPT model release"                           │
│       - "OpenAI API update platform"                         │
│       - "OpenAI partnership announcement"                    │
│       - "OpenAI safety research alignment"                   │
│       - "OpenAI infrastructure scale"                        │
│                                                               │
│  3. Aggregate Results                                         │
│     - Deduplicate URLs                                        │
│     - Track which queries found each URL                     │
│     - Track query categories (technical, market, alignment)  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  PHASE 2: SCORING & VALIDATION                                │
│                                                               │
│  For Each Discovered URL:                                    │
│                                                               │
│  1. Fetch with ContentHarvester                              │
│     - Get raw content                                         │
│     - LLM analyzes significance (0-10)                       │
│     - Extract metadata                                        │
│                                                               │
│  2. Calculate Combined Score (0-10 points)                   │
│     a) Tavily Relevance (0-3 pts)                            │
│        - How relevant to query                               │
│        - Weighted by query importance                        │
│                                                               │
│     b) Recency (0-3 pts)                                     │
│        - 0 days old = 3 points                               │
│        - 30 days old = 0 points                              │
│        - Linear decay                                         │
│                                                               │
│     c) Harvester Significance (0-3 pts)                      │
│        - LLM scores content importance                       │
│        - Scaled from 0-10 to 0-3                             │
│                                                               │
│     d) Multi-Query Bonus (+1 pt)                             │
│        - If found by 2+ queries                              │
│        - High confidence signal                              │
│                                                               │
│  3. Filter by Threshold                                       │
│     - Min score: 5.0/10                                      │
│     - High priority: 7.0/10+                                 │
│     - Discard low-scoring URLs                               │
│                                                               │
│  4. Compare Against Existing DB                              │
│     - Semantic search for similar events                     │
│     - Check similarity (>0.90 = already covered)            │
│     - Skip if content already in database                    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  PHASE 3: SIGNAL EXTRACTION                                   │
│                                                               │
│  For Each Validated URL:                                     │
│                                                               │
│  1. Extract with SignalExtractor                             │
│     - Already have harvested content                         │
│     - Skip harvest step (optimization)                       │
│     - Parse into MarketSignalEvent                           │
│                                                               │
│  2. Deduplicate Events                                        │
│     - Semantic similarity check (0.85 threshold)             │
│     - Date proximity check (7 days)                          │
│     - Merge if duplicate found                               │
│                                                               │
│  3. Store in Database                                         │
│     - Create new event OR update existing                    │
│     - Add to vector store                                     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## Configuration

```yaml
agents:
  source_scout:
    # Multiple queries per provider
    search_queries:
      openai:
        domain: openai.com
        days_back: 30
        queries:
          - query: "OpenAI GPT model release"
            category: "technical_capabilities"
            weight: 1.0
          - query: "OpenAI API update platform"
            category: "technical_capabilities"
            weight: 0.9
          # ... 3 more queries

    # RSS feeds for validation
    rss_feeds:
      openai: https://openai.com/news/rss.xml
      # ...

    # Scoring thresholds
    scoring:
      min_combined_score: 5.0
      high_priority_threshold: 7.0
      multi_query_bonus: 1.0
```

## Key Advantages

### 1. Multi-Angle Coverage
- 5+ queries per provider catch different event types
- Query categories map to I³ pillars
- Comprehensive coverage of blog content

### 2. Quality Filtering
- Combined scoring (0-10 points) ensures quality
- Only high-scoring content proceeds to extraction
- Reduces noise and false positives

### 3. Smart Deduplication
- Compare before extracting (not after)
- Semantic search against existing events
- Merge/enrich instead of duplicate

### 4. Confidence Signals
- Multi-query hits = high confidence
- Same URL found by multiple queries gets bonus points
- Weighted by query importance

### 5. Efficiency
- Harvester pre-filters significance
- Skip extraction for low-value content
- Reuse harvested content (no re-fetch)

## Scoring Examples

### High Priority (8.5/10)
```
URL: openai.com/index/introducing-gpt-5
- Tavily relevance: 2.8 (found by "GPT model release" query)
- Recency: 3.0 (published today)
- Significance: 2.7 (Harvester scored 9/10)
- Multi-query bonus: 0.0 (found by 1 query)
Total: 8.5/10 → EXTRACT
```

### Marginal Pass (5.2/10)
```
URL: openai.com/blog/partnership-announcement
- Tavily relevance: 1.6 (found by "partnership" query)
- Recency: 1.8 (published 12 days ago)
- Significance: 1.8 (Harvester scored 6/10)
- Multi-query bonus: 0.0 (found by 1 query)
Total: 5.2/10 → CHECK DB (might skip if similar)
```

### Rejected (3.5/10)
```
URL: openai.com/blog/community-update
- Tavily relevance: 0.8
- Recency: 1.2 (published 22 days ago)
- Significance: 1.5 (Harvester scored 5/10)
- Multi-query bonus: 0.0
Total: 3.5/10 → SKIP
```

### Multi-Query Hit (9.0/10)
```
URL: anthropic.com/news/claude-4-release
- Tavily relevance: 2.9
- Recency: 3.0 (published today)
- Significance: 2.1 (Harvester scored 7/10)
- Multi-query bonus: 1.0 (found by 3 queries!)
Total: 9.0/10 → HIGH PRIORITY EXTRACT
```

## Usage

### Discover from All Providers
```python
from src.utils import ingest_live_sources

result = ingest_live_sources(
    database=db,
    vector_store=vector_store,
    llm_provider=llm,
    config=config
)
```

### Specific Providers Only
```python
result = ingest_live_sources(
    database=db,
    vector_store=vector_store,
    llm_provider=llm,
    config=config,
    providers=['OpenAI', 'Anthropic']  # Just these two
)
```

### Manual URL List
```python
result = ingest_live_sources(
    database=db,
    vector_store=vector_store,
    llm_provider=llm,
    config=config,
    sources=['https://openai.com/index/gpt-5']  # Skip discovery
)
```

## Testing

```bash
# Test multi-query discovery (without extraction)
python tests/test_multi_query_discovery.py

# Full live ingestion
python tests/run_live_ingestion.py
```

## Requirements

- `TAVILY_API_KEY` in `.env` (for discovery)
- `ANTHROPIC_API_KEY` in `.env` (for scoring/extraction)
- `OPENAI_API_KEY` in `.env` (for embeddings)

## Performance

- **Discovery**: ~5-10 seconds per provider (5+ Tavily queries)
- **Scoring**: ~2-3 seconds per URL (Harvester fetch + LLM analysis)
- **Extraction**: ~5-10 seconds per event (Signal extraction)

**Total**: ~1-2 minutes for all 5 providers (assuming 2-3 validated URLs per provider)

## Future Enhancements

1. **Parallel query execution** - Run Tavily queries in parallel
2. **Caching** - Cache Tavily results for 1 hour
3. **Adaptive thresholds** - Adjust scoring based on provider history
4. **Category weighting** - Boost certain categories (e.g., model releases)
5. **Source reputation** - Track reliability of specific blog sections
