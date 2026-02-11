# Agent Data Flow & Requirements

This document maps the data flow between agents and specifies what each agent needs from previous stages.

## Overview

```
Source Scout → Content Harvester → Signal Extractor (v1/v2) → [Storage] → Competitive Reasoning → Analyst Copilot
```

---

## 1. Source Scout → Content Harvester

### Source Scout Output: `SourceCandidate`

```python
@dataclass
class SourceCandidate:
    url: str                           # URL to fetch
    provider: str                      # Provider name (e.g., "OpenAI")
    source_type: str                   # "official_blog", "github", "api_llm_benchmarks", etc.
    priority: Literal["high", "medium", "low"]
    confidence: float                  # 0-1, how confident this is a good source
    reasoning: str                     # Why this is a good source
    reliability_score: Optional[float] # From DB if source exists
```

### Content Harvester Needs:

✓ **Currently Has:**
- `url` - Required for fetching
- `provider` - Used for LLM context and event attribution
- `source_type` - Determines which parser to use (web, API, GitHub, etc.)

✓ **Optional but Useful:**
- `priority` - Could influence significance threshold
- `reliability_score` - Could influence retry logic

**Status:** ✅ All required fields present

---

## 2. Content Harvester → Signal Extractor

### Content Harvester Output: `HarvestedContent`

```python
@dataclass
class HarvestedContent:
    url: str
    provider: str
    source_type: str

    # Raw content
    raw_text: str           # Full fetched content
    content_hash: str       # For change detection (SHA256)

    # LLM analysis
    significance_score: int      # 0-10
    content_type: str            # "product_announcement", "partnership", etc.
    filtered_content: str        # Relevant sections only (LLM-filtered)
    metadata: Dict[str, Any]     # Dates, products, competitors mentioned
    llm_reasoning: str           # Why this score

    # Timestamps
    fetched_at: datetime
    fetch_successful: bool
```

### Signal Extractor Needs:

✓ **Currently Has:**
- `filtered_content` - Main extraction input (LLM-filtered relevant text)
- `provider` - Required for event schema
- `url` - Required for event schema (source_url)
- `source_type` - Required for event schema
- `metadata` - Used for additional context in prompts
- `fetched_at` - Can be used as published_at if no better date

✓ **Uses in Extraction:**
- `filtered_content` → Extraction prompt
- `metadata` → Enriches prompt with structured info (dates, product names)
- `content_type` → Could influence pillar mapping (not currently used)
- `significance_score` → Validates worthiness of extraction

**Potential Improvements:**
- ⚠️ `published_at` missing - Currently using `fetched_at` as fallback, but actual publication date would be better
  - **Recommendation:** Content Harvester should attempt to extract publication date from content
  - Implementation: Add date extraction to LLM significance analysis prompt

- ℹ️ `content_type` unused - Could help Signal Extractor pre-select likely pillars
  - Example: "product_announcement" → likely TECHNICAL_CAPABILITIES
  - Example: "partnership" → likely MARKET_SHAPING

**Status:** ✅ All critical fields present, minor enhancements possible

---

## 3. Signal Extractor → Storage → Competitive Reasoning

### Signal Extractor Output: `MarketSignalEvent`

```python
class MarketSignalEvent(BaseModel):
    # Identifiers
    event_id: str
    provider: str
    source_type: str
    source_url: str

    # Timestamps
    published_at: datetime
    retrieved_at: datetime

    # Core event
    what_changed: str              # Concise factual summary
    why_it_matters: str            # Competitive impact explanation
    scope: str                     # "provider-specific", "industry-wide", "cross-sector"

    # I³ Analysis
    pillars_impacted: List[PillarImpact]
    competitive_effects: CompetitiveEffects
    temporal_context: TemporalContext

    # Governance
    alignment_implications: str
    regulatory_signal: Literal["none", "emerging", "material"]
```

**Storage:** Database stores full event as JSON + extracts key fields for indexing

### Competitive Reasoning Needs:

**For Different Query Types:**

#### 1. Event Impact Analysis ("When X released Y, how did that change the market?")
Needs:
- ✅ Single event by ID
- ✅ `what_changed`, `why_it_matters`
- ✅ `competitive_effects` (advantages created/eroded, barriers)
- ✅ `temporal_context.likely_to_trigger_events` (follow-on effects)

#### 2. Provider Comparison ("How do OpenAI and Anthropic differ on memory portability?")
Needs:
- ✅ Events filtered by: provider + pillar + date range
- ✅ `pillars_impacted` (filter by pillar, aggregate by provider)
- ✅ `competitive_effects` (compare advantages)
- ✅ Semantic search (find similar events across providers)

#### 3. Leadership Ranking ("Who is leading on infrastructure openness over last 6 months?")
Needs:
- ✅ Events filtered by: pillar + date range (all providers)
- ✅ `pillars_impacted` with `direction_of_change` and `relative_strength_signal`
- ✅ Scoring formula: Sum(strength × direction) per provider
- ✅ `competitive_effects` as evidence

#### 4. Timeline Analysis ("How did governance expectations evolve this year?")
Needs:
- ✅ Events filtered by: pillar + date range
- ✅ Chronological ordering by `published_at`
- ✅ `temporal_context.preceded_by_events` (causal chains)
- ✅ Semantic clustering (group related events)

**Status:** ✅ All required fields present for all query types

**Database Query Capabilities Needed:**
- ✅ Get event by ID
- ✅ Filter by provider
- ✅ Filter by date range
- ✅ Filter by pillar (requires JSON query or extracted field)
- ⚠️ Semantic search (requires vector store)
- ⚠️ Temporal chain traversal (preceded_by/triggers links)

**Recommendations:**
1. ✅ Already implemented: Basic filtering (provider, date)
2. 🔄 Partially implemented: Vector store exists but not integrated with Competitive Reasoning
3. ❌ Not yet implemented: Pillar-based filtering (need to extract pillar list to indexed column)
4. ❌ Not yet implemented: Temporal chain queries (need graph traversal methods)

---

## 4. Competitive Reasoning → Analyst Copilot

### Competitive Reasoning Output: Structured Analysis

**Expected Output Format:**

```python
# For Leadership Ranking
{
    "query_type": "leadership_ranking",
    "pillar": "DATA_PIPELINES",
    "time_range": {"start": "2024-06-01", "end": "2025-01-01"},
    "rankings": [
        {
            "provider": "OpenAI",
            "score": 8.5,
            "evidence_events": ["evt_...", "evt_..."],
            "key_strengths": ["..."],
            "key_weaknesses": ["..."]
        },
        # ... more providers
    ],
    "analysis": "Narrative explanation of leadership dynamics",
    "confidence": 0.85
}

# For Provider Comparison
{
    "query_type": "provider_comparison",
    "providers": ["OpenAI", "Anthropic"],
    "pillar": "TECHNICAL_CAPABILITIES",
    "differences": [
        {
            "dimension": "Context window strategy",
            "provider_positions": {
                "OpenAI": "Cost optimization via caching",
                "Anthropic": "Raw capacity leadership"
            },
            "evidence": ["evt_...", "evt_..."]
        },
        # ... more differences
    ],
    "analysis": "Narrative comparison",
    "confidence": 0.90
}

# For Event Impact
{
    "query_type": "event_impact",
    "event_id": "evt_...",
    "event_summary": "...",
    "immediate_impact": "...",
    "competitive_shifts": ["...", "..."],
    "triggered_responses": [
        {"provider": "...", "response_event": "evt_...", "timing": "2 weeks later"}
    ],
    "long_term_implications": "...",
    "confidence": 0.80
}

# For Timeline
{
    "query_type": "timeline",
    "pillar": "ALIGNMENT",
    "time_range": {"start": "2024-01-01", "end": "2025-01-01"},
    "timeline": [
        {
            "date": "2024-03-15",
            "event_id": "evt_...",
            "provider": "...",
            "description": "...",
            "significance": "high"
        },
        # ... more events chronologically
    ],
    "narrative": "How governance expectations evolved...",
    "key_trends": ["...", "..."],
    "confidence": 0.85
}
```

### Analyst Copilot Needs:

✓ **For Response Formatting:**
- Structured analysis (as above)
- Evidence event IDs (for linking to sources)
- Confidence scores (to communicate certainty)
- Narrative explanations (for user-friendly output)

✓ **For Interactive Follow-up:**
- Ability to drill down into specific events
- Ability to adjust time ranges
- Ability to compare different periods
- Conversation memory (track what user asked before)

**Status:** 🔄 Needs to be defined when building Competitive Reasoning

**Recommendations:**
1. Competitive Reasoning should return structured dicts (as shown above)
2. Analyst Copilot formats these into natural language + rich tables
3. Include source citations (event IDs, URLs) for transparency
4. Maintain conversation context across turns

---

## Data Flow Validation Checklist

### ✅ Implemented & Tested
- [x] Source Scout → Content Harvester (SourceCandidate)
- [x] Content Harvester → Signal Extractor (HarvestedContent)
- [x] Signal Extractor → Database (MarketSignalEvent)
- [x] Database → Signal Extractor retrieval (get_event, get_events_by_provider)

### 🔄 Partially Implemented
- [x] Vector Store (exists but not integrated with Competitive Reasoning)
- [ ] Pillar-based filtering (need indexed column extraction)
- [ ] Temporal chain traversal (need graph query methods)

### ❌ Not Yet Implemented
- [ ] Database → Competitive Reasoning queries (full query API)
- [ ] Competitive Reasoning → Analyst Copilot (structured output format)
- [ ] Analyst Copilot conversation memory

---

## Recommended Next Steps

1. **Enhance Content Harvester** (optional):
   - Extract publication date from content (not just use fetched_at)
   - Use content_type to inform Signal Extractor

2. **Enhance Database** (required for Competitive Reasoning):
   - Add pillar extraction to indexed column (for fast filtering)
   - Add temporal chain traversal methods (preceded_by, likely_to_trigger)
   - Integrate vector store semantic search

3. **Build Competitive Reasoning** (next priority):
   - Define structured output formats (as shown above)
   - Implement 4 query types
   - Use database + vector store

4. **Build Analyst Copilot** (final):
   - Accept natural language queries
   - Route to appropriate Competitive Reasoning function
   - Format responses with rich output (tables, citations)
   - Maintain conversation context

---

## Summary

**Current State:** ✅ Data flow from Source Scout → Signal Extractor → Storage is complete and tested.

**Blocking Issues:** None - all critical fields are present.

**Nice-to-Haves:**
- Publication date extraction in Content Harvester
- Pillar-based database indexing for fast queries
- Temporal chain traversal methods

**Ready for:** Building Competitive Reasoning agent (next step).
