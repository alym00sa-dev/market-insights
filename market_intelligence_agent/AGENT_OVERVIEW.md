# Market Intelligence Agent — Overview

A continuously updated AI market intelligence system that scrapes news, extracts structured events, stores them in a knowledge graph, and answers natural language questions about the competitive AI landscape.

---

## Players Tracked

Seven first-class nodes in the graph, each with a set of aliases used for article tagging:

| Key | Name | Notes |
|---|---|---|
| `openai` | OpenAI | ChatGPT, GPT-4/5, o1, o3, Sora, DALL-E, Whisper, Operator |
| `anthropic` | Anthropic | Claude, Claude Sonnet/Opus/Haiku, Constitutional AI |
| `google` | Google / DeepMind | Gemini, Bard, Vertex AI, NotebookLM, Imagen, AlphaCode |
| `meta` | Meta AI | Llama, LLaMA, Facebook AI, FAIR, PyTorch |
| `microsoft` | Microsoft | Copilot, Azure OpenAI, Azure AI, Bing AI, GitHub Copilot |
| `nvidia` | NVIDIA | CUDA, NIM, DGX, Hopper, Blackwell, NeMo, Jetson, H100/H200/GB200 |
| `emerging` | Emerging Players | Catch-all: xAI/Grok, Mistral, Cohere, Perplexity, DeepSeek, Amazon Bedrock, Apple Intelligence, Qwen, Scale AI, Cognition/Devin, Thinking Machines Lab, and others |

Player tagging uses word-boundary regex matching against article titles (primary) and summaries (fallback). The `emerging` catch-all only fires if no primary player matched.

---

## Agent Architecture at Query Time

When a user submits a question through the chat interface, the following pipeline runs:

```
User question
     │
     ▼
┌─────────────────┐
│  Query Manager  │  Decomposes question → identifies relevant players,
│  (Coordinator)  │  search terms, time frame
└────────┬────────┘
         │
         ├─────────────────────────────────┐
         ▼                                 ▼
┌─────────────────┐              ┌──────────────────┐
│  Research Lead  │  (one per    │   Live Web Search │  Claude web_search
│  (Team Lead)    │   player,    │                   │  tool for fresh context
└────────┬────────┘   parallel)  └──────────────────┘
         │
         ├── KG fetch: recent events + search-term events for player
         │
         ├─────────────────────────────────┐
         ▼                                 ▼
┌──────────────────┐           ┌──────────────────────┐
│  Sub-Researcher  │ × 4       │  Sub-Researcher  × 4  │  Four focus lenses,
│  (Strategist)    │ parallel  │  (Strategist)         │  all see same events
└──────────────────┘           └──────────────────────┘
         │
         ▼
┌─────────────────┐
│   Synthesis     │  Combines sub-researcher findings into a cited brief
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Assembler    │  Merges briefs from all players + web context
│  (Final Draft)  │  into a prose response with inline citations
└─────────────────┘
```

**Sub-researcher focus areas (run in parallel):**
1. Product launches, model releases, technical capabilities
2. Partnerships, integrations, business deals, market positioning
3. Policy, governance, regulation, public statements
4. Funding, hiring, infrastructure, organizational changes

**Two query modes:**

- **Synthesizer** (default) — full pipeline above; returns an AI-generated prose brief with inline citations. Takes 30–90 seconds.
- **Retriever** — skips all LLM synthesis; returns a ranked list of raw KG articles sorted by date (desc) then significance. Near-instant. Supports natural language count requests ("show me 5 articles", default: 10).

**Models used:**
- Query decomposition, sub-research, synthesis, assembly: `claude-sonnet-4-6` (default, switchable in sidebar)
- Live web search: `claude-sonnet-4-6` with `web_search_20250305` tool
- Extraction / backfill scoring: `claude-opus-4-7` (higher quality for ingestion)

---

## Knowledge Graph

**Database:** Neo4j AuraDB Free tier (permanent, cloud-hosted, Streamlit Cloud compatible)

**Graph model:**

```
(:Player)-[:INVOLVED_IN]->(:Event)
```

**Player node properties:**

| Property | Type | Description |
|---|---|---|
| `key` | String (unique) | e.g. `openai`, `anthropic` |
| `name` | String | Display name |
| `aliases` | List | Strings used for article tagging |
| `description` | String | Short description |
| `is_catch_all` | Boolean | True for Emerging Players only |

**Event node properties:**

| Property | Type | Description |
|---|---|---|
| `key` | String (UUID) | Unique event ID |
| `title` | String | Event headline |
| `description` | String | Full paragraph description |
| `event_type` | String | See taxonomy below |
| `significance_score` | Integer (1–10) | Scored by Opus at extraction time |
| `sentiment` | String | `positive`, `negative`, `neutral` |
| `analyst_notes` | String | Competitive implications |
| `published_date` | String (ISO) | Publication date of first source |
| `first_seen` | String (ISO) | When the event entered the KG |
| `last_updated` | String (ISO) | Last source appended |
| `source_count` | Integer | Number of corroborating sources |
| `sources_json` | String (JSON) | Serialized list of source objects |
| `raw_content_hash` | String (MD5) | For fast deduplication |

**Event types:** `product_launch`, `partnership`, `funding`, `hiring`, `policy`, `research`, `infrastructure`, `legal`, `earnings`, `acquisition`, `model_benchmark`, `other`

**Significance score taxonomy (scored by Opus):**

| Score | Meaning |
|---|---|
| 9–10 | Transformative: paradigm shift, landmark deal |
| 7–8 | Major: significant product launch, large funding round |
| 5–6 | Notable: meaningful update, partnership worth tracking |
| 3–4 | Minor: incremental update, routine announcement |
| 1–2 | Noise: tangential mention, low relevance |

Events scoring below 5 are filtered out before insertion.

**Deduplication (two layers):**
1. MD5 hash of `url + title` — fast exact match, no LLM call
2. LLM semantic check against the 10 most recent events for the same player — catches the same story reported by different outlets. On semantic duplicate, the new source is appended to `sources_json` rather than creating a new node (multi-source corroboration model).

**Indexes:** `published_date`, `event_type`, `first_seen`, unique constraint on `raw_content_hash`

---

## News Sources

### RSS Feeds (live pipeline, every 4 hours)

| Source | Tier | Notes |
|---|---|---|
| The Decoder | 1 | Pure AI news, very high relevance |
| TechCrunch | 1 | Strong for funding, partnerships, launches |
| The Verge | 1 | Hyperscaler product coverage |
| Wired AI | 1 | Policy and product depth |
| Techmeme | 1 | Aggregator — surfaces top stories |
| Bloomberg Tech | 1 | Business and financial angle |
| MIT Technology Review | 1 | Technical depth and policy |
| NYT Technology | 1 | NYT tech section |
| NYT Business | 1 | AI deals, funding, earnings |
| Financial Times | 1 | Strong on deals and policy; paywall limits to headline + abstract |
| VentureBeat AI | 1 | AI-focused, product launches and model releases |
| Google DeepMind Blog | 1 | First-party Google AI announcements |
| Meta AI Engineering Blog | 1 | First-party Meta AI posts |
| Microsoft AI Blog | 1 | First-party Microsoft AI announcements |

**Blocked (403):** OpenAI Blog, Anthropic Blog — covered via news queries instead.

### Per-Player News API Queries (live pipeline)

Targeted keyword searches run against The Guardian API and LLM web search at each pipeline cycle. Examples: `"OpenAI"`, `"Claude AI"`, `"NVIDIA Blackwell"`, `"DeepSeek AI"`.

---

## Backfill Setup

The backfill script (`scripts/backfill_historical.py`) populates the KG from a historical start date. It runs three collectors in sequence, then extracts and inserts:

**1. Guardian API** — paginated full-text search for each player's `news_queries`, up to 5 pages per query (200 results/page), date-filtered from `FROM_DATE` to today.

**2. LLM Web Search** — one targeted search per player per month using Claude's `web_search` tool. For a date range of Jan 2026–May 2026 (5 months × 7 players = 35 searches).

**3. NYT Archive API** — full monthly archive pull, filtered to AI-relevant articles. Rate-limited to 1 request per 12 seconds.

After collection, articles are tagged to players via alias matching, saved to a **checkpoint file** (`scripts/backfill_checkpoint_YYYY-MM-DD.json`) before extraction begins. On restart, the checkpoint is loaded directly — skipping the fetch phase entirely.

Each article then goes through Opus extraction and the signal pipeline (dedup → insert).

**Current backfill coverage:** January 2026 → present (smoke test). Full 2024 backfill pending.

---

## Expansion Opportunities

### Data Sources
- **Crunchbase / PitchBook API** — structured funding and acquisition data; would significantly improve coverage of the financial intelligence layer
- **arXiv / Semantic Scholar** — research paper tracking; currently we catch research via news coverage only, missing pre-publication signals
- **Patent filings** — USPTO/EPO API for IP intelligence on model architecture and hardware
- **LinkedIn / hiring signals** — job posting trends as a leading indicator of player investment areas
- **Earnings call transcripts** — structured parsing of Q&A sections for forward-looking AI statements
- **SEC filings** — 10-K/10-Q mentions of AI investment and risk

### Players
- **DeepSeek** — promote from Emerging to a first-class node given its competitive significance
- **Amazon** — AWS Bedrock / Nova / Trainium as a distinct hyperscaler node
- **Apple** — Apple Intelligence as a distinct player node
- **Salesforce, ServiceNow, Adobe** — enterprise AI application layer

### Agent Capabilities
- **Trend analysis** — time-series view of event frequency and significance per player over rolling windows
- **Competitive diff** — "what changed for OpenAI since last week?" using stored KG snapshots
- **Alert system** — significance ≥ 8 events trigger a notification (email, Slack) regardless of app sleep state
- **Gates Foundation context layer** — overlay GF portfolio companies, investment theses, and program area priorities to surface relevant signals
- **Vertical filtering** — filter events by relevance to specific sectors (global health, agriculture, education, financial services)
- **Entity extraction** — named persons, dollar amounts, and geographic regions as additional node types in the graph
