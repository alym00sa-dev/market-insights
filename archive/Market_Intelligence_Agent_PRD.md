# Market Intelligence Agent — Product Requirements Document

**Version:** 1.1 | **Date:** March 2026 | **Status:** v1 Complete
**Target:** Claude Code Implementation | **Stack:** Python, Streamlit, ArangoDB

---

## Build Status (as of March 2026)

All four phases are complete and running. The system is live on `localhost:8502`.

### What Was Built

**Players:** 6 tracked — OpenAI, Anthropic, Google/DeepMind, Meta AI, Microsoft, and an Emerging Players catch-all (xAI, Mistral, Cohere, Perplexity, etc.).

**Sourcing:** 5 scrapers active — RSS (general + first-party player blogs), The Guardian API (historical back to Aug 2025), NewsAPI placeholder, Anthropic direct web scrape (no public RSS), LLM-native web search (Claude `web_search_20250305` tool / OpenAI `gpt-4o-search-preview`).

**Event Schema (actual):** `title`, `description` (LLM-written paragraph — primary record), `scraped_content` (raw source text), `event_type`, `sentiment`, `significance_score`, `key_entities`, `analyst_notes`, `sources[]` (array for multi-source corroboration), `source_count`, `first_seen`, `last_updated`, `raw_content_hash`, `player_keys`.

**Multi-source corroboration:** Semantic duplicates are not discarded — the new source URL is appended to the existing event's `sources[]` array and `source_count` increments. Source count reflects how many outlets covered the story.

**Player tagging:** Title-first strategy — if a player alias appears in the article title, only those players are tagged. Summary is only used as a fallback. Prevents articles that merely mention a player in passing from polluting that player's tab.

**Query routing:** General queries (no specific company named) always route to all 5 hyperscalers — enforced in code, not just prompt instructions.

**Conversation history:** Last 4 exchanges passed as context to the assembler for follow-up question support.

**UI — Agent Chat page:**
- Model selector (Claude Sonnet 4.6, Claude Opus 4.6, Claude Haiku 4.5, GPT-4o, GPT-4o mini) — switchable per query
- URL Analysis toggle — when on, any URL in a chat message is auto-fetched, analyzed, and shown before the agent response
- Reset Chat button
- Response tone: plain news briefing prose, inline citations as `([Source, Date](URL))`

**UI — Dashboard page:**
- 5 top metrics: Knowledge Start date, Last Updated date, Total Events, This Week, Today
- Sidebar: pipeline controls (Run Now button, last run stats, next scheduled run), source health per outlet
- Main area: per-player event tabs with full description, analyst notes, significance score, sentiment, source count badge

**Scheduler:** APScheduler `BackgroundScheduler`, configurable interval via `SCRAPE_INTERVAL_HOURS`, starts on dashboard page load.

**Historical backfill:** `backfill_historical.py` — Guardian API paginated from any start date + LLM web search month-by-month per player. Completed backfill from Aug 2025 → Mar 2026.

**Known limitations:**
- Hyperscaler/time frame sidebar filters on the chat page are not yet wired into `ask()` (commented out with TODO)
- No charts on dashboard yet (deferred)
- No user authentication

---

## 1. Overview

This PRD defines a Market Intelligence Agent that autonomously scrapes the web and news for updates about the major AI hyperscalers (OpenAI, Anthropic, Google, Meta, Microsoft), stores structured data in an ArangoDB knowledge graph, and surfaces insights through an interactive Streamlit application.

The system follows a multi-agent architecture where a central supervisor orchestrates per-hyperscaler research teams, each responsible for news extraction, knowledge graph updating, and competitive reasoning. A shared sourcing layer feeds raw content to all teams, while a query manager decomposes user questions and routes them to the appropriate research teams for KG-grounded answers.

---

## 2. Goals and Non-Goals

### Goals

- **Automated intelligence collection** — Continuously scrape and aggregate news, blog posts, press releases, and announcements about the five hyperscalers via RSS feeds, news APIs, and direct web scraping.
- **Knowledge graph storage** — Persist all extracted events and their relationships to AI players in an ArangoDB graph database with a two-node schema (`AI_Player` ↔ `Event`).
- **Interactive Q&A** — Allow users to ask natural language questions about hyperscaler activity and receive grounded, citation-backed answers synthesized by per-player research teams from the knowledge graph.
- **Operational transparency** — Provide a scraping dashboard showing last scrape times, source health, ingestion counts, and recent events per player.
- **Configurable LLM backend** — Support swapping between Claude (Anthropic API), OpenAI, and other LLM providers via a single config change.

### Non-Goals (v1)

- Vertical-specific analysis (Education, Agriculture, Health) — deferred to v2.
- Real-time streaming or push notifications.
- User authentication or multi-tenancy.
- Predictive analytics or forecasting.
- Coverage of companies beyond the five hyperscalers.

---

## 3. System Architecture

The system consists of five layers: **Orchestration**, **Research Teams**, **Shared Sourcing**, **Knowledge Graph**, and **Presentation**.

### 3.1 Orchestration Layer

#### Market Agent Supervisor ("Managing Director")

The top-level orchestrator that manages the full scraping pipeline. It schedules scraping runs on a configurable interval (default: every 4 hours), dispatches work to research teams, monitors pipeline health, and logs run metadata. Implemented as a Python class using an agentic framework (e.g., LangGraph or a lightweight custom orchestrator).

#### Market Agent Query Manager ("Portfolio Coordinator")

Handles all user-facing queries from the Streamlit chat interface. When a user asks a question, the Query Manager:

1. **Decomposes the request** — Parses the user's question to identify which hyperscaler(s) are relevant, what time frame matters, and what type of information is being asked for.
2. **Adds context** — Enriches the decomposed query with any necessary framing (e.g., clarifying ambiguous terms, setting scope).
3. **Routes to Research Lead(s)** — Dispatches the enriched query to the appropriate Research Lead(s). If the query spans multiple players (e.g., "Compare OpenAI and Google's infrastructure investments"), the Query Manager fans out to multiple Research Leads **in parallel**.
4. **Assembles the final response** — Collects briefs from each Research Lead, synthesizes them into a coherent, cited response, and returns it to the user.

The Query Manager does **not** query the KG directly — it delegates all retrieval and analysis to the research teams.

### 3.2 Research Teams (Per Hyperscaler)

There is one **Research Lead ("Team Lead")** per hyperscaler: OpenAI, Anthropic, Google, Meta, Microsoft. Each team handles two distinct workflows:

#### Ingestion Pipeline (Scheduled)

These are structured pipeline tasks (not autonomous agents) that run during each scrape cycle:

- **Content Extraction ("The QA")** — Parses raw scraped content into structured event data: title, summary, date, source URL, event type (product launch, partnership, funding, hiring, policy, research, infrastructure, legal, earnings, other), key entities mentioned, and sentiment.
- **Information Signal ("The Analyst")** — Scores events for significance and novelty. Runs two-layer deduplication (see Section 3.6). Tags events with relevance categories.

#### Query Pipeline (On-Demand)

When the Query Manager routes a user question to a Research Lead, the lead spins up **sub-researchers (strategists)** that:

1. **Query the KG** — Execute targeted AQL queries against the knowledge graph to pull relevant events, relationships, and patterns for their specific player.
2. **Extract and analyze** — Sift through retrieved events, identify the most relevant information to the user's question, and surface supporting evidence.
3. **Synthesize a brief** — Compile findings into a structured brief with cited events (titles, dates, source URLs).

The **Research Lead** acts as the quality gate — it reviews the sub-researchers' output, ensures coherence and completeness, and passes the final brief back to the Query Manager.

Multiple sub-researchers can be spun up in parallel for complex queries (e.g., one investigating product launches, another investigating partnerships, for the same player).

### 3.3 Shared Sourcing Layer

Sourcing is general and shared across all teams (not team-specific). A single **Sourcing Agent ("The Finder")** handles:

- **RSS feed polling** — Configurable feed list per player.
- **News API queries** — NewsAPI, Google News RSS, and similar services.
- **Direct web scraping** — BeautifulSoup for static pages and Playwright for JS-rendered content.

Raw content is tagged with the likely relevant hyperscaler(s) via keyword matching on player aliases and passed to the appropriate team(s).

The Sourcing Agent maintains a registry of sources with health checks (last successful fetch, error count, average article yield).

**Note on source selection:** The specific news sites, RSS feeds, and scraping targets should be determined during implementation. The implementer has freedom to choose the most effective and reliable sources for each hyperscaler based on availability, quality, and coverage. The `sources.yaml` config file should be populated with the best sources discovered during development and testing.

### 3.4 Knowledge Graph (ArangoDB)

**Schema:**

Two node collections and one edge collection:

- **`ai_players`** (Document Collection) — One document per hyperscaler.
  - Fields: `_key`, `name`, `aliases`, `description`, `website`, `last_updated`

- **`events`** (Document Collection) — One document per extracted event.
  - Fields: `_key`, `title`, `summary`, `source_url`, `published_date`, `scraped_date`, `event_type`, `significance_score`, `sentiment`, `raw_content_hash` (for dedup), `content_embedding` (for semantic dedup), `analyst_notes`

- **`player_events`** (Edge Collection) — Links `ai_players` → `events`.
  - Fields: `_from`, `_to`, `relationship_type` (e.g., "announced", "involved_in"), `relevance_score`

**Why ArangoDB:** Free tier available, native graph + document model, AQL for flexible querying, and good Python driver support (`python-arango`).

Seed data: five `ai_players` documents are pre-loaded (OpenAI, Anthropic, Google, Meta, Microsoft) with known aliases for matching.

### 3.5 Presentation Layer (Streamlit)

Multi-page Streamlit app with two pages:

#### Page 1 — Agent Chat ("Ask the Agent")

- Chat interface where users type natural language questions.
- The Query Manager decomposes the question, routes to Research Lead(s), and assembles a grounded response.
- Responses include inline citations linking back to source events/URLs.
- Sidebar filters: select hyperscaler(s), date range, event type.
- Example queries: "What has OpenAI announced in the last week?", "Compare Google and Meta's recent AI infrastructure investments", "Any new partnerships involving Anthropic?"

#### Page 2 — Scraping Dashboard

- Per-hyperscaler event feed (most recent events, sortable).
- Last scrape timestamp and next scheduled scrape.
- Source health table: source name, URL, status (healthy/degraded/down), last fetch time, article count.
- Ingestion metrics: events added today/this week, duplicates filtered, errors.
- Manual re-scrape trigger button (in addition to scheduled runs).
- Simple bar/line charts: events over time per hyperscaler, event type distribution.

### 3.6 Deduplication Strategy

Two-layer approach to catch duplicates at different levels:

#### Layer 1 — Content Hash (Fast, Cheap)

During ingestion, a normalized hash of each article's content is computed and checked against `raw_content_hash` in the `events` collection (unique index). Catches exact and near-exact duplicates (e.g., same article syndicated across outlets). This runs before any LLM calls, saving cost on obvious duplicates.

#### Layer 2 — LLM Semantic Check (Thorough)

After passing the hash check, the Information Signal task takes the candidate event and compares it against recent events for the same player (last 7 days, same event type). The LLM determines whether the candidate covers the same underlying event as an existing entry (e.g., two different outlets reporting the same product launch). If flagged as a semantic duplicate, the candidate is either merged (updating the existing event with the new source URL) or discarded.

---

## 4. Technical Specifications

### 4.1 Project Structure

```
market-intelligence-agent/
├── app/
│   ├── streamlit_app.py          # Main entry point
│   ├── pages/
│   │   ├── 1_agent_chat.py       # Page 1: Q&A interface
│   │   └── 2_dashboard.py        # Page 2: Scraping dashboard
│   └── components/               # Shared Streamlit components
├── agents/
│   ├── supervisor.py             # Market Agent Supervisor
│   ├── query_manager.py          # Query Manager (decompose, route, assemble)
│   ├── research_lead.py          # Per-player Research Lead
│   ├── sub_researcher.py         # On-demand KG investigator (strategist)
│   ├── sourcing.py               # Shared Sourcing Agent
│   ├── extraction.py             # Content Extraction task
│   └── signal.py                 # Information Signal task (scoring + dedup)
├── graph/
│   ├── client.py                 # ArangoDB connection + helpers
│   ├── schema.py                 # Collection/index setup
│   └── queries.py                # Reusable AQL query templates
├── scrapers/
│   ├── rss.py                    # RSS feed fetcher
│   ├── news_api.py               # NewsAPI integration
│   └── web.py                    # BeautifulSoup + Playwright scraper
├── llm/
│   ├── provider.py               # Abstract LLM interface
│   ├── claude_provider.py        # Anthropic API implementation
│   ├── openai_provider.py        # OpenAI API implementation
│   └── config.py                 # Provider selection + API keys
├── config/
│   ├── settings.py               # App settings (intervals, thresholds)
│   ├── sources.yaml              # RSS feeds and scraping targets per player
│   └── players.yaml              # Hyperscaler definitions and aliases
├── scheduler/
│   └── cron.py                   # APScheduler-based scheduled scraping
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

### 4.2 Key Dependencies

- **Python 3.11+**
- **Streamlit** — UI framework
- **python-arango** — ArangoDB Python driver
- **APScheduler** — Scheduled scraping jobs
- **BeautifulSoup4** + **requests** — Static web scraping
- **Playwright** — JS-rendered page scraping
- **feedparser** — RSS feed parsing
- **anthropic** / **openai** — LLM provider SDKs
- **pydantic** — Data validation and schema enforcement
- **python-dotenv** — Environment variable management

### 4.3 LLM Configuration

The LLM layer uses an abstract `LLMProvider` interface:

```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...
    @abstractmethod
    def complete_structured(self, system_prompt: str, user_prompt: str, schema: dict) -> dict: ...
```

Provider is selected via `LLM_PROVIDER` env var (`"claude"`, `"openai"`, etc.). Each agent's system prompt is stored separately and can be tuned independently.

### 4.4 ArangoDB Setup

The agent initializes the database on first run:

```python
def initialize_db(db):
    if not db.has_collection("ai_players"):
        db.create_collection("ai_players")
    if not db.has_collection("events"):
        col = db.create_collection("events")
        col.add_hash_index(fields=["raw_content_hash"], unique=True)
        col.add_persistent_index(fields=["published_date"])
        col.add_persistent_index(fields=["event_type"])
    if not db.has_graph("market_intel"):
        db.create_graph("market_intel", edge_definitions=[{
            "edge_collection": "player_events",
            "from_vertex_collections": ["ai_players"],
            "to_vertex_collections": ["events"],
        }])
```

Seed data: five `ai_players` documents are pre-loaded with known aliases for matching.

### 4.5 Scraping Pipeline Flow

```
1. Scheduler triggers scrape run
2. Supervisor creates run metadata (run_id, timestamp)
3. Sourcing Agent fetches from all registered sources
   ├── RSS feeds → feedparser
   ├── News APIs → requests
   └── Web targets → BeautifulSoup / Playwright
4. Raw articles tagged with likely player(s) via keyword matching on aliases
5. Per-player teams process (can run in parallel):
   a. Content Extraction → structured Event object
   b. Information Signal → Layer 1 hash dedup check
   c. Information Signal → Layer 2 LLM semantic dedup (if hash passes)
   d. New events inserted into ArangoDB, edges created to player(s)
6. Run metadata updated (counts, errors, duplicates filtered, duration)
7. Dashboard reflects new data on next page load
```

### 4.6 Query Pipeline Flow

```
1. User submits question via Streamlit chat
2. Query Manager decomposes the request:
   ├── Identifies relevant player(s)
   ├── Determines time frame and event types
   └── Adds contextual framing
3. Query Manager routes to Research Lead(s) in parallel
4. Each Research Lead spins up sub-researchers:
   ├── Sub-researcher A: queries KG for relevant events
   ├── Sub-researcher B: queries KG for related patterns/context
   └── (as many as needed for query complexity)
5. Sub-researchers return findings to Research Lead
6. Research Lead synthesizes a brief (with citations)
7. Query Manager collects briefs from all Research Leads
8. Query Manager assembles final response and returns to user
```

---

## 5. Configuration

### 5.1 Environment Variables

```
# LLM
LLM_PROVIDER=claude                    # claude | openai | custom
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# ArangoDB
ARANGO_HOST=http://localhost:8529
ARANGO_DB=market_intel
ARANGO_USER=root
ARANGO_PASSWORD=

# Scraping
NEWS_API_KEY=...
SCRAPE_INTERVAL_HOURS=4
MAX_ARTICLES_PER_SOURCE=50

# App
STREAMLIT_PORT=8501
LOG_LEVEL=INFO
```

### 5.2 Source Configuration (sources.yaml)

```yaml
# Sources should be populated during implementation.
# The implementer has freedom to choose the best news sites,
# RSS feeds, and scraping targets for each hyperscaler.
# Structure per player:
openai:
  rss: []        # RSS feed URLs
  news_queries:  # Search terms for news APIs
    - "OpenAI"
    - "ChatGPT"
  web_targets: [] # Direct scraping targets with CSS selectors

# Repeat for: anthropic, google, meta, microsoft
```

---

## 6. Agent Prompt Design

Each agent/task has a dedicated system prompt. Key principles:

- **Content Extraction:** "You are a structured data extractor. Given a raw news article, extract: title, one-paragraph summary, event_type (from enum: product_launch, partnership, funding, hiring, policy, research, infrastructure, legal, earnings, other), key entities, and sentiment (positive/neutral/negative). Return JSON only."

- **Information Signal:** "You are an intelligence analyst. Given a candidate event and a list of existing recent events for the same player, assign a significance score (1-10) and determine if this covers the same underlying event as any existing entry. Return JSON with: significance_score, is_duplicate (bool), duplicate_of (_key if applicable), relevance_tags."

- **Sub-Researcher:** "You are a market intelligence researcher. Given a specific question and a set of events from the knowledge graph for [Player], identify the most relevant events, extract key facts, and compile your findings with specific citations (event title, date, source URL). Return structured findings."

- **Research Lead:** "You are a senior research analyst for [Player]. Review the findings from your sub-researchers, ensure completeness and accuracy, resolve any contradictions, and synthesize a concise brief that directly answers the original question. Include all relevant citations."

- **Query Manager:** "You are a market intelligence coordinator. Given a user question, decompose it into: relevant players, time frame, information type, and any necessary context. After receiving briefs from research teams, assemble a coherent final answer with proper citations. If briefs from multiple players are received, synthesize cross-player insights."

---

## 7. Milestones and Implementation Order

### Phase 1 — Foundation ✅

1. Project scaffolding, dependency setup, environment config.
2. ArangoDB schema initialization and seed data (6 players including Emerging).
3. LLM provider abstraction with Claude and OpenAI implementations.
4. Basic Streamlit app shell with multi-page navigation (port 8502).

### Phase 2 — Scraping Pipeline ✅

5. Sourcing Agent: RSS fetcher, Guardian API, NewsAPI placeholder, Anthropic direct web scrape, LLM-native web search.
6. Title-first player tagging strategy to prevent false multi-player matches.
7. Content Extraction with LLM-generated `description` paragraph + `scraped_content` field.
8. Information Signal: two-layer dedup (hash + semantic) + multi-source corroboration model.

### Phase 3 — Intelligence Layer ✅

9. Research Lead and 4 parallel sub-researchers per player (products / partnerships / policy / org).
10. Query Manager with decomposition, parallel routing, conversation history, and response assembly.
11. Supervisor orchestration; APScheduler background scheduler.
12. Knowledge graph query templates (by player, by date range, by event type, keyword search).

### Phase 4 — Presentation ✅

13. Agent Chat page: model selector, URL analysis toggle, reset chat, conversation history, plain-prose briefing tone.
14. Dashboard page: knowledge range metrics, sidebar pipeline controls + source health, per-player event tabs with full descriptions.
15. Historical backfill script (Aug 2025 → present) via Guardian API + LLM month-by-month search.

### Phase 5 — Next (Planned)

- Wire hyperscaler and time frame filters on chat page into `ask()`
- Dashboard event charts (events over time per player, event type distribution)
- Conversation history stored per session (persistence across page reloads)
- Embeddings-based semantic event retrieval (ArangoDB vector search)

---

## 8. Success Metrics (v1)

- Successfully scrapes and ingests events from at least 5 sources per hyperscaler.
- Two-layer dedup correctly filters duplicate events with minimal false positives.
- Knowledge graph contains accurate, deduplicated events with correct player linkage.
- Query pipeline correctly routes to relevant Research Lead(s) and returns cited answers at least 90% of the time.
- Parallel fan-out works for multi-player queries.
- Dashboard accurately reflects scraping status and recent events.
- Full scraping pipeline runs complete within 10 minutes.
- LLM provider can be swapped via env var change with no code modifications.

---

## 9. Open Questions and Future Considerations

- **Rate limiting:** How aggressively should we scrape? Need to respect robots.txt and API quotas.
- **Content quality:** How to handle paywalled or low-quality sources?
- **Graph complexity (v2):** Should we add more node types (Person, Product, Technology) and richer relationships?
- **Vertical expansion (v2):** When verticals are added, the same team architecture extends — each vertical gets its own query patterns.
- **Embeddings (v2):** Add vector similarity search (ArangoDB supports this) for semantic event retrieval alongside graph queries.
- **Multi-user (v2):** Add authentication and personalized alert preferences.
