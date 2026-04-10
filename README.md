# Market Intelligence Agent

A continuous intelligence system that tracks AI market activity across major players — OpenAI, Anthropic, Google, Meta, Microsoft, and emerging competitors. It ingests news from multiple sources, extracts structured events using an LLM, stores them in a knowledge graph, and answers natural language questions grounded in that data.

**Live app, BUT NO LONGER FUNCTIONING - see ArangoDB note below:** https://market-intelligence-gf.streamlit.app/

---

> **Note: The ArangoDB instance backing this app has expired and the data is no longer available.**
> The live app is currently non-functional. To run this locally or redeploy, you will need to provision a new ArangoDB instance and repopulate the knowledge graph from scratch. See the [Starting fresh](#starting-fresh) section below.

---

## Starting fresh

The original ArangoDB instance has expired and all data is gone. There is no compatible data export — an older SQLite database exists in the archive but uses a different schema from a prior version of the system and cannot be imported directly.

To get the app running again:

**1. Provision a new ArangoDB instance**

Create a free ArangoDB AuraDB instance at [arangodb.com](https://arangodb.com). Copy the host URL, username, and password into your `.env` file.

**2. Initialize the schema**

```bash
PYTHONPATH=. python -c "from graph.schema import init_schema; init_schema()"
```

**3. Verify the connection**

```bash
PYTHONPATH=. python market_intelligence_agent/scripts/test_db_connection.py
```

**4. Run the historical backfill** (optional, rebuilds data from Aug 2025 onward)

```bash
PYTHONPATH=. python market_intelligence_agent/scripts/backfill_historical.py
```

This pulls historical data from the Guardian API and LLM web search. Requires a valid `GUARDIAN_API_KEY` and LLM API key in your `.env` and will consume API quota.

To go back further than Aug 2025, change the `FROM_DATE` variable at the top of the script to your desired start date (`YYYY-MM-DD`). To also include NYT Archive data, add the `--nyt-archive` flag — note this is rate-limited and adds ~12 seconds per month of history.

**5. Start the live pipeline**

```bash
PYTHONPATH=. python market_intelligence_agent/pipeline_worker.py
```

---

## How it works

Articles are collected from RSS feeds, news APIs (NewsAPI, Guardian, NYT), and LLM-native web search. Each article is processed by an extraction agent that produces structured events with type, sentiment, significance, and key entities. A deduplication layer prevents redundant data using content hashing and semantic comparison.

All events are stored as nodes in a knowledge graph hosted on ArangoDB. Players are nodes, events are nodes, and edges connect players to the events attributed to them. This structure allows for fast, targeted retrieval by player, date range, or event type.

When a user asks a question, a query manager decomposes it, identifies relevant players, and spins up parallel Research Lead agents — one per player. Each Research Lead retrieves events from the graph and synthesizes a cited brief. The final response is assembled from those briefs.

The pipeline runs on a configurable schedule (default: every 4 hours) within the Streamlit app process via APScheduler.

---

## Architecture

```
Scrapers (7 sources)
    RSS feeds, NewsAPI, Guardian, NYT, Artificial Analysis, web scraper, LLM web search
        |
        v
Sourcing Agent
    Tags articles to players via alias matching
        |
        v
Extraction Agent
    LLM-structured event extraction (type, sentiment, significance, entities)
        |
        v
Signal Agent
    Deduplication (hash + semantic) and persistence to ArangoDB
        |
        v
ArangoDB Knowledge Graph
    Collections: ai_players, events, player_events (edge)
        |
        v
Query Manager
    Decomposes user question, routes to Research Leads per player
        |
        v
Research Leads + Sub-Researchers (parallel)
    Retrieve events from graph, synthesize cited briefs
        |
        v
Streamlit UI
    Chat interface (agent Q&A) + Dashboard (pipeline status, source health)
```

**LLM layer:** Claude and OpenAI are both supported and switchable at runtime. Default is Claude Sonnet.

**Knowledge graph:** Hosted on ArangoDB AuraDB. The graph stores players, events, and edges connecting them. AQL queries power all retrieval.

---

## Folder structure

```
market_intelligence_agent/
    agents/         Core pipeline agents (sourcing, extraction, signal, query, research)
    app/            Streamlit UI (chat page, dashboard page)
    config/         Settings, players.yaml, sources.yaml
    graph/          ArangoDB client, schema, and query templates
    llm/            LLM abstraction layer (Claude and OpenAI providers)
    scrapers/       Data source connectors (RSS, APIs, web, LLM search)
    scheduler/      Background scheduler (APScheduler)
    scripts/        Utility scripts (backfill_historical.py, test_db_connection.py)
    pipeline_worker.py   Background worker entry point
    requirements.txt
    .env.example
```

---

## Running locally

**1. Install dependencies**

```bash
pip install -r market_intelligence_agent/requirements.txt
```

**2. Set up environment**

Copy `.env.example` to `.env` and fill in your values (see Environment section below).

**3. Run the pipeline worker** (scrapes and ingests data)

```bash
PYTHONPATH=. python market_intelligence_agent/pipeline_worker.py
```

**4. Run the Streamlit app**

```bash
streamlit run market_intelligence_agent/app/streamlit_app.py
```

---

## Environment variables

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `claude` or `openai` |
| `ANTHROPIC_API_KEY` | Required if using Claude |
| `OPENAI_API_KEY` | Required if using OpenAI |
| `CLAUDE_MODEL` | Claude model ID (e.g. `claude-sonnet-4-6`) |
| `OPENAI_MODEL` | OpenAI model ID (e.g. `gpt-4o`) |
| `ARANGO_HOST` | ArangoDB host URL |
| `ARANGO_DB` | Database name |
| `ARANGO_USER` | Database user |
| `ARANGO_PASSWORD` | Database password |
| `NEWS_API_KEY` | NewsAPI.org key |
| `GUARDIAN_API_KEY` | The Guardian API key |
| `NYT_API_KEY` | New York Times API key |
| `ARTIFICIAL_ANALYSIS_API_KEY` | ArtificialAnalysis.ai key |
| `SCRAPE_INTERVAL_HOURS` | Pipeline run frequency (default: 4) |

---

## market_insights_viz

A Next.js visualization app is available in the `market_insights_viz/` directory. It was built to visualize knowledge graph data but may not be fully functional in its current state and could require code changes to connect to a live data source.
