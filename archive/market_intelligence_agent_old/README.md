# Frontier Market Intelligence Agent

Multi-agent system for tracking and analyzing frontier AI competition across hyperscalers using the I³ Index framework.

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure API keys:**
Edit `.env` and add your API keys:
- `ANTHROPIC_API_KEY` - For Claude Sonnet 4.5
- `OPENAI_API_KEY` - For GPT-4o-mini and embeddings
- `TAVILY_API_KEY` - For web search

3. **Initialize database:**
```bash
python src/main.py init
```

4. **Seed with demo data:**
```bash
python src/main.py seed
```

5. **Start chatting:**
```bash
python src/main.py chat
```

## Architecture

- **Source Scout** - Discovers sources to monitor
- **Content Harvester** - Fetches and filters content
- **Signal Extractor** - Converts content to structured events
- **Competitive Reasoning** - Analyzes competitive dynamics
- **Analyst Copilot** - Natural language interface

## I³ Index Pillars

1. Data Pipelines & Standards
2. Technical Capabilities & Platforms
3. Education & Advisory Influence
4. Market Shaping & Partnerships
5. Alignment / Governance
