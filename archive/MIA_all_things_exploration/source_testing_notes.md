# Source Testing Notes
*Date: March 11, 2026*

## What We Did

Tested 88 RSS feeds from the AI/ML/Big Data News section of the README source list.

Two scripts were run:
- `scrape_test.py` — evaluated each source on scrapability, AI relevance %, and recency (past 14 days)
- `latest_news.py` — pulled the latest 2-3 AI-relevant headlines from each working source

Full results saved as JSON:
- `scrape_test_results.json`
- `latest_news.json`

---

## Evaluation Criteria

1. **Scrapability** — can the RSS feed be fetched and parsed? (great / ok / blocked / dead)
2. **AI Relevance %** — what % of items contain AI-specific keywords in title/summary
3. **Recency** — how many articles published in the past 14 days (hot ≥5 / warm 2-4 / stale 1 / dead 0)

---

## Overall Results (88 sources)

| Outcome | Count |
|---|---|
| Active & recent (hot) | 52 |
| Successfully scraped | 79 |
| Blocked (403) | 3 |
| Dead/empty | 25 |

---

## Source Tiers

### Tier 1 — Pure AI Focus, Hot & High Relevance

| Source | 14-day items | AI% | Notes |
|---|---|---|---|
| The Decoder | 10/10 | 100% | Pure AI news, perfectly targeted |
| arXiv cs.CL | 50 | 98% | Research papers, very high volume |
| arXiv cs.LG | 50 | 88% | ML papers, very high volume |
| Simon Willison | 30 | 87% | High-quality AI commentary, posts daily |
| Last Week in AI | 3 | 85% | Weekly digest, great signal density |
| InfoQ AI/ML | 15 | 67% | Practitioner-focused |
| The Guardian AI | 20 | 60% | Reputable, policy-aware |
| NVIDIA Dev Blog | 13 | 60% | Technical, product launches |
| Phys.org AI | 30 | 60% | Research-adjacent, high volume |
| Towards AI | 10 | 60% | Solid AI focus |
| DEV Community | 12 | 58% | Developer perspective |
| Engadget | 50 | 54% | High volume, mixed signal |
| The Register AI | 50 | 54% | High volume, critical/editorial angle |
| Techmeme | 15 | 53% | News aggregator, good for top stories |
| AI Business | 41 | 50% | Industry news, high volume |
| Artificial Intelligence News | 12 | 50% | Focused AI news outlet |
| Wired AI | 10 | 50% | Quality journalism |
| MarkTechPost | 10 | 50% | High volume AI-specific coverage |
| MIT Technology Review | 10 | 50% | Thoughtful, policy + technical |
| Gradient Flow | 7 | 50% | Strategic/business AI angle |

### Tier 2 — Good AI Coverage, Broader Scope

| Source | 14-day items | AI% | Notes |
|---|---|---|---|
| Latent Space | 18 | 45% | Daily AINews digests, practitioner-focused |
| Towards Data Science | 20 | 45% | Technical tutorials + analysis |
| DeepMind Blog | 2 | 46% | First-party Google/DeepMind announcements |
| The New Stack | 26 | 50% | DevOps/infra angle on AI |
| The Algorithmic Bridge | 7 | 40% | Analytical essays, editorial |
| Hugging Face Blog | 7 | 40% | Open-source models, research |
| NY Times Tech | 20 | 40% | Mainstream, policy-heavy |
| Databricks Blog | 10 | 40% | Data/ML platform angle |
| Interconnects | 4 | 35% | LLM research/policy analysis |
| The Sequence | 9 | 35% | Weekly AI roundup newsletter |
| Bloomberg Tech | 29 | 37% | Business/financial angle |
| TechCrunch | 20 | 25% | Broad tech, AI when big news breaks |
| bdTechTalks | 2 | 64% | Technical deep dives |
| SiliconANGLE AI | 2 | 70% | Enterprise AI news |

### Tier 3 — Research / Slow Publishing but High Quality

| Source | Notes |
|---|---|
| Ahead of AI | Monthly-ish, high-quality LLM architecture deep dives |
| Chip Huyen | Infrequent but authoritative on AI engineering |
| BAIR Blog | UC Berkeley research, slow cadence |
| The Gradient | Long-form research essays, slow |
| SemiAnalysis | Deep hardware/infra analysis — likely paywalled recent content |
| Chain of Thought (Every) | Thoughtful AI essays, monthly-ish |
| One Useful Thing | Practical AI use, slow cadence |
| AI Snake Oil | Critical/skeptical take on AI claims |

---

## Dead / Problematic Sources

| Source | Issue |
|---|---|
| OpenAI Blog | Blocked (403) — needs workaround for first-party announcements |
| Microsoft Research | Blocked (403) |
| Datanami | Blocked (403) |
| Google AI Blog | Dead — last post March 2024. Use DeepMind Blog instead |
| WandB Blog | RSS stuck at 2023 |
| Synced Review | Last post August 2025, effectively inactive |
| Unwind AI | Moved platforms in 2024, substack RSS dead |
| Reuters Tech | 404 |
| Computerworld | 404 |
| Analytics India Magazine | RSS parses empty |
| InfoWorld ML | RSS parses empty |
| Unite.AI | RSS parses empty |
| VentureBeat AI | Date parsing broken — shows as dead but content is AI-relevant |

---

## Key Observations

- **Dominant story on test date (March 9)**: Anthropic vs. Pentagon/DOD — covered by 10+ sources from different angles. Good sign that cross-source triangulation will work well for major stories.
- **OpenAI acquisition of Promptfoo** (AI security testing) appeared as breaking news across multiple sources simultaneously.
- **arXiv feeds** are extremely high volume and relevant — useful for research signal but will need filtering to extract industry-relevant content.
- **Latent Space's AINews** format (daily digest) is interesting — already aggregates signal from many sources.
- **VentureBeat** appears to have date metadata issues in its RSS feed — content is good but recency scoring was unreliable.
- **SemiAnalysis** is one of the best sources for hardware/infra analysis but likely paywalls its most recent content.

---

## Next Steps (TBD)

- Decide which sources to include in the new architecture
- Determine how to handle blocked sources (OpenAI Blog workaround)
- Consider how to handle high-volume sources (arXiv, The Register) vs low-volume high-quality (BAIR, Chip Huyen)
- Figure out deduplication strategy — major stories appear across many sources
