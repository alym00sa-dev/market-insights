Product Requirements Document (PRD – Revised MVP)

Product Name (Working): Frontier Market Intelligence Agent
Audience: Claude Code / Engineering Agent
Author: Aly Moosa
Status: Revised MVP Definition
Primary Goal: Enable structured, longitudinal, and comparative analysis of frontier AI market shifts across hyperscalers using agentic research and persistent memory.

1. Problem Statement (Revised)

Frontier AI competition is defined by fast, uneven, and compounding changes across:

infrastructure,

product capabilities,

governance,

partnerships,

and ecosystem influence.

Currently, it is extremely difficult to:

understand how a single event changes the competitive landscape

compare who is leading vs reacting

trace causal timelines across providers

explain why leadership shifts occur using a consistent framework

This product exists to make those dynamics explicit, explainable, and queryable.

2. MVP Vision (What This System Must Do)

The MVP is a Market Intelligence Reasoning System that can:

Ingest product, infrastructure, and governance signals from frontier AI providers

Store them as structured historical events

Analyze how events change competitive dynamics

Compare hyperscalers over time and against each other

Explain:

who is leading

who is following

who is diverging

and why

This is not a dashboard.
This is not a visualization tool.
This is a reasoning and memory system.

3. Analytical Framework (Still Non-Negotiable)

All analysis MUST be grounded in the I³ Index (5 Pillars):

Data Pipelines & Standards

Technical Capabilities & Platforms

Education & Advisory Influence

Market Shaping & Partnerships

Alignment / Governance

The system must reason within this framework, not around it.

4. Core Concept: Competitive Market Event

The atomic unit remains a Market Signal Event, but the MVP elevates its purpose:

A Market Signal Event is evidence that changes the competitive balance along one or more I³ pillars.

Events are not just “what happened”; they are inputs to market structure change.

5. Event Schema (MVP-Revised)

Each event MUST include:

5.1 Event Metadata

event_id

provider

source_type

source_url

published_at

retrieved_at

5.2 Event Description

what_changed (concise, factual)

why_it_matters (mechanism, not opinion)

scope (regions, users, APIs, partners affected)

5.3 I³ Classification

pillars_impacted[]

pillar_name

direction_of_change (advance | neutral | constrain)

relative_strength_signal (strong | moderate | weak)

evidence

5.4 Competitive Impact (New – MVP Critical)

competitive_effects:

advantages_created[]

advantages_eroded[]

new_barriers[]

lock_in_or_openness_shift

5.5 Temporal Context (New)

preceded_by_events[]

likely_to_trigger_events[]

time_horizon (immediate | medium | long)

5.6 Governance & Risk Lens

alignment_implications

regulatory_signal (none | emerging | material)

6. Market Reasoning Capabilities (MVP Requirements)

The system MUST be able to answer the following classes of questions:

6.1 Event-Driven Market Shift Analysis

“When X released feature Y, how did that change the market?”

Identify which pillars shifted

Identify who gained advantage

Identify who was forced to respond

6.2 Cross-Hyperscaler Comparison

“How do OpenAI and Anthropic differ on memory portability today?”

Compare event histories

Contrast pillar-specific strategies

Explain divergence using evidence

6.3 Leadership & Momentum Analysis

“Who is leading on infrastructure openness over the last 6 months?”

Rank providers per pillar

Justify leadership with event sequences

Explain momentum vs stagnation

6.4 Timeline & Causal Chain Explanation

“How did governance expectations around agents evolve this year?”

Construct a narrative timeline

Show cause → response → escalation

Attribute leadership vs reaction

7. Agent Architecture (Revised Emphasis)
7.1 Source Scout

Unchanged — but queries must prioritize signals with competitive implications, not feature trivia.

7.2 Content Harvester

Unchanged — but must track revision history and “quiet updates.”

7.3 Signal Extractor (Expanded Responsibility)

Now required to:

Detect market relevance

Identify which competitors are implicitly affected

Flag “forcing functions” (events others must respond to)

7.4 Competitive Reasoning Agent (NEW, MVP-Critical)

This is the core of the MVP.

Responsibilities:

Compare events across providers

Determine leadership vs followership

Detect convergence or divergence

Maintain rolling pillar-specific standings

Outputs:

Comparative summaries

Leadership explanations

Pillar momentum assessments

7.5 Analyst Copilot (Chat Interface)

Now optimized for:

Market comparison

Timeline explanation

Competitive reasoning

Not optimized for:

UI generation

Visualization logic

Generic Q&A

8. Persistent Memory Requirements

The system must maintain:

Event History (append-only)

Pillar-Specific Timelines

Provider Profiles

Strengths by pillar

Known strategies

Historical behavior patterns

Memory must be actively used in reasoning, not passive storage.

9. MVP Success Criteria (Revised)

The MVP is successful if:

The system ingests ≥ 30 high-signal events across ≥ 3 hyperscalers

Each event is mapped to:

at least one pillar

at least one competitive effect

The system can correctly answer:

“What changed?”

“Who benefited?”

“Who was pressured to respond?”

“Who is leading on X and why?”

The chatbot can generate:

a coherent market timeline

a justified leadership comparison

Re-asking similar questions improves answers due to memory

10. Explicit Non-Goals (MVP)

Dashboards or visualizations

Predictive or probabilistic forecasting

Numeric scoring systems

Automated recommendations

External benchmarking datasets

11. Design Principle (Revised)

The product is not the data.
The product is the explanation of how competition evolves over time.

Claude Code should optimize for:

clarity of causal reasoning

traceable evidence

longitudinal coherence

comparative insightl