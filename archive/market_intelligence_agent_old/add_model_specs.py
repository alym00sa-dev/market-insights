"""
Add major model releases with technical specifications to the database.

This script adds reference data for key 2023-2024 model releases that are missing
from the database, including context window sizes and technical capabilities.
"""

from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.storage import EventDatabase, EventVectorStore
from src.models import (MarketSignalEvent, PillarImpact, CompetitiveEffects, Pillar,
                        DirectionOfChange, RelativeStrength, TemporalContext)

def create_model_events():
    """Create events for major model releases with technical specs."""

    events = [
        # GPT-4 Turbo
        {
            "event_id": "evt_openai_gpt4_turbo_128k_20231106",
            "provider": "OpenAI",
            "source_type": "official_blog",
            "source_url": "https://openai.com/blog/new-models-and-developer-products-announced-at-devday",
            "published_at": datetime(2023, 11, 6),
            "what_changed": "OpenAI released GPT-4 Turbo with 128K context window (4x larger than GPT-4), enabling processing of ~300 pages of text in a single prompt. Includes updated knowledge cutoff (April 2023), JSON mode, function calling improvements, and lower pricing ($0.01/1K input tokens, 3x cheaper than GPT-4).",
            "why_it_matters": "Massive context window expansion enables new use cases like full codebase analysis, long document processing, and extended conversation memory. The 128K token capacity (equivalent to ~96,000 words) positions GPT-4 Turbo as a leader in long-context capabilities, reducing the need for external memory systems and RAG architectures. Lower pricing makes it more accessible for production applications.",
            "scope": "Global availability through OpenAI API. Impacts all developers building applications requiring long-context understanding, including coding assistants, document analysis tools, and conversational AI systems.",
            "pillars": [Pillar.TECHNICAL_CAPABILITIES],
            "pillar_details": {
                Pillar.TECHNICAL_CAPABILITIES: "128K context window represents 4x improvement over GPT-4's 32K, enabling full codebase and document processing. JSON mode and improved function calling enhance reliability for production applications."
            }
        },

        # Gemini 1.5 Pro
        {
            "event_id": "evt_google_gemini_15_pro_1m_20240215",
            "provider": "Google",
            "source_type": "official_blog",
            "source_url": "https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/",
            "published_at": datetime(2024, 2, 15),
            "what_changed": "Google launched Gemini 1.5 Pro with breakthrough 1 million token context window, capable of processing 1 hour of video, 11 hours of audio, 30,000 lines of code, or 700,000 words in a single prompt. Uses Mixture-of-Experts (MoE) architecture for efficient scaling. Matches Gemini 1.0 Ultra quality at lower computational cost.",
            "why_it_matters": "The 1M token context window is a massive leap - 8x larger than GPT-4 Turbo's 128K and 5x larger than Claude 2.1's 200K. This enables entirely new categories of applications: analyzing full movies, processing entire codebases, reasoning over massive documents. Dramatically reduces need for chunking, summarization, and external memory systems. Represents significant competitive pressure on OpenAI and Anthropic to expand their context capabilities.",
            "scope": "Initially available to developers and enterprises through Vertex AI and AI Studio. Global rollout planned. Impacts video analysis, document processing, code understanding, and any application requiring massive context understanding.",
            "pillars": [Pillar.TECHNICAL_CAPABILITIES, Pillar.MARKET_SHAPING],
            "pillar_details": {
                Pillar.TECHNICAL_CAPABILITIES: "1 million token context window - industry-leading by significant margin. MoE architecture enables efficient scaling. Multimodal processing across text, code, audio, and video.",
                Pillar.MARKET_SHAPING: "Sets new standard for context length, forcing competitors to respond. Opens entirely new market segments for long-context applications."
            }
        },

        # Gemini 1.5 Flash
        {
            "event_id": "evt_google_gemini_15_flash_20240515",
            "provider": "Google",
            "source_type": "official_blog",
            "source_url": "https://developers.googleblog.com/en/gemini-15-flash-building-the-future/",
            "published_at": datetime(2024, 5, 15),
            "what_changed": "Google released Gemini 1.5 Flash, a faster and more efficient model with 1 million token context window at lower cost than 1.5 Pro. Optimized for high-frequency tasks like chat, code generation, and API calls. Delivers similar quality to 1.5 Pro for most tasks while being significantly faster and cheaper.",
            "why_it_matters": "Democratizes access to massive context windows by offering 1M token capacity at lower cost and latency. Enables production deployment of long-context applications at scale. Strategic move to capture high-volume API traffic from developers who need long context but can't justify 1.5 Pro pricing. Puts competitive pressure on OpenAI and Anthropic's pricing and speed.",
            "scope": "Available globally through Vertex AI and AI Studio. Targets high-frequency production workloads including chatbots, coding assistants, content generation, and data processing pipelines.",
            "pillars": [Pillar.TECHNICAL_CAPABILITIES, Pillar.DATA_PIPELINES],
            "pillar_details": {
                Pillar.TECHNICAL_CAPABILITIES: "1 million token context window with faster inference and lower latency than 1.5 Pro. Optimized for throughput without sacrificing long-context capabilities.",
                Pillar.DATA_PIPELINES: "Lower cost and higher speed enable production-scale deployment of long-context applications, changing economics of data processing pipelines."
            }
        },

        # Claude 3 Opus
        {
            "event_id": "evt_anthropic_claude3_opus_200k_20240304",
            "provider": "Anthropic",
            "source_type": "official_blog",
            "source_url": "https://www.anthropic.com/news/claude-3-family",
            "published_at": datetime(2024, 3, 4),
            "what_changed": "Anthropic launched Claude 3 Opus with 200K token context window (double Claude 2.1's 100K), achieving state-of-the-art performance across reasoning, math, coding, and graduate-level knowledge benchmarks. Outperforms GPT-4 and Gemini 1.0 Ultra on most evaluations. Improved instruction following, reduced hallucinations, and better long-context recall.",
            "why_it_matters": "200K context window (equivalent to ~150,000 words or a 500-page novel) enables comprehensive document analysis and extended conversations without memory loss. Superior performance on reasoning benchmarks positions Claude 3 Opus as the quality leader for complex tasks. Near-perfect recall across the full context window solves the 'lost in the middle' problem that plagued earlier long-context models. Directly competes with GPT-4 Turbo and Gemini for enterprise customers prioritizing quality and safety.",
            "scope": "Available globally through Claude API and AWS Bedrock. Targets enterprise customers, researchers, and developers building applications requiring high-quality reasoning over large contexts.",
            "pillars": [Pillar.TECHNICAL_CAPABILITIES, Pillar.ALIGNMENT],
            "pillar_details": {
                Pillar.TECHNICAL_CAPABILITIES: "200K token context window with near-perfect recall. State-of-the-art performance on reasoning, math, and coding benchmarks. Improved instruction following and reduced hallucinations.",
                Pillar.ALIGNMENT: "Constitutional AI training emphasizes safety and alignment. Reduced harmful outputs while maintaining capabilities. Transparent about limitations and uncertainties."
            }
        },

        # Claude 3 Sonnet
        {
            "event_id": "evt_anthropic_claude3_sonnet_200k_20240304",
            "provider": "Anthropic",
            "source_type": "official_blog",
            "source_url": "https://www.anthropic.com/news/claude-3-family",
            "published_at": datetime(2024, 3, 4),
            "what_changed": "Anthropic launched Claude 3 Sonnet as the balanced model in the Claude 3 family, offering 200K token context window at 2x the speed and 1/5 the cost of Claude 3 Opus. Outperforms GPT-4 and previous Claude models on most benchmarks while being optimized for high-throughput production workloads.",
            "why_it_matters": "Brings flagship-level context capacity (200K tokens) to a mid-tier pricing point, making long-context capabilities accessible for production applications. 2x speed advantage over Opus enables real-time applications. Competitive pricing threatens GPT-4 Turbo and Gemini 1.5 Flash in the high-volume enterprise segment. Sweet spot of quality, speed, and cost positions it as the default choice for most applications.",
            "scope": "Available globally through Claude API, AWS Bedrock, and Google Cloud Vertex AI. Targets enterprise production workloads, customer support chatbots, content generation, and code assistants.",
            "pillars": [Pillar.TECHNICAL_CAPABILITIES, Pillar.DATA_PIPELINES],
            "pillar_details": {
                Pillar.TECHNICAL_CAPABILITIES: "200K context window with 2x speed and lower cost than Opus. Maintains strong performance on reasoning and coding while optimized for throughput.",
                Pillar.DATA_PIPELINES: "Lower cost and higher speed enable large-scale deployment. Efficient processing of high-volume document analysis and conversation pipelines."
            }
        },

        # GPT-4o
        {
            "event_id": "evt_openai_gpt4o_128k_20240513",
            "provider": "OpenAI",
            "source_type": "official_blog",
            "source_url": "https://openai.com/index/hello-gpt-4o/",
            "published_at": datetime(2024, 5, 13),
            "what_changed": "OpenAI released GPT-4o ('o' for 'omni'), a natively multimodal model that processes text, audio, and vision with 128K token context window. 2x faster than GPT-4 Turbo with 50% lower API pricing. Real-time voice conversation with ~320ms response time. Improved performance on non-English languages and vision understanding.",
            "why_it_matters": "First truly multimodal model from OpenAI that processes audio and vision natively rather than through separate models. 2x speed improvement and 50% cost reduction make GPT-4-level capabilities accessible for production. 128K context window maintained across all modalities. Real-time voice with natural interruptions enables new conversational AI applications. Competitive response to Google's multimodal Gemini models.",
            "scope": "Available globally through OpenAI API and ChatGPT Plus/Team/Enterprise. Free tier access in ChatGPT. Impacts voice assistants, multimodal applications, real-time translation, accessibility tools, and conversational AI.",
            "pillars": [Pillar.TECHNICAL_CAPABILITIES, Pillar.MARKET_SHAPING],
            "pillar_details": {
                Pillar.TECHNICAL_CAPABILITIES: "128K context window across text, audio, and vision. Native multimodal processing with 2x speed improvement. Real-time voice with ~320ms latency.",
                Pillar.MARKET_SHAPING: "50% cost reduction expands market accessibility. Real-time voice capabilities enable new application categories. Competitive response to Gemini's multimodal push."
            }
        }
    ]

    return events


def ingest_model_events():
    """Ingest model events into database and vector store."""

    # Initialize database and vector store
    db_path = Path(__file__).parent / "data" / "events.db"
    vector_path = Path(__file__).parent / "data" / "vector_store"

    db = EventDatabase(str(db_path))
    vector_store = EventVectorStore(str(vector_path))

    events_data = create_model_events()

    print(f"Adding {len(events_data)} model release events...\n")

    for event_data in events_data:
        # Create PillarImpact objects
        pillar_impacts = []
        for pillar in event_data["pillars"]:
            evidence = event_data["pillar_details"].get(pillar, "")
            pillar_impacts.append(
                PillarImpact(
                    pillar_name=pillar,
                    direction_of_change=DirectionOfChange.ADVANCE,  # Major model releases advance capabilities
                    relative_strength_signal=RelativeStrength.STRONG,  # Industry-leading releases
                    evidence=evidence,
                    impact_score=85  # High impact for major model releases
                )
            )

        # Create CompetitiveEffects
        competitive_effects = CompetitiveEffects(
            advantages_created=[
                "Expanded context window capabilities",
                "Improved performance on benchmarks",
                "Enhanced multimodal processing" if "multimodal" in event_data["what_changed"].lower() else "Better long-context handling"
            ],
            disadvantages_created=[],
            market_gaps_filled=["Long-context understanding", "Enterprise-scale AI applications"],
            lock_in_or_openness_shift="Increased lock-in potential - larger context windows reduce need for external memory systems and open standards, as models can store more information internally."
        )

        # Create TemporalContext
        temporal_context = TemporalContext(
            preceded_by_events=[],
            likely_to_trigger_events=["Competitors will likely respond with model releases or capability announcements"],
            time_horizon="immediate"
        )

        # Create MarketSignalEvent
        event = MarketSignalEvent(
            event_id=event_data["event_id"],
            provider=event_data["provider"],
            source_type=event_data["source_type"],
            source_url=event_data["source_url"],
            published_at=event_data["published_at"],
            retrieved_at=datetime.now(),
            what_changed=event_data["what_changed"],
            why_it_matters=event_data["why_it_matters"],
            scope=event_data["scope"],
            pillars_impacted=pillar_impacts,
            competitive_effects=competitive_effects,
            temporal_context=temporal_context,
            alignment_implications="Major model release with expanded capabilities. Standard safety testing and responsible deployment practices assumed based on provider track record.",
            extraction_confidence=0.95  # High confidence for reference data
        )

        # Add to database
        try:
            db.create_event(event)
            print(f"✅ Added: {event.provider} - {event.event_id}")
        except Exception as e:
            print(f"⚠️  Skipped {event.event_id}: {e}")

        # Add to vector store
        try:
            vector_store.add_event(event)
            print(f"   → Indexed in vector store")
        except Exception as e:
            print(f"   ⚠️  Vector indexing failed: {e}")

        print()

    print("✅ Model spec ingestion complete!")
    print(f"\nAdded events:")
    print(f"  - GPT-4 Turbo (Nov 2023): 128K tokens")
    print(f"  - GPT-4o (May 2024): 128K tokens, multimodal")
    print(f"  - Claude 3 Opus (Mar 2024): 200K tokens")
    print(f"  - Claude 3 Sonnet (Mar 2024): 200K tokens")
    print(f"  - Gemini 1.5 Pro (Feb 2024): 1M tokens 🏆")
    print(f"  - Gemini 1.5 Flash (May 2024): 1M tokens")


if __name__ == "__main__":
    ingest_model_events()
