"""
Source Scout Agent

Discovers and prioritizes sources for market signal tracking.

Design:
- Two modes: curated (from config) vs automated (LLM + web search)
- Ranks sources by reliability score (successful_events / total_fetches)
- Uses GPT-4o-mini (simple task - pattern matching)
- Tavily for web search (when use_web_search=True)

Responsibilities:
1. Return curated sources from config
2. Discover new sources using LLM + web search (automated mode)
3. Rank sources by historical reliability
"""

import os
from typing import List, Dict, Any, Optional, Literal
from dataclasses import dataclass
from dotenv import load_dotenv

from ..llm import LLMProvider, SOURCE_SCOUT_SYSTEM_PROMPT, SOURCE_SCOUT_USER_PROMPT
from ..storage import EventDatabase
from ..models import Pillar

load_dotenv()


@dataclass
class SourceCandidate:
    """
    A potential source for market signals.

    Why dataclass instead of Pydantic?
    - Simple data structure (no complex validation needed)
    - Lightweight (not stored in DB)
    - Easy to work with
    """
    url: str
    provider: str
    source_type: str  # official_blog, github, documentation, rss_feed, etc.
    priority: Literal["high", "medium", "low"]
    confidence: float  # 0-1, how confident we are this is a good source
    reasoning: str  # Why is this a good source?
    reliability_score: Optional[float] = None  # From DB if source exists


class SourceScout:
    """
    Discovers and ranks sources for competitive intelligence.

    Why separate from Content Harvester?
    - Different concerns: discovery vs fetching
    - Can run independently (batch source discovery)
    - Simpler to test/debug
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        database: EventDatabase,
        config: Dict[str, Any]
    ):
        """
        Initialize Source Scout.

        Args:
            llm_provider: LLM provider for source discovery
            database: Database to query source reliability
            config: Configuration dict (from config.yaml)
        """
        self.llm = llm_provider
        self.db = database
        self.config = config

        # Extract default sources from config
        self.default_sources = config['agents']['source_scout'].get('default_sources', [])

        # Tavily client (for web search)
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.tavily_client = None

        # Initialize Tavily if API key present
        if self.tavily_api_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
            except ImportError:
                print("Warning: tavily-python not installed. Web search disabled.")

    def discover_sources(
        self,
        provider: str,
        pillar: Optional[Pillar] = None,
        mode: Literal["curated", "automated"] = "curated",
        use_web_search: bool = False,
        limit: int = 10
    ) -> List[SourceCandidate]:
        """
        Discover sources for a provider and pillar.

        This is the main entry point for source discovery.

        Args:
            provider: Provider name (e.g., "OpenAI", "Anthropic")
            pillar: Optional pillar to focus on
            mode: "curated" (use config) or "automated" (LLM discovery)
            use_web_search: Whether to use Tavily for discovery
            limit: Max number of sources to return

        Returns:
            List of SourceCandidate objects, ranked by priority and reliability

        Example:
            sources = scout.discover_sources(
                provider="OpenAI",
                pillar=Pillar.TECHNICAL_CAPABILITIES,
                mode="automated",
                use_web_search=True
            )
        """
        if mode == "curated":
            # Return curated sources from config
            sources = self._get_curated_sources(provider, pillar)
        else:
            # Automated discovery
            sources = self._discover_automated(provider, pillar, use_web_search)

        # Enrich with reliability scores from database
        sources = self._enrich_with_reliability(sources)

        # Rank: reliability_score × confidence × priority_weight
        sources = self._rank_sources(sources)

        return sources[:limit]

    def _get_curated_sources(
        self,
        provider: str,
        pillar: Optional[Pillar] = None
    ) -> List[SourceCandidate]:
        """
        Get curated sources from config.

        Why start with curated?
        - Known high-quality sources
        - Faster than LLM discovery
        - Good for MVP validation

        Returns:
            List of SourceCandidate objects from config
        """
        sources = []

        # Add API sources (always high-value for technical capabilities)
        sources.extend(self._get_api_sources(pillar))

        # Map provider to known sources
        # For MVP, we have a simple default_sources list
        # Can expand to provider-specific mapping later
        provider_lower = provider.lower()

        for url in self.default_sources:
            # Simple heuristic: check if provider name in URL
            if provider_lower in url.lower():
                # Infer source type from URL
                source_type = self._infer_source_type(url)

                sources.append(SourceCandidate(
                    url=url,
                    provider=provider,
                    source_type=source_type,
                    priority="high",  # Config sources are high priority
                    confidence=1.0,  # We curated these, so 100% confident
                    reasoning=f"Curated source from configuration for {provider}"
                ))

        # If no sources found in config, use fallback heuristics
        if not sources:
            sources = self._generate_fallback_sources(provider)

        return sources

    def _get_api_sources(self, pillar: Optional[Pillar] = None) -> List[SourceCandidate]:
        """
        Get API sources from config.

        API sources provide structured data (not web scraping).
        Example: Artificial Analysis API provides model benchmarks, pricing, rankings.

        Why include APIs?
        - Structured data (easier to extract signals)
        - Authoritative (benchmarks, not just announcements)
        - Timely (updated frequently)
        - Cross-provider (can detect competitive shifts)

        Args:
            pillar: If specified, filter to APIs relevant to this pillar

        Returns:
            List of SourceCandidate objects for API endpoints
        """
        api_sources = []

        # Check if Artificial Analysis is configured
        api_config = self.config.get('external_apis', {}).get('artificial_analysis', {})

        if api_config.get('enabled'):
            base_url = api_config['base_url']
            endpoints = api_config['endpoints']

            # Add LLM benchmarks endpoint (relevant to TECHNICAL_CAPABILITIES)
            if pillar is None or pillar == Pillar.TECHNICAL_CAPABILITIES:
                api_sources.append(SourceCandidate(
                    url=f"{base_url}{endpoints['llms']}",
                    provider="all",  # Cross-provider data
                    source_type="api_llm_benchmarks",
                    priority="high",
                    confidence=1.0,
                    reasoning="Artificial Analysis API: LLM model benchmarks, pricing, and performance metrics"
                ))

            # Add image generation rankings (if needed)
            if pillar is None or pillar == Pillar.TECHNICAL_CAPABILITIES:
                api_sources.append(SourceCandidate(
                    url=f"{base_url}{endpoints['text_to_image']}",
                    provider="all",
                    source_type="api_image_gen_rankings",
                    priority="medium",
                    confidence=1.0,
                    reasoning="Artificial Analysis API: Text-to-image model ELO rankings"
                ))

        return api_sources

    def _discover_automated(
        self,
        provider: str,
        pillar: Optional[Pillar] = None,
        use_web_search: bool = False
    ) -> List[SourceCandidate]:
        """
        Automated source discovery using LLM (+ optionally web search).

        Flow:
        1. LLM generates search queries
        2. Execute Tavily search (if enabled)
        3. LLM evaluates each URL for quality
        4. Return ranked candidates

        Why LLM + web search?
        - Finds sources we didn't curate
        - Adapts to new providers/platforms
        - Can focus on specific pillars

        Args:
            provider: Provider name
            pillar: Optional pillar focus
            use_web_search: Whether to use Tavily

        Returns:
            List of discovered SourceCandidate objects
        """
        discovered_sources = []

        # Step 1: Generate search queries using LLM
        search_queries = self._generate_search_queries(provider, pillar)

        # Step 2: Execute web search (if enabled)
        if use_web_search and self.tavily_client:
            search_results = []
            for query in search_queries[:3]:  # Limit to top 3 queries
                try:
                    results = self.tavily_client.search(
                        query=query,
                        max_results=5
                    )
                    search_results.extend(results.get('results', []))
                except Exception as e:
                    print(f"Tavily search failed for '{query}': {e}")

            # Step 3: LLM evaluates each URL
            for result in search_results:
                url = result.get('url', '')
                title = result.get('title', '')
                snippet = result.get('content', '')

                # Ask LLM to evaluate this source
                evaluation = self._evaluate_source_with_llm(
                    url, title, snippet, provider, pillar
                )

                if evaluation['is_relevant']:
                    discovered_sources.append(SourceCandidate(
                        url=url,
                        provider=provider,
                        source_type=evaluation['source_type'],
                        priority=evaluation['priority'],
                        confidence=evaluation['confidence'],
                        reasoning=evaluation['reasoning']
                    ))

        else:
            # No web search - just use LLM to suggest sources
            # (Less effective but doesn't require Tavily API key)
            discovered_sources = self._discover_with_llm_only(provider, pillar)

        return discovered_sources

    def _generate_search_queries(
        self,
        provider: str,
        pillar: Optional[Pillar] = None
    ) -> List[str]:
        """
        Generate search queries for finding sources.

        Uses GPT-4o-mini to generate targeted queries.

        Returns:
            List of search query strings
        """
        pillar_context = f" focusing on {pillar.value}" if pillar else ""

        prompt = f"""Generate 3-5 search queries to find high-quality sources about {provider}{pillar_context}.

Focus on official sources (blogs, documentation, GitHub, policy pages).

Return as JSON list: ["query 1", "query 2", ...]

Examples:
- "{provider} official blog announcements"
- "{provider} API documentation updates"
- "{provider} GitHub releases"
"""

        messages = [
            {"role": "user", "content": prompt}
        ]

        response = self.llm.generate(
            messages=messages,
            task_complexity="simple",  # GPT-4o-mini
            temperature=0.5
        )

        # Parse JSON from response
        import json
        try:
            queries = json.loads(response['content'])
            return queries
        except:
            # Fallback: basic queries
            return [
                f"{provider} official blog",
                f"{provider} API documentation",
                f"{provider} GitHub releases"
            ]

    def _evaluate_source_with_llm(
        self,
        url: str,
        title: str,
        snippet: str,
        provider: str,
        pillar: Optional[Pillar]
    ) -> Dict[str, Any]:
        """
        Ask LLM to evaluate if a URL is a good source.

        Returns:
            Dict with: is_relevant, source_type, priority, confidence, reasoning
        """
        prompt = f"""Evaluate if this is a high-quality source for competitive intelligence about {provider}:

URL: {url}
Title: {title}
Snippet: {snippet}

Is this:
1. Official (from the company itself)?
2. Technical (API docs, GitHub, research)?
3. Likely to contain competitive signals (not just marketing)?

Return JSON:
{{
  "is_relevant": true/false,
  "source_type": "official_blog" | "github" | "documentation" | "other",
  "priority": "high" | "medium" | "low",
  "confidence": 0.0-1.0,
  "reasoning": "why this is or isn't a good source"
}}
"""

        messages = [{"role": "user", "content": prompt}]

        response = self.llm.generate(
            messages=messages,
            task_complexity="simple",
            temperature=0.3
        )

        # Parse JSON
        import json
        try:
            return json.loads(response['content'])
        except:
            # Fallback: conservative evaluation
            return {
                "is_relevant": False,
                "source_type": "other",
                "priority": "low",
                "confidence": 0.3,
                "reasoning": "Could not evaluate"
            }

    def _discover_with_llm_only(
        self,
        provider: str,
        pillar: Optional[Pillar] = None
    ) -> List[SourceCandidate]:
        """
        Discover sources using LLM only (no web search).

        Less effective than web search but works without Tavily.
        LLM suggests URLs based on its training data.
        """
        pillar_text = f" for pillar: {pillar.value}" if pillar else ""

        prompt = SOURCE_SCOUT_USER_PROMPT.format(
            provider=provider,
            pillar=pillar_text
        )

        messages = [
            {"role": "system", "content": SOURCE_SCOUT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = self.llm.generate(
            messages=messages,
            task_complexity="simple",
            temperature=0.6
        )

        # Parse JSON response
        import json
        try:
            sources_data = json.loads(response['content'])

            return [
                SourceCandidate(
                    url=s['url'],
                    provider=provider,
                    source_type=s['source_type'],
                    priority=s['priority'],
                    confidence=0.7,  # Lower confidence without web search validation
                    reasoning=s['reasoning']
                )
                for s in sources_data
            ]
        except:
            # Fallback: generate basic sources
            return self._generate_fallback_sources(provider)

    def _generate_fallback_sources(self, provider: str) -> List[SourceCandidate]:
        """
        Generate basic fallback sources when discovery fails.

        Uses heuristics based on common patterns.
        """
        provider_lower = provider.lower().replace(" ", "")

        # Common patterns for frontier AI providers
        fallback_urls = [
            f"https://{provider_lower}.com/blog",
            f"https://www.{provider_lower}.com/blog",
            f"https://blog.{provider_lower}.com",
            f"https://github.com/{provider_lower}",
        ]

        return [
            SourceCandidate(
                url=url,
                provider=provider,
                source_type="official_blog",
                priority="medium",
                confidence=0.5,  # Low confidence - just guessing
                reasoning="Fallback heuristic based on common patterns"
            )
            for url in fallback_urls
        ]

    def _enrich_with_reliability(
        self,
        sources: List[SourceCandidate]
    ) -> List[SourceCandidate]:
        """
        Enrich sources with reliability scores from database.

        Reliability = successful_events / total_fetches

        Sources with history get reliability_score populated.
        New sources have reliability_score = None.
        """
        for source in sources:
            # Query database for this URL
            reliability_data = self.db.get_sources_by_reliability(
                min_reliability=0.0,
                limit=1000  # Get all, we'll filter
            )

            # Find this URL in results
            for data in reliability_data:
                if data['url'] == source.url:
                    source.reliability_score = data['reliability_score']
                    break

        return sources

    def _rank_sources(self, sources: List[SourceCandidate]) -> List[SourceCandidate]:
        """
        Rank sources by composite score.

        Score = reliability_weight × reliability + confidence_weight × confidence + priority_weight

        Why this formula?
        - reliability_score: Historical performance (high = produced events)
        - confidence: How confident we are this is a good source
        - priority: high > medium > low

        New sources (no reliability) rely on confidence and priority.
        """
        priority_scores = {"high": 1.0, "medium": 0.67, "low": 0.33}

        def calculate_score(source: SourceCandidate) -> float:
            # Reliability (if available)
            reliability = source.reliability_score if source.reliability_score is not None else 0.5

            # Confidence
            confidence = source.confidence

            # Priority
            priority = priority_scores[source.priority]

            # Weighted average
            # Weight reliability highest (proven track record)
            return (0.5 * reliability) + (0.3 * confidence) + (0.2 * priority)

        # Sort by score (descending)
        sources.sort(key=calculate_score, reverse=True)

        return sources

    def _infer_source_type(self, url: str) -> str:
        """
        Infer source type from URL patterns.

        Simple heuristics for common patterns.
        """
        url_lower = url.lower()

        if 'github.com' in url_lower:
            return 'github'
        elif 'blog' in url_lower or 'news' in url_lower:
            return 'official_blog'
        elif 'docs' in url_lower or 'documentation' in url_lower:
            return 'documentation'
        elif 'api' in url_lower:
            return 'api_documentation'
        else:
            return 'website'
