"""
Content Harvester Agent

Fetches and filters content from sources.

Design: Two-stage pipeline
1. Scraper: Get raw text (deterministic, fast)
2. LLM Analysis: Determine significance (reasoning, filters noise)

Handles multiple source types:
- Web pages (BeautifulSoup + Trafilatura)
- APIs (api_adapters)
- GitHub (GitHub API)
- RSS feeds (feedparser)
- PDFs (PyPDF2)

Responsibilities:
1. Fetch content from URL
2. Parse based on source_type
3. LLM analyzes significance (0-10 score)
4. Filter out low-value content (score < threshold)
5. Track content changes (hash-based revision detection)
"""

import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from dateutil import parser as date_parser

import requests
import trafilatura
from bs4 import BeautifulSoup

from ..llm import LLMProvider, CONTENT_SIGNIFICANCE_SYSTEM_PROMPT, CONTENT_SIGNIFICANCE_USER_PROMPT
from ..storage import EventDatabase
from ..utils.api_adapters import APIAdapterFactory


@dataclass
class HarvestedContent:
    """
    Content fetched and analyzed by Content Harvester.

    Why dataclass?
    - Intermediate data structure (not stored long-term)
    - Simple validation (Pydantic overkill here)
    """
    url: str
    provider: str
    source_type: str

    # Raw content
    raw_text: str
    content_hash: str  # For change detection

    # LLM analysis
    significance_score: int  # 0-10
    content_type: str  # product_announcement, partnership, etc.
    filtered_content: str  # Relevant sections only
    metadata: Dict[str, Any]  # Dates, products, competitors mentioned
    llm_reasoning: str

    # Timestamps
    fetched_at: datetime
    fetch_successful: bool
    published_at: Optional[datetime] = None  # Extracted from content if available


class ContentHarvester:
    """
    Fetches and filters content from diverse sources.

    Why separate from Source Scout?
    - Different concerns: fetching vs discovering
    - Can be run independently (batch content fetching)
    - Different error handling (network vs logic)
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        database: EventDatabase,
        config: Dict[str, Any]
    ):
        """
        Initialize Content Harvester.

        Args:
            llm_provider: LLM for significance analysis
            database: Database for tracking fetches and source reliability
            config: Configuration dict
        """
        self.llm = llm_provider
        self.db = database
        self.config = config

        # Extract config
        harvester_config = config['agents']['content_harvester']
        self.user_agent = harvester_config.get('user_agent', 'Frontier-Market-Intelligence-Agent/1.0')
        self.timeout = harvester_config.get('timeout_seconds', 30)
        self.min_significance = harvester_config['significance_thresholds']['min_for_extraction']

        # API adapter factory
        self.api_factory = APIAdapterFactory(config)

    def harvest(
        self,
        url: str,
        provider: str,
        source_type: str
    ) -> Optional[HarvestedContent]:
        """
        Main entry point: Fetch and analyze content.

        Two-stage pipeline:
        1. Fetch & parse raw content
        2. LLM significance analysis

        Args:
            url: URL to fetch
            provider: Provider name (e.g., "OpenAI")
            source_type: Type of source (determines parser)

        Returns:
            HarvestedContent if successful and significant, else None

        Example:
            content = harvester.harvest(
                url="https://openai.com/blog/new-feature",
                provider="OpenAI",
                source_type="official_blog"
            )
            if content and content.significance_score >= 6:
                # Pass to Signal Extractor
        """
        try:
            # Stage 1: Fetch & parse
            raw_text = self._fetch_and_parse(url, source_type)

            if not raw_text or len(raw_text) < 100:
                print(f"Content too short or empty: {url}")
                self._update_source_reliability(url, False)
                return None

            # Calculate content hash (for change detection)
            content_hash = self._hash_content(raw_text)

            # Check if we've seen this exact content before
            if self._content_unchanged(url, content_hash):
                print(f"Content unchanged: {url}")
                return None

            # Stage 2: LLM significance analysis
            analysis = self._analyze_significance(raw_text, provider, url)

            # Extract publication date from metadata
            published_at = self._parse_published_date(analysis['metadata'])

            # Create result
            result = HarvestedContent(
                url=url,
                provider=provider,
                source_type=source_type,
                raw_text=raw_text,
                content_hash=content_hash,
                significance_score=analysis['significance_score'],
                content_type=analysis['content_type'],
                filtered_content=analysis['filtered_content'],
                metadata=analysis['metadata'],
                llm_reasoning=analysis['reasoning'],
                fetched_at=datetime.now(),
                fetch_successful=True,
                published_at=published_at
            )

            # Store in database (for tracking)
            self._store_fetch(result)

            # Update source reliability (successful if significant)
            self._update_source_reliability(url, result.significance_score >= self.min_significance)

            # Return only if significant enough
            if result.significance_score >= self.min_significance:
                return result
            else:
                print(f"Content not significant enough ({result.significance_score}/10): {url}")
                return None

        except Exception as e:
            print(f"Harvest failed for {url}: {e}")
            self._update_source_reliability(url, False)
            return None

    def _fetch_and_parse(self, url: str, source_type: str) -> str:
        """
        Stage 1: Fetch and parse content based on source type.

        Deterministic parsing - no LLM needed yet.

        Args:
            url: URL to fetch
            source_type: Determines which parser to use

        Returns:
            Raw text content

        Parsers:
        - api_*: Use API adapter
        - official_blog, website: Trafilatura (article extraction)
        - github: GitHub API (if available) or HTML
        - rss_feed: feedparser
        - pdf: PyPDF2
        """
        # Handle API sources
        if source_type.startswith('api_'):
            return self._fetch_from_api(url, source_type)

        # Handle web pages (blogs, documentation)
        if source_type in ['official_blog', 'website', 'documentation']:
            return self._fetch_web_page(url)

        # Handle GitHub
        if source_type == 'github':
            return self._fetch_github(url)

        # Fallback: generic web page
        return self._fetch_web_page(url)

    def _fetch_from_api(self, url: str, source_type: str) -> str:
        """
        Fetch from API using appropriate adapter.

        APIs return structured JSON. We convert to text for LLM analysis.

        Why convert JSON to text?
        - LLM needs to analyze significance (not just extract data)
        - Need to detect "what changed" (requires context)
        """
        try:
            # Determine which API
            if 'artificial_analysis' in url or source_type == 'api_llm_benchmarks':
                adapter = self.api_factory.get_adapter('artificial_analysis')

                # Fetch data
                if source_type == 'api_llm_benchmarks':
                    data = adapter.fetch_llm_models()
                elif source_type == 'api_image_gen_rankings':
                    data = adapter.fetch_text_to_image_models()
                else:
                    # Generic fetch
                    data = adapter.fetch_llm_models()

                # Convert to text format for LLM
                text = self._format_api_data_as_text(data, source_type)
                return text

            raise ValueError(f"Unknown API source: {source_type}")
        except Exception as e:
            print(f"API fetch error details: {type(e).__name__}: {e}")
            raise

    def _format_api_data_as_text(self, data: Dict[str, Any], source_type: str) -> str:
        """
        Format API JSON data as readable text for LLM analysis.

        Why text format?
        - LLM analyzes in natural language
        - Easier to detect significance ("new model X released")
        - Includes context (not just raw JSON)
        """
        if source_type == 'api_llm_benchmarks':
            # Get the API response data
            api_response = data.get('data', {})

            # Handle different possible API response structures
            if isinstance(api_response, list):
                models = api_response
            elif isinstance(api_response, dict):
                # Try common keys for model lists
                models = api_response.get('models', api_response.get('data', []))
            else:
                print(f"Warning: Unexpected API response type: {type(api_response)}")
                models = []

            # If still not a list, try to convert
            if not isinstance(models, list):
                print(f"Warning: Could not extract model list from API response")
                # Fall back to generic JSON dump
                return json.dumps(api_response, indent=2, default=str)

            lines = [
                "LLM Model Benchmarks from Artificial Analysis API",
                f"Retrieved: {data['retrieved_at'].strftime('%Y-%m-%d %H:%M')}",
                f"Total models tracked: {len(models)}",
                "",
                "Models:"
            ]

            for model in models[:50]:  # Limit to avoid token overflow
                if isinstance(model, dict):
                    # Handle different field name variations
                    model_name = model.get('model_name') or model.get('name') or model.get('model') or 'Unknown'
                    provider_name = model.get('provider') or model.get('provider_name') or 'Unknown'

                    lines.append(f"\n- Model: {model_name}")
                    lines.append(f"  Provider: {provider_name}")

                    # Pricing info
                    if 'price_input' in model or 'input_price' in model:
                        input_price = model.get('price_input') or model.get('input_price')
                        output_price = model.get('price_output') or model.get('output_price')
                        if input_price and output_price:
                            lines.append(f"  Pricing: ${input_price}/1M input, ${output_price}/1M output")

                    # Performance info
                    if 'output_speed' in model or 'speed' in model:
                        speed = model.get('output_speed') or model.get('speed')
                        if speed:
                            lines.append(f"  Speed: {speed} tokens/sec")

                    if 'quality_index' in model or 'quality' in model:
                        quality = model.get('quality_index') or model.get('quality')
                        if quality:
                            lines.append(f"  Quality Index: {quality}")

            return "\n".join(lines)

        # Generic JSON dump (with datetime serialization)
        return json.dumps(data, indent=2, default=str)

    def _fetch_web_page(self, url: str) -> str:
        """
        Fetch and extract main content from web page.

        Uses Trafilatura (better than BeautifulSoup for articles).

        Why Trafilatura?
        - Purpose-built for extracting main content
        - Removes navigation, ads, footers automatically
        - Handles diverse site structures
        """
        headers = {'User-Agent': self.user_agent}

        response = requests.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        # Use Trafilatura for main content extraction
        text = trafilatura.extract(response.text)

        if not text:
            # Fallback to BeautifulSoup if Trafilatura fails
            soup = BeautifulSoup(response.text, 'lxml')
            text = soup.get_text(separator='\n', strip=True)

        return text

    def _fetch_github(self, url: str) -> str:
        """
        Fetch from GitHub (releases, README, etc.).

        For MVP: Simple web scraping
        Future: Use GitHub API for structured data
        """
        return self._fetch_web_page(url)

    def _analyze_significance(
        self,
        raw_text: str,
        provider: str,
        url: str
    ) -> Dict[str, Any]:
        """
        Stage 2: LLM analyzes content significance.

        Uses GPT-4o-mini (fast classification task).

        Returns:
            Dict with:
                - significance_score (0-10)
                - content_type (product_announcement, etc.)
                - filtered_content (relevant sections only)
                - metadata (dates, products, competitors)
                - reasoning (why this score)
        """
        # Truncate if too long (to fit in context)
        max_length = 15000
        if len(raw_text) > max_length:
            raw_text = raw_text[:max_length] + "\n... (truncated)"

        # Build prompt
        prompt = CONTENT_SIGNIFICANCE_USER_PROMPT.format(
            provider=provider,
            url=url,
            content=raw_text
        )

        messages = [
            {"role": "system", "content": CONTENT_SIGNIFICANCE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]

        # Call LLM
        response = self.llm.generate(
            messages=messages,
            task_complexity="simple",  # GPT-4o-mini
            temperature=0.4  # Consistent scoring
        )

        # Parse JSON response (handle markdown code blocks)
        try:
            content = response['content'].strip()

            # Strip markdown code blocks if present
            if content.startswith('```'):
                # Remove opening ```json or ```
                lines = content.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                # Remove closing ```
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                content = '\n'.join(lines).strip()

            analysis = json.loads(content)

            # Build filtered content from relevant_sections
            filtered_content = "\n\n".join(analysis.get('relevant_sections', []))

            return {
                'significance_score': analysis.get('significance_score', 0),
                'content_type': analysis.get('content_type', 'other'),
                'filtered_content': filtered_content if filtered_content else raw_text[:5000],
                'metadata': analysis.get('metadata', {}),
                'reasoning': analysis.get('reasoning', '')
            }

        except json.JSONDecodeError as e:
            # Fallback: conservative scoring
            print(f"Failed to parse LLM response for {url}")
            print(f"  Parse error: {e}")
            print(f"  Response preview: {response['content'][:200]}...")
            return {
                'significance_score': 3,  # Low but not zero
                'content_type': 'other',
                'filtered_content': raw_text[:5000],
                'metadata': {},
                'reasoning': 'Could not parse LLM analysis'
            }

    def _hash_content(self, content: str) -> str:
        """
        Hash content for change detection.

        SHA256 hash of content (deterministic).
        """
        return hashlib.sha256(content.encode()).hexdigest()

    def _parse_published_date(self, metadata: Dict[str, Any]) -> Optional[datetime]:
        """
        Parse publication date from metadata.

        The LLM extracts 'announced_date' from content. We try to parse it.

        Args:
            metadata: Metadata dict from LLM analysis

        Returns:
            datetime or None if parsing fails

        Why this matters:
        - published_at is more accurate than fetched_at for event timeline
        - Enables correct temporal ordering of events
        - Important for "what happened first" analysis
        """
        announced_date = metadata.get('announced_date')

        if not announced_date:
            return None

        # Try to parse the date string
        try:
            # dateutil.parser is very flexible (handles many formats)
            parsed_date = date_parser.parse(announced_date, fuzzy=True)
            return parsed_date
        except (ValueError, TypeError) as e:
            # If parsing fails, return None (will fall back to fetched_at)
            print(f"Could not parse date '{announced_date}': {e}")
            return None

    def _content_unchanged(self, url: str, content_hash: str) -> bool:
        """
        Check if we've seen this exact content before.

        Queries database for previous fetches of this URL.
        If latest fetch has same hash, content unchanged.

        Why this matters:
        - Don't re-process unchanged content
        - Save LLM costs
        - Detect "quiet updates" (hash changes)
        """
        # Query database for latest fetch
        # For MVP: Skip this check (always process)
        # Future: Implement content_fetches table query
        return False

    def _store_fetch(self, content: HarvestedContent) -> None:
        """
        Store fetch record in database.

        Tracks:
        - What we fetched and when
        - Content hash (for change detection)
        - Significance score
        - Whether fetch was successful

        This enables:
        - Change detection (compare hashes)
        - Source reliability tracking
        - Audit trail
        """
        # For MVP: Skip storing individual fetches
        # Database schema supports this but not implemented yet
        pass

    def _update_source_reliability(self, url: str, successful: bool) -> None:
        """
        Update source reliability after fetch.

        successful = True if:
        - Fetch succeeded AND
        - Content significance >= threshold

        This feeds back into Source Scout's ranking.
        """
        self.db.update_source_reliability(url, successful)
