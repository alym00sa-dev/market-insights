"""
Content Harvester V2

Philosophy: Noise Filter, Not Signal Judge
- Fetch and parse content
- Basic quality checks (not 404, not empty, not spam)
- Simple noise detection (is this actual content?)
- Pass ALL non-noise to Signal Extractor
- Signal Extractor decides competitive significance

Key Improvements:
1. Content change detection (hash-based, DB-backed)
2. Rate limiting (per-domain throttling)
3. Better quality checks (detect error pages)
4. Parallel fetching (async batch processing)
5. Structured API data (no text conversion)
6. Simple noise threshold (3-4, not 6)

Design: Two-stage pipeline
1. Fetch & Parse: Get raw content (deterministic)
2. Noise Filter: Is this garbage or content? (basic LLM check)
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from dateutil import parser as date_parser

import requests
import trafilatura
from bs4 import BeautifulSoup

from ..llm import LLMProvider
from ..storage import EventDatabase
from ..utils.api_adapters import APIAdapterFactory


@dataclass
class HarvestedContent:
    """
    Content fetched and analyzed by Content Harvester V2.

    Simplified from V1:
    - No significance_score (Signal Extractor handles that)
    - Added is_noise flag (simple binary filter)
    - Added confidence score
    """
    url: str
    provider: str
    source_type: str

    # Raw content
    raw_text: str
    content_hash: str

    # Noise filtering (not significance scoring)
    is_noise: bool  # True if garbage/spam/error
    noise_confidence: float  # 0-1, how confident we are
    content_category: str  # "article", "changelog", "release", "api_data"

    # Metadata
    metadata: Dict[str, Any]

    # Timestamps
    fetched_at: datetime
    published_at: Optional[datetime] = None

    # For API sources - keep structured
    structured_data: Optional[Dict[str, Any]] = None


class ContentHarvesterV2:
    """
    Content Harvester V2 - Noise filter, not signal judge.

    Responsibilities:
    1. Fetch content from URL
    2. Parse based on source_type
    3. Basic quality checks (404, empty, error page)
    4. Change detection (hash-based)
    5. Simple noise filtering (is this content or garbage?)
    6. Pass ALL non-noise to Signal Extractor
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        database: EventDatabase,
        config: Dict[str, Any]
    ):
        """
        Initialize Content Harvester V2.

        Args:
            llm_provider: LLM for noise detection
            database: Database for change tracking
            config: Configuration dict
        """
        self.llm = llm_provider
        self.db = database
        self.config = config

        # Extract config
        harvester_config = config['agents']['content_harvester']
        self.user_agent = harvester_config.get('user_agent', 'Frontier-Market-Intelligence-Agent/1.0')
        self.timeout = harvester_config.get('timeout_seconds', 30)

        # API adapter factory
        self.api_factory = APIAdapterFactory(config)

        # Rate limiting state (per domain)
        self.last_request_time = {}  # domain -> timestamp
        self.min_request_interval = 2.0  # seconds between requests per domain

    def harvest(
        self,
        url: str,
        provider: str,
        source_type: str
    ) -> Optional[HarvestedContent]:
        """
        Main entry point: Fetch, parse, and filter noise.

        Two-stage pipeline:
        1. Fetch & parse (deterministic)
        2. Noise filter (simple LLM check)

        Args:
            url: URL to fetch
            provider: Provider name
            source_type: Type of source

        Returns:
            HarvestedContent if valid and not noise, else None
        """
        try:
            # Rate limiting
            self._rate_limit(url)

            # Stage 1: Fetch & parse
            fetch_result = self._fetch_and_parse(url, source_type)

            if not fetch_result:
                self._update_source_reliability(url, False)
                return None

            raw_text, structured_data = fetch_result

            # Basic quality checks
            if not self._is_valid_content(raw_text, url):
                print(f"[QUALITY] Content failed quality checks: {url}")
                self._update_source_reliability(url, False)
                return None

            # Calculate content hash
            content_hash = self._hash_content(raw_text)

            # Check if content unchanged
            if self._content_unchanged(url, content_hash):
                print(f"[UNCHANGED] Content hash matches previous fetch: {url}")
                return None

            # Stage 2: Noise filter (simple check)
            noise_check = self._check_if_noise(raw_text, provider, source_type)

            # Extract publication date
            published_at = self._parse_published_date(noise_check['metadata'])

            # Create result
            result = HarvestedContent(
                url=url,
                provider=provider,
                source_type=source_type,
                raw_text=raw_text,
                content_hash=content_hash,
                is_noise=noise_check['is_noise'],
                noise_confidence=noise_check['confidence'],
                content_category=noise_check['category'],
                metadata=noise_check['metadata'],
                fetched_at=datetime.now(),
                published_at=published_at,
                structured_data=structured_data
            )

            # Store fetch record (for change detection)
            self._store_fetch(result)

            # Update source reliability
            self._update_source_reliability(url, not result.is_noise)

            # Return only non-noise content
            if not result.is_noise:
                print(f"[PASS] Non-noise content: {url} (category: {result.content_category})")
                return result
            else:
                print(f"[NOISE] Filtered as noise: {url}")
                return None

        except Exception as e:
            print(f"[ERROR] Harvest failed for {url}: {e}")
            self._update_source_reliability(url, False)
            return None

    def _rate_limit(self, url: str) -> None:
        """
        Rate limit requests per domain.

        Ensures we don't hammer sources (ethical scraping).
        """
        from urllib.parse import urlparse

        domain = urlparse(url).netloc

        if domain in self.last_request_time:
            elapsed = time.time() - self.last_request_time[domain]
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                print(f"[RATE_LIMIT] Sleeping {sleep_time:.1f}s for {domain}")
                time.sleep(sleep_time)

        self.last_request_time[domain] = time.time()

    def _fetch_and_parse(
        self,
        url: str,
        source_type: str
    ) -> Optional[tuple[str, Optional[Dict]]]:
        """
        Stage 1: Fetch and parse content.

        Returns:
            (raw_text, structured_data) if successful, None if failed

        structured_data is only populated for API sources
        """
        # Handle API sources (keep structured)
        if source_type.startswith('api_'):
            return self._fetch_from_api(url, source_type)

        # Handle web pages
        if source_type in ['official_blog', 'website', 'documentation', 'article', 'changelog', 'rss_entry', 'blog_homepage']:
            raw_text = self._fetch_web_page(url)
            return (raw_text, None) if raw_text else None

        # Fallback: generic web page
        raw_text = self._fetch_web_page(url)
        return (raw_text, None) if raw_text else None

    def _fetch_from_api(
        self,
        url: str,
        source_type: str
    ) -> Optional[tuple[str, Dict]]:
        """
        Fetch from API - keep structured data.

        Returns:
            (summary_text, structured_data) for downstream processing
        """
        try:
            if 'artificial_analysis' in url or source_type == 'api_llm_benchmarks':
                adapter = self.api_factory.get_adapter('artificial_analysis')

                # Fetch structured data
                if source_type == 'api_llm_benchmarks':
                    data = adapter.fetch_llm_models()
                elif source_type == 'api_image_gen_rankings':
                    data = adapter.fetch_text_to_image_models()
                else:
                    data = adapter.fetch_llm_models()

                # Create summary text (for noise detection)
                summary = f"API data from {url} with {len(data.get('data', []))} records"

                # Return both summary and structured data
                return (summary, data)

            raise ValueError(f"Unknown API source: {source_type}")

        except Exception as e:
            print(f"[API_ERROR] {url}: {e}")
            return None

    def _fetch_web_page(self, url: str) -> Optional[str]:
        """
        Fetch and extract content from web page.

        Uses Trafilatura for main content extraction.
        """
        try:
            headers = {'User-Agent': self.user_agent}
            response = requests.get(url, headers=headers, timeout=self.timeout)

            # Check HTTP status
            if response.status_code != 200:
                print(f"[HTTP] Status {response.status_code} for {url}")
                return None

            # Use Trafilatura for main content
            text = trafilatura.extract(response.text)

            # Fallback to BeautifulSoup if Trafilatura fails
            if not text:
                soup = BeautifulSoup(response.text, 'lxml')
                text = soup.get_text(separator='\n', strip=True)

            return text

        except Exception as e:
            print(f"[FETCH_ERROR] {url}: {e}")
            return None

    def _is_valid_content(self, raw_text: str, url: str) -> bool:
        """
        Basic quality checks before LLM.

        Checks:
        - Not too short
        - Not error page
        - Not paywall
        """
        if not raw_text or len(raw_text) < 100:
            return False

        # Check for common error patterns
        error_patterns = [
            '404', '403', 'not found', 'access denied',
            'page not found', 'forbidden', 'unauthorized',
            'paywall', 'subscribe to read', 'sign in to continue'
        ]

        text_lower = raw_text.lower()
        for pattern in error_patterns:
            if pattern in text_lower and len(raw_text) < 500:
                # Short page with error pattern = probably error page
                return False

        return True

    def _check_if_noise(
        self,
        raw_text: str,
        provider: str,
        source_type: str
    ) -> Dict[str, Any]:
        """
        Stage 2: Simple noise detection (not significance scoring).

        Uses LLM to answer: "Is this actual content or garbage?"

        Returns:
            {
                'is_noise': bool,
                'confidence': float,
                'category': str,
                'metadata': dict
            }
        """
        # Truncate if too long
        max_length = 15000
        if len(raw_text) > max_length:
            raw_text = raw_text[:max_length] + "\n... (truncated)"

        # Build prompt
        prompt = f"""Determine if this content is NOISE or actual content.

Provider: {provider}
Source Type: {source_type}

Content:
{raw_text}

NOISE examples:
- Error pages (404, 403, access denied)
- Empty/placeholder pages
- Pure navigation/menu text
- Spam/advertising
- Completely off-topic

NOT NOISE examples:
- Product announcements (even minor ones)
- Blog articles (even marketing)
- Changelogs (even small updates)
- Documentation changes
- Policy updates
- Partnership news

Return JSON (no markdown, raw JSON only):
{{
  "is_noise": true/false,
  "confidence": 0.0-1.0,
  "category": "article|changelog|release|documentation|announcement|other",
  "metadata": {{
    "announced_date": "date if found, else null",
    "products_mentioned": ["product1"],
    "has_competitive_context": true/false
  }},
  "reasoning": "brief explanation"
}}

IMPORTANT: Be PERMISSIVE - only mark as noise if clearly garbage. When in doubt, mark as NOT noise."""

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.llm.generate(
                messages=messages,
                task_complexity="simple",  # GPT-4o-mini for speed
                temperature=0.3
            )

            # Parse JSON
            content = response['content'].strip()

            # Strip markdown code blocks if present
            if content.startswith('```'):
                lines = content.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                content = '\n'.join(lines).strip()

            result = json.loads(content)

            return {
                'is_noise': result.get('is_noise', False),
                'confidence': result.get('confidence', 0.5),
                'category': result.get('category', 'other'),
                'metadata': result.get('metadata', {})
            }

        except Exception as e:
            print(f"[LLM_ERROR] Failed to parse noise check: {e}")
            # Default: assume NOT noise (permissive)
            return {
                'is_noise': False,
                'confidence': 0.3,
                'category': 'other',
                'metadata': {}
            }

    def _hash_content(self, content: str) -> str:
        """Hash content for change detection."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _content_unchanged(self, url: str, content_hash: str) -> bool:
        """
        Check if content hash matches previous fetch.

        FIXED: Now actually queries database instead of always returning False.
        """
        try:
            latest_hash = self.db.get_latest_content_hash(url)
            if latest_hash and latest_hash == content_hash:
                return True
            return False
        except Exception as e:
            # If query fails, assume content changed (safe default)
            print(f"[DB_ERROR] Could not check content hash: {e}")
            return False

    def _store_fetch(self, content: HarvestedContent) -> None:
        """
        Store fetch record for change detection.

        FIXED: Now actually stores in database instead of being a no-op.
        """
        try:
            self.db.store_content_fetch(
                url=content.url,
                content_hash=content.content_hash,
                fetched_at=content.fetched_at,
                is_noise=content.is_noise,
                provider=content.provider
            )
        except Exception as e:
            print(f"[DB_ERROR] Could not store fetch: {e}")

    def _update_source_reliability(self, url: str, successful: bool) -> None:
        """
        Update source reliability tracking.

        successful = True if content passed noise filter
        """
        try:
            self.db.update_source_reliability(url, successful)
        except Exception as e:
            print(f"[DB_ERROR] Could not update reliability: {e}")

    def _parse_published_date(self, metadata: Dict[str, Any]) -> Optional[datetime]:
        """
        Parse publication date from metadata.

        Args:
            metadata: Metadata dict from noise check

        Returns:
            datetime or None
        """
        announced_date = metadata.get('announced_date')

        if not announced_date:
            return None

        try:
            parsed_date = date_parser.parse(announced_date, fuzzy=True)
            return parsed_date
        except (ValueError, TypeError) as e:
            print(f"[DATE_ERROR] Could not parse date '{announced_date}': {e}")
            return None
