"""
Monitored Source dataclass and source registries.

Defines sources that are actively monitored by Source Scout.

NOTE: All URLs have been validated for scrapability.
- RSS feeds are preferred (direct article access)
- Blog homepages require link extraction
- GitHub releases are scrapable
- Changelogs are direct content pages
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal


@dataclass
class MonitoredSource:
    """
    A source being actively monitored by Source Scout.

    Tracks scheduling, performance, and metadata for continuous monitoring.
    """
    url: str
    provider: str  # "OpenAI", "Anthropic", "Google", "Microsoft", "Meta", "all"
    source_type: str  # "changelog", "official_blog", "github_releases", "rss_feed", "blog_homepage"
    priority: Literal["critical", "high", "medium", "low"]
    check_frequency: Literal["hourly", "daily", "weekly", "monthly"]

    # Link extraction flag
    needs_link_extraction: bool = False  # True for blog homepages without RSS

    # Tracking
    last_checked_at: Optional[datetime] = None
    last_updated_at: Optional[datetime] = None  # When source had new content
    total_checks: int = 0
    successful_checks: int = 0  # Content passed to harvester
    reliability_score: float = 0.0

    # Discovery metadata
    discovered_via: str = "manual"  # "manual", "web_search", "rss_discovery"
    discovered_at: datetime = field(default_factory=datetime.now)


# Hyperscaler source registry - validated, scrapable sources
HYPERSCALER_SOURCES = {
    "OpenAI": [
        # OpenAI blocks direct scraping (Cloudflare), use GitHub
        MonitoredSource(
            url="https://github.com/openai/openai-python/releases",
            provider="OpenAI",
            source_type="github_releases",
            priority="critical",
            check_frequency="daily",
            needs_link_extraction=True  # Extract individual release links
        ),
        MonitoredSource(
            url="https://github.com/openai/openai-cookbook",
            provider="OpenAI",
            source_type="github_releases",
            priority="medium",
            check_frequency="weekly",
            needs_link_extraction=True
        ),
    ],

    "Anthropic": [
        # Changelog - direct scrapable page
        MonitoredSource(
            url="https://docs.anthropic.com/en/release-notes/changelog",
            provider="Anthropic",
            source_type="changelog",
            priority="critical",
            check_frequency="daily"
        ),
        # News - NO RSS, needs link extraction
        MonitoredSource(
            url="https://www.anthropic.com/news",
            provider="Anthropic",
            source_type="blog_homepage",
            priority="high",
            check_frequency="daily",
            needs_link_extraction=True  # Extract article links from homepage
        ),
        MonitoredSource(
            url="https://github.com/anthropics/anthropic-sdk-python/releases",
            provider="Anthropic",
            source_type="github_releases",
            priority="high",
            check_frequency="weekly",
            needs_link_extraction=True
        )
    ],

    "Google": [
        # Changelog - direct scrapable page
        MonitoredSource(
            url="https://ai.google.dev/gemini-api/docs/changelog",
            provider="Google",
            source_type="changelog",
            priority="critical",
            check_frequency="daily"
        ),
        # Blog RSS feed - direct article access
        MonitoredSource(
            url="https://blog.google/technology/ai/rss",
            provider="Google",
            source_type="rss_feed",
            priority="high",
            check_frequency="daily"
        ),
        MonitoredSource(
            url="https://github.com/google/generative-ai-python/releases",
            provider="Google",
            source_type="github_releases",
            priority="medium",
            check_frequency="weekly",
            needs_link_extraction=True
        )
    ],

    "Microsoft": [
        # Changelog - direct scrapable page
        MonitoredSource(
            url="https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new",
            provider="Microsoft",
            source_type="changelog",
            priority="critical",
            check_frequency="daily"
        ),
        # DevBlogs RSS feed - direct article access
        MonitoredSource(
            url="https://devblogs.microsoft.com/dotnet/category/ai/feed",
            provider="Microsoft",
            source_type="rss_feed",
            priority="medium",
            check_frequency="weekly"
        ),
    ],

    "Meta": [
        # Blog homepage - NO RSS, needs link extraction
        MonitoredSource(
            url="https://ai.meta.com/blog/",
            provider="Meta",
            source_type="blog_homepage",
            priority="high",
            check_frequency="daily",
            needs_link_extraction=True  # Extract article links from homepage
        ),
        MonitoredSource(
            url="https://github.com/meta-llama/llama/releases",
            provider="Meta",
            source_type="github_releases",
            priority="critical",
            check_frequency="weekly",
            needs_link_extraction=True
        ),
        MonitoredSource(
            url="https://github.com/meta-llama/llama-stack/releases",
            provider="Meta",
            source_type="github_releases",
            priority="high",
            check_frequency="weekly",
            needs_link_extraction=True
        )
    ]
}

# Cross-provider industry news sources - all have RSS feeds
CROSS_PROVIDER_SOURCES = [
    MonitoredSource(
        url="https://techcrunch.com/tag/artificial-intelligence/feed",
        provider="all",
        source_type="rss_feed",
        priority="medium",
        check_frequency="daily"
    ),
    MonitoredSource(
        url="https://www.theverge.com/rss/ai-artificial-intelligence/index.xml",
        provider="all",
        source_type="rss_feed",
        priority="low",
        check_frequency="weekly"
    ),
]

# External API sources - structured data (not RSS/HTML)
API_SOURCES = [
    MonitoredSource(
        url="https://artificialanalysis.ai/api/v2/data/llms/models",
        provider="all",
        source_type="api_llm_benchmarks",
        priority="high",
        check_frequency="daily",  # Model releases/pricing changes are important
        needs_link_extraction=False
    ),
    MonitoredSource(
        url="https://artificialanalysis.ai/api/v2/data/media/text-to-image",
        provider="all",
        source_type="api_image_gen_rankings",
        priority="medium",
        check_frequency="weekly",
        needs_link_extraction=False
    ),
    MonitoredSource(
        url="https://artificialanalysis.ai/api/v2/data/media/text-to-speech",
        provider="all",
        source_type="api_tts_rankings",
        priority="medium",
        check_frequency="weekly",
        needs_link_extraction=False
    ),
    MonitoredSource(
        url="https://artificialanalysis.ai/api/v2/data/media/text-to-video",
        provider="all",
        source_type="api_video_gen_rankings",
        priority="medium",
        check_frequency="weekly",
        needs_link_extraction=False
    ),
]
