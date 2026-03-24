"""
Link Extractor

Extracts article links from blog homepages when RSS is not available.

Strategy:
1. Fetch homepage HTML
2. Find article links (using heuristics)
3. Filter for recent/relevant links
4. Return list of article URLs
"""

import requests
from bs4 import BeautifulSoup
from typing import List, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime, timedelta


class LinkExtractor:
    """
    Extracts article links from blog homepages.

    For sources without RSS feeds (e.g., Anthropic News, Meta AI Blog).
    """

    def __init__(self, user_agent: str = "Mozilla/5.0"):
        self.user_agent = user_agent

    def extract_article_links(
        self,
        homepage_url: str,
        limit: int = 10,
        max_age_days: Optional[int] = None
    ) -> List[str]:
        """
        Extract article links from a blog homepage.

        Args:
            homepage_url: Blog homepage URL
            limit: Maximum number of links to return
            max_age_days: Only return links newer than this (if detectable)

        Returns:
            List of article URLs
        """
        try:
            # Fetch homepage
            headers = {'User-Agent': self.user_agent}
            response = requests.get(homepage_url, headers=headers, timeout=15)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract links using provider-specific strategies
            if 'anthropic.com' in homepage_url:
                links = self._extract_anthropic_links(soup, homepage_url)
            elif 'ai.meta.com' in homepage_url:
                links = self._extract_meta_links(soup, homepage_url)
            else:
                # Generic extraction
                links = self._extract_generic_links(soup, homepage_url)

            # Deduplicate and limit
            links = list(dict.fromkeys(links))  # Remove duplicates, preserve order
            return links[:limit]

        except Exception as e:
            print(f"Link extraction failed for {homepage_url}: {e}")
            return []

    def _extract_anthropic_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract article links from Anthropic News page.

        Strategy: Find links with /news/ in the path
        """
        links = []

        # Find all links
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Make absolute URL
            absolute_url = urljoin(base_url, href)

            # Filter for news articles
            if '/news/' in absolute_url and absolute_url != base_url:
                # Avoid pagination links
                if not any(x in absolute_url for x in ['page=', '?', '#']):
                    links.append(absolute_url)

        return links

    def _extract_meta_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract article links from Meta AI Blog.

        Strategy: Find links with /blog/ and exclude homepage
        """
        links = []

        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(base_url, href)

            # Filter for blog posts
            if '/blog/' in absolute_url:
                # Exclude homepage and category pages
                parsed = urlparse(absolute_url)
                path_parts = [p for p in parsed.path.split('/') if p]

                # Blog post should have at least: ['blog', 'post-slug']
                if len(path_parts) >= 2 and path_parts[0] == 'blog':
                    # Avoid pagination/category
                    if not any(x in absolute_url for x in ['page=', 'category=', '?', '#']):
                        links.append(absolute_url)

        return links

    def _extract_generic_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Generic article link extraction.

        Strategy:
        - Find links in <article> tags
        - Find links with article-like patterns
        - Filter out navigation/footer links
        """
        links = []

        # Try finding <article> tags first
        articles = soup.find_all('article')
        for article in articles:
            for link in article.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(base_url, href)

                # Basic filtering
                if self._looks_like_article(absolute_url, base_url):
                    links.append(absolute_url)

        # If no articles found, try finding all links
        if not links:
            for link in soup.find_all('a', href=True):
                href = link['href']
                absolute_url = urljoin(base_url, href)

                if self._looks_like_article(absolute_url, base_url):
                    links.append(absolute_url)

        return links

    def _looks_like_article(self, url: str, base_url: str) -> bool:
        """
        Heuristic to determine if a URL looks like an article.

        Returns:
            True if URL likely points to an article
        """
        # Must be from same domain
        if not url.startswith(base_url.rstrip('/')):
            return False

        # Exclude common non-article patterns
        exclude_patterns = [
            '/tag/', '/category/', '/author/',
            '/page/', '/search/', '/archive/',
            '?', '#', 'mailto:', 'javascript:',
            '.pdf', '.jpg', '.png', '.gif'
        ]

        if any(pattern in url for pattern in exclude_patterns):
            return False

        # Must have path beyond base
        if url.rstrip('/') == base_url.rstrip('/'):
            return False

        return True
