"""
Discover RSS feeds for curated sources.
"""

import requests
from bs4 import BeautifulSoup
import feedparser

def find_rss_feeds(url):
    """
    Find RSS/Atom feeds for a given URL.

    Strategy:
    1. Check common RSS locations
    2. Parse HTML for <link rel="alternate" type="application/rss+xml">
    3. Validate feed works
    """
    print(f"\nSearching for RSS feeds: {url}")

    feeds_found = []

    # Common RSS patterns
    common_patterns = [
        "/feed",
        "/feed/",
        "/rss",
        "/rss/",
        "/atom",
        "/atom.xml",
        "/rss.xml",
        "/feed.xml"
    ]

    # Try common patterns
    base_url = url.rstrip('/')
    for pattern in common_patterns:
        test_url = base_url + pattern
        try:
            feed = feedparser.parse(test_url)
            if feed.entries and len(feed.entries) > 0:
                feeds_found.append({
                    'url': test_url,
                    'title': feed.feed.get('title', 'Unknown'),
                    'entries': len(feed.entries)
                })
                print(f"  [FOUND] {test_url} ({len(feed.entries)} entries)")
        except:
            pass

    # Parse HTML for feed links
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for RSS/Atom links
        link_tags = soup.find_all('link', {'type': ['application/rss+xml', 'application/atom+xml']})
        for link in link_tags:
            feed_url = link.get('href')
            if feed_url:
                # Make absolute URL
                if feed_url.startswith('/'):
                    from urllib.parse import urlparse
                    parsed = urlparse(url)
                    feed_url = f"{parsed.scheme}://{parsed.netloc}{feed_url}"

                # Validate feed
                try:
                    feed = feedparser.parse(feed_url)
                    if feed.entries:
                        feeds_found.append({
                            'url': feed_url,
                            'title': feed.feed.get('title', 'Unknown'),
                            'entries': len(feed.entries)
                        })
                        print(f"  [FOUND] {feed_url} ({len(feed.entries)} entries)")
                except:
                    pass
    except Exception as e:
        print(f"  [ERROR] Could not parse HTML: {e}")

    if not feeds_found:
        print("  [NOT FOUND] No RSS feeds discovered")

    return feeds_found


# Test blog URLs
BLOG_URLS = [
    ("Anthropic News", "https://www.anthropic.com/news"),
    ("Google AI Blog", "https://blog.google/technology/ai/"),
    ("Meta AI Blog", "https://ai.meta.com/blog/"),
    ("Microsoft DevBlogs", "https://devblogs.microsoft.com/dotnet/category/ai/"),
    ("TechCrunch AI", "https://techcrunch.com/tag/artificial-intelligence/"),
    ("The Verge AI", "https://www.theverge.com/ai-artificial-intelligence"),
]

print("="*60)
print("RSS FEED DISCOVERY")
print("="*60)

results = {}
for name, url in BLOG_URLS:
    results[name] = find_rss_feeds(url)

print("\n" + "="*60)
print("SUMMARY")
print("="*60)

for name, feeds in results.items():
    if feeds:
        print(f"\n{name}:")
        for feed in feeds:
            print(f"  - {feed['url']}")
            print(f"    Title: {feed['title']}")
            print(f"    Entries: {feed['entries']}")
    else:
        print(f"\n{name}: No RSS feed found")
