"""
Quick script to validate source URLs are scrapable.
"""

import requests
import time

URLS_TO_TEST = [
    # OpenAI
    ("OpenAI Changelog", "https://platform.openai.com/docs/changelog"),
    ("OpenAI Blog", "https://openai.com/blog"),

    # Anthropic
    ("Anthropic Changelog", "https://docs.anthropic.com/en/release-notes/changelog"),
    ("Anthropic News", "https://www.anthropic.com/news"),

    # Google
    ("Google Gemini Changelog", "https://ai.google.dev/gemini-api/docs/changelog"),
    ("Google AI Blog", "https://blog.google/technology/ai/"),

    # Microsoft
    ("Microsoft Azure OpenAI", "https://learn.microsoft.com/en-us/azure/ai-services/openai/whats-new"),

    # Meta
    ("Meta AI Blog", "https://ai.meta.com/blog/"),

    # News
    ("TechCrunch AI", "https://techcrunch.com/tag/artificial-intelligence/"),
]

def test_url(name, url):
    """Test if URL is accessible and returns content."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        response = requests.get(url, headers=headers, timeout=10)

        status = response.status_code
        content_length = len(response.text)

        if status == 200 and content_length > 1000:
            print(f"[OK] {name}")
            print(f"     URL: {url}")
            print(f"     Status: {status}, Content: {content_length} chars")
            return True
        else:
            print(f"[WARN] {name}")
            print(f"       URL: {url}")
            print(f"       Status: {status}, Content: {content_length} chars")
            return False

    except Exception as e:
        print(f"[ERROR] {name}")
        print(f"        URL: {url}")
        print(f"        Error: {e}")
        return False

print("Testing source URLs...\n")

results = {}
for name, url in URLS_TO_TEST:
    results[name] = test_url(name, url)
    print()
    time.sleep(1)  # Rate limiting

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
working = sum(1 for v in results.values() if v)
total = len(results)
print(f"Working: {working}/{total}")
print("\nFailed URLs:")
for name, success in results.items():
    if not success:
        print(f"  - {name}")
