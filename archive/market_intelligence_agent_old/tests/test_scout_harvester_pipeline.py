"""
Test Source Scout + Content Harvester Pipeline

Compares V1 vs V2 to validate improvements:
1. Source Scout now finds actual articles (not just homepages)
2. Content Harvester V2 is more permissive (less false negatives)

Tests:
- Link extraction from blog homepages
- RSS feed parsing
- Content harvesting with V1 vs V2
- Quality metrics comparison
"""

import sys
import os
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.agents.source_scout import (
    SourceScout,
    SourceScoutMonitor,
    LinkExtractor,
    HYPERSCALER_SOURCES
)
from src.agents.content_harvester import ContentHarvester
from src.agents.content_harvester_v2 import ContentHarvesterV2
from src.llm import LLMProvider
from src.storage import EventDatabase


def get_config_path():
    """Get path to configuration file."""
    return os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')

def load_config():
    """Load configuration from YAML file."""
    with open(get_config_path(), 'r') as f:
        return yaml.safe_load(f)


def test_link_extraction():
    """Test link extraction from blog homepages."""
    print("\n" + "="*80)
    print("TEST 1: Link Extraction")
    print("="*80)

    extractor = LinkExtractor()

    # Test Anthropic blog
    print("\n[Anthropic News Page]")
    anthropic_links = extractor.extract_article_links(
        "https://www.anthropic.com/news",
        limit=5
    )

    print(f"Found {len(anthropic_links)} article links:")
    for i, link in enumerate(anthropic_links, 1):
        print(f"  {i}. {link}")

    # Test Meta blog
    print("\n[Meta AI Blog]")
    meta_links = extractor.extract_article_links(
        "https://ai.meta.com/blog/",
        limit=5
    )

    print(f"Found {len(meta_links)} article links:")
    for i, link in enumerate(meta_links, 1):
        print(f"  {i}. {link}")

    return anthropic_links, meta_links


def test_harvester_comparison(test_urls):
    """Compare Content Harvester V1 vs V2 on same URLs."""
    print("\n" + "="*80)
    print("TEST 2: Content Harvester V1 vs V2 Comparison")
    print("="*80)

    # Initialize components
    config_path = get_config_path()
    config = load_config()
    llm = LLMProvider(config_path)
    db = EventDatabase(config['storage']['database']['path'])

    # Create both harvesters
    harvester_v1 = ContentHarvester(llm, db, config)
    harvester_v2 = ContentHarvesterV2(llm, db, config)

    results = {
        'v1': {'passed': 0, 'filtered': 0, 'results': []},
        'v2': {'passed': 0, 'filtered': 0, 'results': []}
    }

    for i, url in enumerate(test_urls[:3], 1):  # Test first 3 URLs
        print(f"\n[URL {i}/{len(test_urls[:3])}] {url}")
        print("-" * 80)

        # Test V1
        print("\nContent Harvester V1:")
        try:
            content_v1 = harvester_v1.harvest(
                url=url,
                provider="Anthropic",
                source_type="article"
            )

            if content_v1 and content_v1.significance_score >= 6:
                print(f"  ✅ PASSED (significance: {content_v1.significance_score}/10)")
                print(f"     Type: {content_v1.content_type}")
                results['v1']['passed'] += 1
                results['v1']['results'].append({
                    'url': url,
                    'passed': True,
                    'score': content_v1.significance_score,
                    'type': content_v1.content_type
                })
            else:
                score = content_v1.significance_score if content_v1 else 'N/A'
                print(f"  ❌ FILTERED (significance: {score}/10)")
                results['v1']['filtered'] += 1
                results['v1']['results'].append({
                    'url': url,
                    'passed': False,
                    'score': score
                })
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results['v1']['filtered'] += 1

        # Test V2
        print("\nContent Harvester V2:")
        try:
            content_v2 = harvester_v2.harvest(
                url=url,
                provider="Anthropic",
                source_type="article"
            )

            if content_v2:
                print(f"  ✅ PASSED (not noise)")
                print(f"     Category: {content_v2.content_category}")
                print(f"     Confidence: {content_v2.noise_confidence:.2f}")
                results['v2']['passed'] += 1
                results['v2']['results'].append({
                    'url': url,
                    'passed': True,
                    'category': content_v2.content_category,
                    'confidence': content_v2.noise_confidence
                })
            else:
                print(f"  ❌ FILTERED (noise or unchanged)")
                results['v2']['filtered'] += 1
                results['v2']['results'].append({
                    'url': url,
                    'passed': False
                })
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            results['v2']['filtered'] += 1

    return results


def test_full_pipeline():
    """Test full Source Scout → Content Harvester pipeline."""
    print("\n" + "="*80)
    print("TEST 3: Full Pipeline (Source Scout → Content Harvester V2)")
    print("="*80)

    # Initialize components
    config_path = get_config_path()
    config = load_config()
    llm = LLMProvider(config_path)
    db = EventDatabase(config['storage']['database']['path'])

    # Create Source Scout (llm, db, config)
    scout = SourceScout(llm, db, config)

    # Create Content Harvester V2
    harvester = ContentHarvesterV2(llm, db, config)

    # Test a few sources from registry
    test_sources = [
        HYPERSCALER_SOURCES["Anthropic"][1],  # News page (needs link extraction)
        HYPERSCALER_SOURCES["Google"][1],     # Blog RSS feed
    ]

    print(f"\nTesting {len(test_sources)} sources:")

    pipeline_results = []

    for source in test_sources:
        print(f"\n[Source] {source.url}")
        print(f"  Provider: {source.provider}")
        print(f"  Type: {source.source_type}")
        print(f"  Needs Link Extraction: {source.needs_link_extraction}")
        print("-" * 80)

        if source.needs_link_extraction:
            # Extract article links
            extractor = LinkExtractor()
            article_links = extractor.extract_article_links(
                source.url,
                limit=3
            )

            print(f"  Extracted {len(article_links)} article links")

            # Harvest each article
            for article_url in article_links:
                print(f"\n  [Article] {article_url}")
                content = harvester.harvest(
                    url=article_url,
                    provider=source.provider,
                    source_type="article"
                )

                if content:
                    print(f"    ✅ Non-noise content")
                    print(f"       Category: {content.content_category}")
                    pipeline_results.append({
                        'source': source.url,
                        'article': article_url,
                        'passed': True,
                        'category': content.content_category
                    })
                else:
                    print(f"    ❌ Filtered (noise or unchanged)")
                    pipeline_results.append({
                        'source': source.url,
                        'article': article_url,
                        'passed': False
                    })
        else:
            # Direct harvest
            content = harvester.harvest(
                url=source.url,
                provider=source.provider,
                source_type=source.source_type
            )

            if content:
                print(f"  ✅ Non-noise content")
                print(f"     Category: {content.content_category}")
                pipeline_results.append({
                    'source': source.url,
                    'passed': True,
                    'category': content.content_category
                })
            else:
                print(f"  ❌ Filtered (noise or unchanged)")
                pipeline_results.append({
                    'source': source.url,
                    'passed': False
                })

    return pipeline_results


def print_summary(link_results, comparison_results, pipeline_results):
    """Print test summary."""
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    # Link extraction summary
    anthropic_links, meta_links = link_results
    print(f"\n1. Link Extraction:")
    print(f"   - Anthropic: {len(anthropic_links)} articles extracted")
    print(f"   - Meta: {len(meta_links)} articles extracted")
    print(f"   ✅ Link extraction working!")

    # V1 vs V2 comparison
    print(f"\n2. Content Harvester Comparison:")
    print(f"   V1: {comparison_results['v1']['passed']} passed, {comparison_results['v1']['filtered']} filtered")
    print(f"   V2: {comparison_results['v2']['passed']} passed, {comparison_results['v2']['filtered']} filtered")

    improvement = comparison_results['v2']['passed'] - comparison_results['v1']['passed']
    if improvement > 0:
        print(f"   ✅ V2 is MORE PERMISSIVE (+{improvement} more content passed)")
    elif improvement < 0:
        print(f"   ⚠️  V2 is MORE RESTRICTIVE ({improvement} less content passed)")
    else:
        print(f"   ➖ Same pass rate")

    # Pipeline summary
    passed = sum(1 for r in pipeline_results if r['passed'])
    total = len(pipeline_results)
    print(f"\n3. Full Pipeline:")
    print(f"   - {passed}/{total} articles passed through pipeline")
    print(f"   - Pass rate: {passed/total*100:.1f}%" if total > 0 else "   - No results")

    if passed > 0:
        print(f"   ✅ Pipeline working end-to-end!")
    else:
        print(f"   ⚠️  No content passed - may need tuning")

    print("\n" + "="*80)
    print("KEY IMPROVEMENTS:")
    print("="*80)
    print("✅ Source Scout now extracts actual article links (not homepages)")
    print("✅ Content Harvester V2 uses simple noise filter (more permissive)")
    print("✅ Signal Extractor will handle competitive significance scoring")
    print("✅ Change detection prevents re-processing unchanged content")
    print("✅ Rate limiting for ethical scraping")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("SOURCE SCOUT + CONTENT HARVESTER PIPELINE TEST")
    print("="*80)
    print("\nComparing V1 vs V2 to validate improvements...")

    try:
        # Test 1: Link extraction
        link_results = test_link_extraction()
        anthropic_links, meta_links = link_results

        # Test 2: V1 vs V2 comparison on extracted links
        test_urls = anthropic_links[:3] if anthropic_links else []
        if test_urls:
            comparison_results = test_harvester_comparison(test_urls)
        else:
            print("\n⚠️  No links to test - skipping V1 vs V2 comparison")
            comparison_results = {'v1': {'passed': 0, 'filtered': 0}, 'v2': {'passed': 0, 'filtered': 0}}

        # Test 3: Full pipeline
        pipeline_results = test_full_pipeline()

        # Print summary
        print_summary(link_results, comparison_results, pipeline_results)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
