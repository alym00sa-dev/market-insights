"""
RSS Feed Scrape Quality Tester
Evaluates each source on:
  1. Scrapability    - can we fetch & parse the feed?
  2. AI Relevance    - how much of the content is AI-specific?
  3. Recency         - how many articles in the past 14 days?
"""

import feedparser
import requests
import time
import json
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
from typing import Optional

CUTOFF = datetime.now(timezone.utc) - timedelta(days=14)

AI_KEYWORDS = [
    "artificial intelligence", "machine learning", "deep learning", "neural network",
    "large language model", "llm", "gpt", "claude", "gemini", "openai", "anthropic",
    "transformer", "generative ai", "gen ai", "foundation model", "language model",
    "ai agent", "chatgpt", "mistral", "llama", "inference", "fine-tuning", "rag",
    "multimodal", "diffusion model", "embedding", "vector", "nlp", "computer vision",
    "reinforcement learning", "benchmark", "reasoning model", "o1", "o3",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FeedTester/1.0)"
}

RSS_FEEDS = [
    ("404 Media", "https://www.404media.co/rss"),
    ("Ahead of AI", "https://magazine.sebastianraschka.com/feed"),
    ("AI Accelerator Institute", "https://aiacceleratorinstitute.com/rss/"),
    ("AI-TechPark", "https://ai-techpark.com/category/ai/feed/"),
    ("KnowTechie AI", "https://knowtechie.com/category/ai/feed/"),
    ("AI Business", "https://aibusiness.com/rss.xml"),
    ("AIModels.fyi", "https://aimodels.substack.com/feed"),
    ("Artificial Intelligence News", "https://www.artificialintelligence-news.com/feed/rss/"),
    ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
    ("AI Now Institute", "https://ainowinstitute.org/category/news/feed"),
    ("SiliconANGLE AI", "https://siliconangle.com/category/ai/feed"),
    ("AI Snake Oil", "https://aisnakeoil.substack.com/feed"),
    ("Anaconda Blog", "https://www.anaconda.com/blog/feed"),
    ("Analytics India Magazine", "https://analyticsindiamag.com/feed/"),
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    ("The Guardian AI", "https://www.theguardian.com/technology/artificialintelligenceai/rss"),
    ("Futurism AI", "https://futurism.com/categories/ai-artificial-intelligence/feed"),
    ("Wired AI", "https://www.wired.com/feed/tag/ai/latest/rss"),
    ("ScienceDaily AI", "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml"),
    ("TechRepublic AI", "https://www.techrepublic.com/rssfeeds/topic/artificial-intelligence/"),
    ("Bloomberg Tech", "https://feeds.bloomberg.com/technology/news.rss"),
    ("Business Insider", "https://feeds.businessinsider.com/custom/all"),
    ("Chain of Thought (Every)", "https://every.to/chain-of-thought/feed.xml"),
    ("Chip Huyen", "https://huyenchip.com/feed"),
    ("Computerworld", "http://www.computerworld.com/index.rss"),
    ("Crunchbase News", "https://news.crunchbase.com/feed"),
    ("arXiv cs.CL", "https://arxiv.org/rss/cs.CL"),
    ("arXiv cs.LG", "https://arxiv.org/rss/cs.LG"),
    ("Datanami", "https://www.datanami.com/feed/"),
    ("DeepMind Blog", "https://deepmind.com/blog/feed/basic/"),
    ("DEV Community", "https://dev.to/feed"),
    ("Engadget", "https://www.engadget.com/rss.xml"),
    ("Freethink", "https://www.freethink.com/feed/all"),
    ("Generational", "https://www.generational.pub/feed"),
    ("gHacks", "https://www.ghacks.net/feed/"),
    ("Gizmodo", "https://gizmodo.com/rss"),
    ("Google AI Blog", "http://googleaiblog.blogspot.com/atom.xml"),
    ("Gradient Flow", "https://gradientflow.com/feed/"),
    ("Hacker Noon AI", "https://hackernoon.com/tagged/ai/feed"),
    ("Hugging Face Blog", "https://huggingface.co/blog/feed.xml"),
    ("IEEE Spectrum AI", "https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss"),
    ("InfoQ AI/ML", "https://feed.infoq.com/ai-ml-data-eng/"),
    ("InfoWorld ML", "https://www.infoworld.com/category/machine-learning/index.rss"),
    ("KDnuggets", "https://www.kdnuggets.com/feed"),
    ("LangChain Blog", "https://blog.langchain.dev/rss/"),
    ("Last Week in AI", "https://lastweekin.ai/feed"),
    ("Latent Space", "https://www.latent.space/feed"),
    ("ZDNET AI", "https://www.zdnet.com/topic/artificial-intelligence/rss.xml"),
    ("ML@CMU Blog", "https://blog.ml.cmu.edu/feed"),
    ("MarkTechPost", "https://www.marktechpost.com/feed"),
    ("Microsoft Research", "https://www.microsoft.com/en-us/research/feed/"),
    ("MIT News ML", "https://news.mit.edu/topic/mitmachine-learning-rss.xml"),
    ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
    ("New Scientist Tech", "https://www.newscientist.com/subject/technology/feed/"),
    ("NVIDIA Dev Blog", "https://developer.nvidia.com/blog/feed"),
    ("NY Times Tech", "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"),
    ("One Useful Thing", "https://www.oneusefulthing.org/feed"),
    ("OpenAI Blog", "https://openai.com/blog/rss/"),
    ("Quanta Magazine", "https://api.quantamagazine.org/feed"),
    ("Replicate Blog", "https://replicate.com/blog/rss"),
    ("Rest of World", "https://restofworld.org/feed/latest"),
    ("SemiAnalysis", "https://www.semianalysis.com/feed"),
    ("Simon Willison", "https://simonwillison.net/atom/everything/"),
    ("Stack Overflow Blog", "https://stackoverflow.blog/feed/"),
    ("Synced Review", "https://syncedreview.com/feed"),
    ("TechCrunch", "https://techcrunch.com/feed/"),
    ("Techmeme", "https://www.techmeme.com/feed.xml"),
    ("The Algorithmic Bridge", "https://thealgorithmicbridge.substack.com/feed"),
    ("The Decoder", "https://the-decoder.com/feed/"),
    ("The Gradient", "https://thegradient.pub/rss/"),
    ("The New Stack", "https://thenewstack.io/feed"),
    ("The Next Web Neural", "https://thenextweb.com/neural/feed"),
    ("The Register AI", "https://www.theregister.com/software/ai_ml/headlines.atom"),
    ("The Sequence", "https://thesequence.substack.com/feed"),
    ("The Verge", "https://www.theverge.com/rss/index.xml"),
    ("Towards AI", "https://pub.towardsai.net/feed"),
    ("Towards Data Science", "https://towardsdatascience.com/feed"),
    ("Unite.AI", "https://www.unite.ai/feed/"),
    ("Interconnects", "https://www.interconnects.ai/feed"),
    ("Unwind AI", "https://unwindai.substack.com/feed"),
    ("Reuters Tech", "https://www.reutersagency.com/feed/?best-topics=tech"),
    ("bdTechTalks", "https://bdtechtalks.com/feed/"),
    ("BAIR Blog", "https://bair.berkeley.edu/blog/feed.xml"),
    ("Databricks Blog", "https://www.databricks.com/feed"),
    ("Phys.org AI", "https://phys.org/rss-feed/technology-news/machine-learning-ai/"),
    ("TechXplore AI", "https://techmonitor.ai/feed"),
    ("WandB Blog", "https://wandb.ai/fully-connected/rss.xml"),
    ("Wolfram Blog", "https://blog.wolfram.com/feed/"),
]


@dataclass
class FeedResult:
    name: str
    url: str
    status: str                    # ok | timeout | error | parse_error
    http_code: Optional[int]
    total_items: int
    recent_items: int              # past 14 days
    ai_relevant_items: int         # items with AI keywords in title/summary
    ai_relevance_pct: float        # % of total items that are AI-relevant
    recency_score: str             # hot / warm / stale / dead
    scrape_score: str              # great / ok / blocked / dead
    sample_titles: list[str]
    error: Optional[str] = None


def score_text(text: str) -> bool:
    """Returns True if text contains AI keywords."""
    lower = text.lower()
    return any(kw in lower for kw in AI_KEYWORDS)


def parse_date(entry) -> Optional[datetime]:
    """Try to extract a timezone-aware datetime from a feed entry."""
    for field in ("published_parsed", "updated_parsed"):
        t = getattr(entry, field, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def test_feed(name: str, url: str) -> FeedResult:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12, allow_redirects=True)
        http_code = resp.status_code

        if resp.status_code in (403, 401, 429):
            return FeedResult(
                name=name, url=url, status="blocked", http_code=http_code,
                total_items=0, recent_items=0, ai_relevant_items=0,
                ai_relevance_pct=0.0, recency_score="dead", scrape_score="blocked",
                sample_titles=[], error=f"HTTP {resp.status_code}"
            )

        if resp.status_code >= 400:
            return FeedResult(
                name=name, url=url, status="error", http_code=http_code,
                total_items=0, recent_items=0, ai_relevant_items=0,
                ai_relevance_pct=0.0, recency_score="dead", scrape_score="dead",
                sample_titles=[], error=f"HTTP {resp.status_code}"
            )

        feed = feedparser.parse(resp.content)
        entries = feed.entries

        if not entries:
            return FeedResult(
                name=name, url=url, status="parse_error", http_code=http_code,
                total_items=0, recent_items=0, ai_relevant_items=0,
                ai_relevance_pct=0.0, recency_score="dead", scrape_score="ok",
                sample_titles=[], error="No entries parsed"
            )

        total = len(entries)
        recent = 0
        ai_hits = 0
        sample_titles = []

        for entry in entries[:50]:  # cap at 50 for speed
            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            text = f"{title} {summary}"

            if score_text(text):
                ai_hits += 1

            pub = parse_date(entry)
            if pub and pub >= CUTOFF:
                recent += 1

            if len(sample_titles) < 3 and title:
                sample_titles.append(title[:100])

        ai_pct = round((ai_hits / min(total, 50)) * 100, 1)

        # Recency score
        if recent >= 5:
            recency = "hot"
        elif recent >= 2:
            recency = "warm"
        elif recent >= 1:
            recency = "stale"
        else:
            recency = "dead"

        # Scrape score
        if total >= 10 and http_code == 200:
            scrape = "great"
        elif total >= 3:
            scrape = "ok"
        else:
            scrape = "poor"

        return FeedResult(
            name=name, url=url, status="ok", http_code=http_code,
            total_items=total, recent_items=recent,
            ai_relevant_items=ai_hits, ai_relevance_pct=ai_pct,
            recency_score=recency, scrape_score=scrape,
            sample_titles=sample_titles
        )

    except requests.exceptions.Timeout:
        return FeedResult(
            name=name, url=url, status="timeout", http_code=None,
            total_items=0, recent_items=0, ai_relevant_items=0,
            ai_relevance_pct=0.0, recency_score="dead", scrape_score="dead",
            sample_titles=[], error="Timeout"
        )
    except Exception as e:
        return FeedResult(
            name=name, url=url, status="error", http_code=None,
            total_items=0, recent_items=0, ai_relevant_items=0,
            ai_relevance_pct=0.0, recency_score="dead", scrape_score="dead",
            sample_titles=[], error=str(e)[:120]
        )


def print_results(results: list[FeedResult]):
    print(f"\n{'='*100}")
    print(f"{'SOURCE':<35} {'SCRAPE':<8} {'RECENCY':<8} {'ITEMS':<7} {'14-DAY':<7} {'AI%':<7} {'SAMPLE TITLE'}")
    print(f"{'='*100}")

    # Sort: best first (hot+great, then by AI%)
    order = {"hot": 0, "warm": 1, "stale": 2, "dead": 3}
    results.sort(key=lambda r: (order.get(r.recency_score, 4), -r.ai_relevance_pct))

    for r in results:
        title_preview = r.sample_titles[0][:55] if r.sample_titles else (r.error or "")
        print(
            f"{r.name:<35} {r.scrape_score:<8} {r.recency_score:<8} "
            f"{r.total_items:<7} {r.recent_items:<7} {r.ai_relevance_pct:<7} {title_preview}"
        )

    print(f"\n{'='*100}")
    # Summary
    ok = [r for r in results if r.status == "ok"]
    hot = [r for r in ok if r.recency_score == "hot"]
    blocked = [r for r in results if r.scrape_score == "blocked"]
    dead = [r for r in results if r.recency_score == "dead"]
    print(f"\nSUMMARY: {len(results)} sources tested")
    print(f"  Active & recent (hot):   {len(hot)}")
    print(f"  Successfully scraped:    {len(ok)}")
    print(f"  Blocked:                 {len(blocked)}")
    print(f"  Dead/empty:              {len(dead)}")

    print(f"\nTOP SOURCES (hot + high AI relevance):")
    top = [r for r in results if r.recency_score in ("hot", "warm") and r.ai_relevance_pct >= 50]
    for r in sorted(top, key=lambda x: (-x.recent_items, -x.ai_relevance_pct))[:20]:
        print(f"  {r.name:<35}  14-day: {r.recent_items:<4}  AI%: {r.ai_relevance_pct}%")


def main():
    print(f"Testing {len(RSS_FEEDS)} RSS feeds (past 14 days cutoff: {CUTOFF.strftime('%Y-%m-%d')})")
    print("This may take a minute...\n")

    results = []
    for i, (name, url) in enumerate(RSS_FEEDS, 1):
        print(f"[{i:>3}/{len(RSS_FEEDS)}] {name:<40}", end=" ", flush=True)
        result = test_feed(name, url)
        results.append(result)
        status_icon = {
            "ok": "✓",
            "blocked": "✗ BLOCKED",
            "error": "✗ ERROR",
            "timeout": "✗ TIMEOUT",
            "parse_error": "~ EMPTY",
        }.get(result.status, "?")
        print(f"{status_icon}  |  14-day: {result.recent_items}  |  AI%: {result.ai_relevance_pct}%")
        time.sleep(0.3)  # polite delay

    print_results(results)

    # Save JSON report
    out_path = "/Users/alymoosa/Documents/A-Moosa-Dev/market-insights-agent:viz/MIA_all_things_exploration/scrape_test_results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
