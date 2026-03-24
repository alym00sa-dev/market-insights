"""
Pull latest 2-3 AI-relevant headlines from each working RSS source.
"""

import feedparser
import requests
import time
import json
from datetime import datetime, timezone, timedelta

CUTOFF = datetime.now(timezone.utc) - timedelta(days=30)  # broader window to get at least 2-3

AI_KEYWORDS = [
    "artificial intelligence", "machine learning", "deep learning", "neural network",
    "large language model", "llm", "gpt", "claude", "gemini", "openai", "anthropic",
    "transformer", "generative ai", "gen ai", "foundation model", "language model",
    "ai agent", "chatgpt", "mistral", "llama", "inference", "fine-tuning", "rag",
    "multimodal", "diffusion model", "embedding", "nlp", "computer vision",
    "reinforcement learning", "benchmark", "reasoning model", "o1", "o3",
    "model", "ai ", " ai", "nvidia", "microsoft ai", "google ai", "meta ai",
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FeedTester/1.0)"}

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
    ("Crunchbase News", "https://news.crunchbase.com/feed"),
    ("arXiv cs.CL", "https://arxiv.org/rss/cs.CL"),
    ("arXiv cs.LG", "https://arxiv.org/rss/cs.LG"),
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
    ("KDnuggets", "https://www.kdnuggets.com/feed"),
    ("LangChain Blog", "https://blog.langchain.dev/rss/"),
    ("Last Week in AI", "https://lastweekin.ai/feed"),
    ("Latent Space", "https://www.latent.space/feed"),
    ("ZDNET AI", "https://www.zdnet.com/topic/artificial-intelligence/rss.xml"),
    ("ML@CMU Blog", "https://blog.ml.cmu.edu/feed"),
    ("MarkTechPost", "https://www.marktechpost.com/feed"),
    ("MIT News ML", "https://news.mit.edu/topic/mitmachine-learning-rss.xml"),
    ("MIT Technology Review", "https://www.technologyreview.com/feed/"),
    ("New Scientist Tech", "https://www.newscientist.com/subject/technology/feed/"),
    ("NVIDIA Dev Blog", "https://developer.nvidia.com/blog/feed"),
    ("NY Times Tech", "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"),
    ("One Useful Thing", "https://www.oneusefulthing.org/feed"),
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
    ("Interconnects", "https://www.interconnects.ai/feed"),
    ("Unwind AI", "https://unwindai.substack.com/feed"),
    ("bdTechTalks", "https://bdtechtalks.com/feed/"),
    ("BAIR Blog", "https://bair.berkeley.edu/blog/feed.xml"),
    ("Databricks Blog", "https://www.databricks.com/feed"),
    ("Phys.org AI", "https://phys.org/rss-feed/technology-news/machine-learning-ai/"),
    ("TechXplore AI", "https://techmonitor.ai/feed"),
    ("WandB Blog", "https://wandb.ai/fully-connected/rss.xml"),
    ("Wolfram Blog", "https://blog.wolfram.com/feed/"),
]


def score_text(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in AI_KEYWORDS)


def parse_date(entry):
    for field in ("published_parsed", "updated_parsed"):
        t = getattr(entry, field, None)
        if t:
            try:
                return datetime(*t[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def format_date(dt):
    if dt:
        return dt.strftime("%Y-%m-%d")
    return "unknown date"


def get_latest(name, url, n=3):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12, allow_redirects=True)
        if resp.status_code >= 400:
            return name, None, f"HTTP {resp.status_code}"

        feed = feedparser.parse(resp.content)
        entries = feed.entries
        if not entries:
            return name, None, "No entries"

        # Collect AI-relevant items first, fall back to any items
        ai_items = []
        all_items = []
        for entry in entries[:30]:
            title = getattr(entry, "title", "") or ""
            link = getattr(entry, "link", "") or ""
            summary = getattr(entry, "summary", "") or ""
            dt = parse_date(entry)
            item = {
                "title": title.strip(),
                "url": link,
                "date": format_date(dt),
                "ai_relevant": score_text(f"{title} {summary}")
            }
            all_items.append(item)
            if item["ai_relevant"]:
                ai_items.append(item)

        # Prefer AI-relevant; fall back to all items
        picks = ai_items[:n] if len(ai_items) >= 2 else all_items[:n]
        return name, picks, None

    except requests.exceptions.Timeout:
        return name, None, "Timeout"
    except Exception as e:
        return name, None, str(e)[:80]


def main():
    results = {}
    print(f"Fetching latest AI news from {len(RSS_FEEDS)} sources...\n")

    for i, (name, url) in enumerate(RSS_FEEDS, 1):
        print(f"[{i:>2}/{len(RSS_FEEDS)}] {name}", flush=True)
        _, items, err = get_latest(name, url)
        if items:
            results[name] = items
        else:
            results[name] = [{"error": err}]
        time.sleep(0.25)

    # Print formatted report
    print("\n" + "="*90)
    print("LATEST AI NEWS BY SOURCE")
    print("="*90 + "\n")

    for name, items in results.items():
        print(f"── {name}")
        if len(items) == 1 and "error" in items[0]:
            print(f"   ✗ {items[0]['error']}")
        else:
            for item in items:
                ai_flag = "●" if item.get("ai_relevant") else "○"
                print(f"   {ai_flag} [{item['date']}] {item['title'][:90]}")
                if item.get("url"):
                    print(f"     {item['url'][:100]}")
        print()

    # Save JSON
    out = "/Users/alymoosa/Documents/A-Moosa-Dev/market-insights-agent:viz/MIA_all_things_exploration/latest_news.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to: {out}")


if __name__ == "__main__":
    main()
