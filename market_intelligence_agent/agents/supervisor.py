"""
Market Agent Supervisor ("Managing Director").
Orchestrates a full scrape-extract-signal pipeline run.
Can be triggered manually or by the scheduler.
"""
from datetime import datetime, timezone
from typing import Optional

from agents.sourcing import collect
from agents.extraction import extract
from agents.signal import process


def run_pipeline(player_key: Optional[str] = None) -> dict:
    """
    Execute one full ingestion cycle:
      1. Sourcing  — collect + tag raw articles
      2. Extraction — LLM-parse each article into a structured event
      3. Signal    — dedup and persist to ArangoDB

    Returns a run summary dict.
    """
    started_at = datetime.now(timezone.utc)
    print(f"\n[Supervisor] Pipeline run started at {started_at.isoformat()}")
    if player_key:
        print(f"[Supervisor] Scoped to player: {player_key}")

    stats = {
        "started_at": started_at.isoformat(),
        "player_key": player_key,
        "raw_articles": 0,
        "extracted": 0,
        "inserted": 0,
        "source_appended": 0,
        "hash_skipped": 0,
        "extraction_failed": 0,
        "errors": [],
    }

    # Step 1: Sourcing
    try:
        tagged_articles = collect(player_key)
        stats["raw_articles"] = len(tagged_articles)
    except Exception as e:
        stats["errors"].append(f"Sourcing failed: {e}")
        print(f"[Supervisor] Sourcing error: {e}")
        return _finish(stats, started_at)

    # Step 2 + 3: Extract then Signal, per article
    for article in tagged_articles:
        try:
            event = extract(article)
            if event is None:
                stats["extraction_failed"] += 1
                continue
            stats["extracted"] += 1

            result_key = process(event)
            if result_key is None:
                stats["hash_skipped"] += 1
            else:
                # Determine if it was an insert or source append
                # Signal logs this; we track via source_count heuristic
                stats["inserted"] += 1

        except Exception as e:
            stats["errors"].append(str(e)[:120])
            print(f"[Supervisor] Error processing article '{article.get('title', '')[:50]}': {e}")

    return _finish(stats, started_at)


def _finish(stats: dict, started_at: datetime) -> dict:
    finished_at = datetime.now(timezone.utc)
    duration = (finished_at - started_at).total_seconds()
    stats["finished_at"] = finished_at.isoformat()
    stats["duration_seconds"] = round(duration, 1)

    print(f"\n[Supervisor] Run complete in {duration:.1f}s")
    print(f"  Raw articles:       {stats['raw_articles']}")
    print(f"  Extracted:          {stats['extracted']}")
    print(f"  Inserted (new):     {stats['inserted']}")
    print(f"  Hash skipped:       {stats['hash_skipped']}")
    print(f"  Extraction failed:  {stats['extraction_failed']}")
    if stats["errors"]:
        print(f"  Errors:             {len(stats['errors'])}")

    return stats


if __name__ == "__main__":
    run_pipeline()
