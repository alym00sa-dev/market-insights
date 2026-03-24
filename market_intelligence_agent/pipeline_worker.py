"""
Render background worker — continuous ingestion pipeline.

Runs the full scrape → extract → signal pipeline on a configurable interval.
Deploy as a Background Worker service on Render.

Run locally:
  PYTHONPATH=. python pipeline_worker.py
"""
import sys
import logging
from pathlib import Path

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apscheduler.schedulers.blocking import BlockingScheduler
from config.settings import settings
from agents.supervisor import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def job():
    log.info("Pipeline run starting...")
    try:
        stats = run_pipeline()
        log.info(
            f"Pipeline complete — raw: {stats['raw_articles']}, "
            f"inserted: {stats['inserted']}, skipped: {stats['hash_skipped']}, "
            f"duration: {stats['duration_seconds']}s"
        )
    except Exception as e:
        log.error(f"Pipeline failed: {e}")


if __name__ == "__main__":
    interval_hours = settings.SCRAPE_INTERVAL_HOURS
    log.info(f"Starting pipeline worker. Interval: every {interval_hours}h.")

    # Run once immediately on startup so the DB is fresh on first deploy
    log.info("Running initial pipeline on startup...")
    job()

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(job, trigger="interval", hours=interval_hours, max_instances=1)

    log.info(f"Scheduler started. Next run in {interval_hours}h.")
    scheduler.start()
