"""
Pipeline scheduler.
Runs the ingestion pipeline on a configurable interval using APScheduler.
Exposes start(), trigger_now(), and get_status() for the dashboard.
"""
import threading
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from config.settings import settings
from agents.supervisor import run_pipeline

_scheduler: BackgroundScheduler | None = None
_lock = threading.Lock()

_state = {
    "is_running": False,
    "last_run_time": None,
    "last_run_stats": None,
    "next_run_time": None,
}


def _run_job():
    with _lock:
        if _state["is_running"]:
            print("[Scheduler] Pipeline already running — skipping trigger.")
            return
        _state["is_running"] = True

    try:
        print(f"[Scheduler] Starting pipeline run at {datetime.now(timezone.utc).isoformat()}")
        stats = run_pipeline()
        _state["last_run_stats"] = stats
        _state["last_run_time"] = stats.get("finished_at")
    except Exception as e:
        print(f"[Scheduler] Pipeline error: {e}")
        _state["last_run_stats"] = {"error": str(e)}
    finally:
        _state["is_running"] = False

    # Refresh next_run_time after job completes
    _refresh_next_run()


def _refresh_next_run():
    if _scheduler and _scheduler.running:
        jobs = _scheduler.get_jobs()
        if jobs and jobs[0].next_run_time:
            _state["next_run_time"] = jobs[0].next_run_time.isoformat()


def start():
    """Start the background scheduler. Safe to call multiple times — only starts once."""
    global _scheduler
    if _scheduler and _scheduler.running:
        return

    _scheduler = BackgroundScheduler(timezone="UTC")
    _scheduler.add_job(
        _run_job,
        trigger="interval",
        hours=settings.SCRAPE_INTERVAL_HOURS,
        id="pipeline",
        max_instances=1,
    )
    _scheduler.start()
    _refresh_next_run()
    print(f"[Scheduler] Started. Interval: every {settings.SCRAPE_INTERVAL_HOURS}h. Next run: {_state['next_run_time']}")


def trigger_now():
    """Trigger a pipeline run immediately in a background thread."""
    thread = threading.Thread(target=_run_job, daemon=True)
    thread.start()


def get_status() -> dict:
    """Return current scheduler state for display in the dashboard."""
    _refresh_next_run()
    return dict(_state)
