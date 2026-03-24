"""
Source Scout Monitor

Continuous monitoring orchestrator for Source Scout.

Responsibilities:
1. Schedule checks based on frequency
2. Execute monitoring campaigns
3. Feed URLs to Content Harvester
4. Track performance and adjust priorities
5. Discover new sources via web search
6. Monitor RSS feeds for real-time updates
"""

import time
import feedparser
from typing import List, Set
from datetime import datetime

from .monitored_source import MonitoredSource, HYPERSCALER_SOURCES, CROSS_PROVIDER_SOURCES
from .link_extractor import LinkExtractor
from ..content_harvester import ContentHarvester
from ...storage import EventDatabase


class SourceScoutMonitor:
    """
    Continuous monitoring orchestrator for Source Scout.

    Runs in "always on" mode - continuously checks sources and discovers new ones.
    """

    def __init__(
        self,
        source_scout,  # Type: SourceScout (avoid circular import)
        content_harvester: ContentHarvester,
        database: EventDatabase
    ):
        """
        Initialize monitor.

        Args:
            source_scout: Source Scout for discovery
            content_harvester: Content Harvester for processing
            database: Database for tracking
        """
        self.scout = source_scout
        self.harvester = content_harvester
        self.db = database
        self.sources: List[MonitoredSource] = []
        self.seen_rss_entries: Set[str] = set()
        self.link_extractor = LinkExtractor()  # For blog homepages without RSS

        # Load initial sources
        self._load_all_sources()

    def _load_all_sources(self) -> None:
        """Load all monitored sources (curated + discovered)."""
        print("Loading monitored sources...")

        # Load curated hyperscaler sources
        for provider, provider_sources in HYPERSCALER_SOURCES.items():
            self.sources.extend(provider_sources)
            print(f"  Loaded {len(provider_sources)} sources for {provider}")

        # Load cross-provider sources
        self.sources.extend(CROSS_PROVIDER_SOURCES)
        print(f"  Loaded {len(CROSS_PROVIDER_SOURCES)} cross-provider sources")

        # TODO: Load previously discovered sources from DB
        # discovered = self._load_discovered_sources_from_db()
        # self.sources.extend(discovered)

        print(f"Total sources: {len(self.sources)}")

    def run_continuous_monitoring(self, check_interval_minutes: int = 60):
        """
        Main monitoring loop - runs indefinitely.

        Strategy:
        - Every check_interval: Check sources due for update
        - Every 6 hours: RSS feed refresh
        - Every 24 hours: Discovery campaign (find new sources)

        Args:
            check_interval_minutes: Minutes between check cycles (default 60)
        """
        print("Source Scout Monitor started (continuous mode)")
        print(f"Check interval: {check_interval_minutes} minutes")

        last_discovery_campaign = datetime.now()
        last_rss_check = datetime.now()

        while True:
            try:
                cycle_start = datetime.now()
                print(f"\n[{cycle_start.strftime('%Y-%m-%d %H:%M:%S')}] Starting check cycle...")

                # 1. Check sources due for update
                due_sources = self._get_sources_due_for_check()
                print(f"Checking {len(due_sources)} sources...")

                for source in due_sources:
                    self._check_and_harvest(source)
                    time.sleep(2)  # Rate limiting between checks

                # 2. RSS feed refresh (every 6 hours)
                hours_since_rss = (datetime.now() - last_rss_check).total_seconds() / 3600
                if hours_since_rss >= 6:
                    print("Refreshing RSS feeds...")
                    self._monitor_rss_feeds()
                    last_rss_check = datetime.now()

                # 3. Discovery campaign (every 24 hours)
                hours_since_discovery = (datetime.now() - last_discovery_campaign).total_seconds() / 3600
                if hours_since_discovery >= 24:
                    print("Running discovery campaign...")
                    self._run_discovery_campaign()
                    last_discovery_campaign = datetime.now()

                # 4. Sleep until next check cycle
                cycle_duration = (datetime.now() - cycle_start).total_seconds()
                sleep_time = max(0, (check_interval_minutes * 60) - cycle_duration)

                if sleep_time > 0:
                    print(f"Sleeping for {sleep_time / 60:.1f} minutes...")
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\nMonitoring stopped by user")
                break
            except Exception as e:
                print(f"Monitoring error: {e}")
                print("Sleeping for 5 minutes before retry...")
                time.sleep(300)

    def _get_sources_due_for_check(self) -> List[MonitoredSource]:
        """
        Return sources that need to be checked now.

        Based on check_frequency and last_checked_at.
        """
        due_sources = []
        now = datetime.now()

        frequency_to_hours = {
            "hourly": 1,
            "daily": 24,
            "weekly": 168,
            "monthly": 720
        }

        for source in self.sources:
            if source.last_checked_at is None:
                # Never checked - add it
                due_sources.append(source)
                continue

            # Check if enough time has passed based on frequency
            hours_since_check = (now - source.last_checked_at).total_seconds() / 3600
            required_hours = frequency_to_hours.get(source.check_frequency, 24)

            if hours_since_check >= required_hours:
                due_sources.append(source)

        # Sort by priority (critical first)
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        due_sources.sort(key=lambda s: priority_order.get(s.priority, 4))

        return due_sources

    def _check_and_harvest(self, source: MonitoredSource) -> None:
        """
        Check a source and pass to Content Harvester if updated.

        Strategy:
        - If needs_link_extraction: Extract article links, then harvest each
        - If rss_feed: Already handled by _monitor_rss_feeds
        - Otherwise: Harvest the URL directly
        """
        print(f"  Checking: {source.url} ({source.provider})")

        try:
            # Update tracking
            source.last_checked_at = datetime.now()
            source.total_checks += 1

            # Strategy 1: Link extraction for blog homepages
            if source.needs_link_extraction:
                article_links = self.link_extractor.extract_article_links(
                    homepage_url=source.url,
                    limit=5  # Limit to 5 most recent articles
                )

                if article_links:
                    print(f"    [EXTRACTED] Found {len(article_links)} article links")

                    for article_url in article_links:
                        try:
                            content = self.harvester.harvest(
                                url=article_url,
                                provider=source.provider,
                                source_type="article"
                            )

                            if content:
                                print(f"      [SUCCESS] {article_url} (category: {content.content_category})")
                                source.successful_checks += 1
                                source.last_updated_at = datetime.now()
                        except Exception as e:
                            print(f"      [ERROR] Failed to harvest {article_url}: {e}")

                    # Update reliability
                    source.reliability_score = source.successful_checks / source.total_checks

                else:
                    print(f"    [SKIP] No article links found")

            # Strategy 2: Direct harvest (changelogs, direct URLs)
            elif source.source_type not in ['rss_feed']:  # RSS handled separately
                content = self.harvester.harvest(
                    url=source.url,
                    provider=source.provider,
                    source_type=source.source_type
                )

                if content:
                    print(f"    [SUCCESS] Non-noise content found (category: {content.content_category})")
                    source.successful_checks += 1
                    source.last_updated_at = datetime.now()
                    source.reliability_score = source.successful_checks / source.total_checks
                else:
                    print(f"    [SKIP] No new content (filtered as noise or unchanged)")

            # Update source tracking in DB
            self._update_source_tracking(source)

        except Exception as e:
            print(f"    [ERROR] Failed to check source: {e}")
            source.last_checked_at = datetime.now()
            source.total_checks += 1

    def _update_source_tracking(self, source: MonitoredSource) -> None:
        """
        Update source tracking in database.

        Stores monitoring metadata for performance analysis.
        """
        # TODO: Implement database storage for MonitoredSource tracking
        # For MVP: Skip database persistence
        pass

    def _run_discovery_campaign(self) -> None:
        """
        Periodic discovery campaign to find new sources.

        Strategy:
        - Search for recent announcements from each hyperscaler
        - Validate discovered URLs
        - Add high-quality sources to monitoring pool
        """
        print("Discovery Campaign:")

        for provider in ["OpenAI", "Anthropic", "Google", "Microsoft", "Meta"]:
            print(f"  Discovering sources for {provider}...")

            try:
                # Use Source Scout's automated discovery
                discovered = self.scout.discover_sources(
                    provider=provider,
                    mode="automated",
                    use_web_search=True,
                    limit=5
                )

                # Add new sources to monitoring pool
                for candidate in discovered:
                    if not self._source_already_monitored(candidate.url):
                        new_source = MonitoredSource(
                            url=candidate.url,
                            provider=provider,
                            source_type=candidate.source_type,
                            priority=candidate.priority,
                            check_frequency="weekly",  # Conservative for discovered sources
                            discovered_via="web_search"
                        )
                        self.sources.append(new_source)
                        print(f"    [NEW] Added: {candidate.url}")

            except Exception as e:
                print(f"    [ERROR] Discovery failed for {provider}: {e}")

    def _monitor_rss_feeds(self) -> None:
        """
        Monitor RSS feeds for new entries.

        Strategy:
        - Check RSS feeds for new entries
        - Only process new entries (track GUIDs/links)
        - Pass new entries directly to Content Harvester
        """
        rss_sources = [s for s in self.sources if s.source_type == "rss_feed"]

        if not rss_sources:
            print("  No RSS feeds configured")
            return

        print(f"  Checking {len(rss_sources)} RSS feeds...")

        for source in rss_sources:
            try:
                feed = feedparser.parse(source.url)

                if not feed.entries:
                    print(f"    [SKIP] No entries in feed: {source.url}")
                    continue

                new_entries = 0
                for entry in feed.entries:
                    entry_id = entry.get('id') or entry.get('link')

                    if not entry_id:
                        continue

                    # Check if we've seen this entry before
                    if entry_id not in self.seen_rss_entries:
                        print(f"    [NEW] RSS entry: {entry.get('title', 'Untitled')}")

                        # Pass to Content Harvester
                        try:
                            content = self.harvester.harvest(
                                url=entry.link,
                                provider=source.provider,
                                source_type="rss_entry"
                            )

                            if content:
                                print(f"      [SUCCESS] Non-noise content (category: {content.content_category})")
                                new_entries += 1
                        except Exception as e:
                            print(f"      [ERROR] Failed to harvest: {e}")

                        # Mark as seen
                        self.seen_rss_entries.add(entry_id)

                if new_entries > 0:
                    print(f"    Found {new_entries} significant entries")

            except Exception as e:
                print(f"    [ERROR] RSS feed error for {source.url}: {e}")

    def _source_already_monitored(self, url: str) -> bool:
        """Check if a URL is already in the monitoring pool."""
        return any(source.url == url for source in self.sources)

    def _load_discovered_sources_from_db(self) -> List[MonitoredSource]:
        """
        Load previously discovered sources from database.

        TODO: Implement database query for MonitoredSource records.
        """
        return []

    def get_monitoring_stats(self) -> dict:
        """
        Get monitoring statistics.

        Returns summary of monitoring performance.
        """
        total_sources = len(self.sources)
        checked_sources = sum(1 for s in self.sources if s.last_checked_at is not None)
        active_sources = sum(1 for s in self.sources if s.successful_checks > 0)

        return {
            "total_sources": total_sources,
            "checked_sources": checked_sources,
            "active_sources": active_sources,
            "avg_reliability": sum(s.reliability_score for s in self.sources) / total_sources if total_sources > 0 else 0,
            "by_provider": self._get_provider_stats(),
            "by_priority": self._get_priority_stats()
        }

    def _get_provider_stats(self) -> dict:
        """Get statistics by provider."""
        providers = {}
        for source in self.sources:
            if source.provider not in providers:
                providers[source.provider] = {"total": 0, "active": 0}
            providers[source.provider]["total"] += 1
            if source.successful_checks > 0:
                providers[source.provider]["active"] += 1
        return providers

    def _get_priority_stats(self) -> dict:
        """Get statistics by priority."""
        priorities = {}
        for source in self.sources:
            if source.priority not in priorities:
                priorities[source.priority] = {"total": 0, "active": 0}
            priorities[source.priority]["total"] += 1
            if source.successful_checks > 0:
                priorities[source.priority]["active"] += 1
        return priorities
