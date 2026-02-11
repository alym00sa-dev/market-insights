"""
Live Source Ingestion - Part 2

Pulls from live sources (blogs, docs) with smart deduplication.

Key features:
1. Tavily-based source discovery with RSS validation
2. Semantic deduplication (find similar events, not just exact IDs)
3. Event merging/enrichment (keep most complete info)
4. Respects existing data (doesn't overwrite good data with worse)
"""

import os
import feedparser
import requests
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv

from ..models import MarketSignalEvent
from ..storage import EventDatabase, EventVectorStore
from ..llm import LLMProvider

load_dotenv()


class LiveIngestionManager:
    """
    Manages ingestion from live sources with intelligent deduplication.

    Deduplication strategy:
    1. Semantic similarity check (vector store query)
    2. Date proximity check (within 7 days)
    3. Provider matching
    4. If match found: merge fields (keep most complete)
    5. If no match: create new event
    """

    def __init__(
        self,
        database: EventDatabase,
        vector_store: EventVectorStore,
        llm_provider: LLMProvider,
        config: Dict[str, Any]
    ):
        """
        Initialize live ingestion manager.

        Args:
            database: Event database
            vector_store: Vector store
            llm_provider: LLM provider
            config: Configuration dict
        """
        self.db = database
        self.vector_store = vector_store
        self.llm = llm_provider
        self.config = config

        # Deduplication thresholds
        self.similarity_threshold = 0.85  # Cosine similarity threshold
        self.date_proximity_days = 7  # Consider events within 7 days as potential duplicates

        # Tavily client (for source discovery)
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.tavily_client = None
        if self.tavily_api_key:
            try:
                from tavily import TavilyClient
                self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
            except ImportError:
                print("Warning: tavily-python not installed. Source discovery limited.")

    # ========================================================================
    # SOURCE DISCOVERY WITH VALIDATION
    # ========================================================================

    def check_rss_freshness(self, rss_url: str) -> Optional[datetime]:
        """
        Check when RSS feed was last updated.

        Returns:
            Datetime of most recent entry, or None if failed
        """
        try:
            feed = feedparser.parse(rss_url)
            if feed.entries:
                # Get most recent entry
                latest_entry = feed.entries[0]

                # Try different date fields
                if hasattr(latest_entry, 'published_parsed') and latest_entry.published_parsed:
                    return datetime(*latest_entry.published_parsed[:6])
                elif hasattr(latest_entry, 'updated_parsed') and latest_entry.updated_parsed:
                    return datetime(*latest_entry.updated_parsed[:6])

            return None
        except Exception as e:
            print(f"   Warning: Failed to check RSS freshness for {rss_url}: {e}")
            return None

    def run_multi_query_discovery(
        self,
        provider: str,
        days_back: int = 30,
        use_proxy: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run multiple Tavily queries for a provider and aggregate results.

        This runs 5+ different queries per provider (model releases, partnerships,
        safety, infrastructure, etc.) and aggregates the results.

        Args:
            provider: Provider name (e.g., "OpenAI")
            days_back: How many days back to search
            use_proxy: If True, search proxy domains (news sites) instead of primary domain

        Returns:
            Dict mapping URL -> metadata with aggregated info:
                {
                    'url': str,
                    'title': str,
                    'queries_found_by': List[str],  # Which queries found this
                    'query_categories': List[str],  # Categories of those queries
                    'max_tavily_score': float,      # Best Tavily relevance score
                    'published_date': Optional[str]
                }
        """
        if not self.tavily_client:
            print(f"   ⚠️  Tavily not available for {provider}, skipping search")
            return {}

        # Get search config for this provider
        search_queries = self.config['agents']['source_scout'].get('search_queries', {})
        provider_key = provider.lower().replace(' ', '')

        if provider_key not in search_queries:
            print(f"   ⚠️  No search queries configured for {provider}")
            return {}

        query_config = search_queries[provider_key]
        primary_domain = query_config['domain']
        proxy_domains = query_config.get('proxy_domains', [])
        queries = query_config['queries']

        # Decide which domains to search
        if use_proxy and proxy_domains:
            domains = proxy_domains
            print(f"   Using proxy domains: {', '.join(domains)}")
        else:
            domains = [primary_domain]
            if proxy_domains:
                print(f"   Primary domain: {primary_domain} (proxies available if needed)")

        print(f"   Running {len(queries)} Tavily queries for {provider}...")

        # Aggregate results by URL
        url_aggregator = {}

        for query_item in queries:
            query_text = query_item['query']
            category = query_item['category']
            weight = query_item.get('weight', 1.0)

            print(f"     → '{query_text}' ({category})")

            try:
                # Search with Tavily
                results = self.tavily_client.search(
                    query=query_text,
                    search_depth="advanced",
                    max_results=5,  # 5 per query
                    include_domains=domains,
                    days=days_back
                )

                found_count = 0
                for result in results.get('results', []):
                    url = result['url']
                    tavily_score = result.get('score', 0) * weight

                    if url not in url_aggregator:
                        # First time seeing this URL
                        url_aggregator[url] = {
                            'url': url,
                            'title': result.get('title', ''),
                            'queries_found_by': [query_text],
                            'query_categories': [category],
                            'max_tavily_score': tavily_score,
                            'published_date': result.get('published_date', None),
                            'is_proxy': use_proxy  # Track if from proxy domain
                        }
                        found_count += 1
                    else:
                        # URL already found by another query - aggregate
                        url_aggregator[url]['queries_found_by'].append(query_text)
                        url_aggregator[url]['query_categories'].append(category)
                        url_aggregator[url]['max_tavily_score'] = max(
                            url_aggregator[url]['max_tavily_score'],
                            tavily_score
                        )

                print(f"       ✓ {found_count} new URLs")

            except Exception as e:
                print(f"       ✗ Query failed: {e}")

        print(f"   ✓ Aggregated {len(url_aggregator)} unique URLs from {len(queries)} queries")

        # Show multi-query hits
        multi_query_urls = [url for url, data in url_aggregator.items()
                           if len(data['queries_found_by']) > 1]
        if multi_query_urls:
            print(f"   🎯 {len(multi_query_urls)} URLs found by multiple queries (high confidence)")

        return url_aggregator

    def score_discovered_url(
        self,
        url: str,
        provider: str,
        aggregated_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Score a discovered URL using Harvester and calculate combined score.

        Scoring components:
        1. Tavily relevance score (0-1) * 3 = 0-3 points
        2. Recency score (0-30 days) = 0-3 points
        3. Harvester significance (0-10) * 0.3 = 0-3 points
        4. Multi-query bonus (+1 if found by 2+ queries)

        Total: 0-10 points

        Args:
            url: URL to score
            provider: Provider name
            aggregated_data: Aggregated metadata from Tavily

        Returns:
            Dict with scoring details, or None if fetching failed
        """
        # Import here to avoid circular dependency
        from ..agents import ContentHarvester

        print(f"     Scoring: {url[:80]}...")

        # Component 1: Tavily relevance (0-3 points)
        tavily_score = aggregated_data['max_tavily_score'] * 3

        # Component 2: Recency (0-3 points)
        recency_score = 0
        if aggregated_data.get('published_date'):
            try:
                pub_date = datetime.fromisoformat(aggregated_data['published_date'].replace('Z', '+00:00'))
                days_old = (datetime.now(pub_date.tzinfo) - pub_date).days
                # Linear decay: 0 days = 3 points, 30 days = 0 points
                recency_score = max(0, 3 * (1 - days_old / 30))
            except:
                recency_score = 1.5  # Default if can't parse

        # Component 3: Harvester significance (0-3 points)
        # Use the existing harvester to score content
        harvester = ContentHarvester(self.llm, self.db, self.config)

        try:
            # Fetch and analyze with Harvester
            harvested = harvester.harvest(url, provider, 'official_blog')

            if not harvested:
                print(f"       ✗ Failed to harvest content")
                return None

            # Harvester gives 0-10, we scale to 0-3
            significance_score = harvested.significance_score * 0.3

        except Exception as e:
            print(f"       ✗ Harvest error: {e}")
            return None

        # Component 4: Multi-query bonus (0-1 points)
        num_queries = len(aggregated_data['queries_found_by'])
        multi_query_bonus = self.config['agents']['source_scout']['scoring'].get('multi_query_bonus', 1.0) if num_queries > 1 else 0

        # Calculate combined score
        combined_score = tavily_score + recency_score + significance_score + multi_query_bonus

        print(f"       ✓ Score: {combined_score:.1f}/10 (tavily:{tavily_score:.1f} + recency:{recency_score:.1f} + sig:{significance_score:.1f} + bonus:{multi_query_bonus:.1f})")
        print(f"         Queries: {num_queries} | Categories: {', '.join(set(aggregated_data['query_categories']))}")

        return {
            'url': url,
            'provider': provider,
            'title': aggregated_data['title'],
            'combined_score': combined_score,
            'tavily_score': tavily_score,
            'recency_score': recency_score,
            'significance_score': significance_score,
            'multi_query_bonus': multi_query_bonus,
            'num_queries': num_queries,
            'categories': aggregated_data['query_categories'],
            'harvested_content': harvested
        }

    def check_if_already_covered(self, url: str, harvested_content: Any) -> Optional[str]:
        """
        Check if URL content is already covered by existing events in DB.

        Args:
            url: URL to check
            harvested_content: HarvestedContent from Harvester

        Returns:
            Event ID if already covered, None if new content
        """
        # Use the harvested content's filtered text for semantic search
        query_text = f"{harvested_content.content_type} {harvested_content.filtered_content[:500]}"

        try:
            # Search for similar events
            results = self.db.semantic_search_with_filters(
                query_text=query_text,
                vector_store=self.vector_store,
                provider=harvested_content.provider,
                limit=3
            )

            # Check if highly similar event exists
            for result in results:
                similarity = result.get('similarity_score', 0)
                if similarity >= 0.90:  # Very high similarity
                    print(f"       ℹ️  Already covered by {result['event_id']} (similarity: {similarity:.2f})")
                    return result['event_id']

            return None

        except Exception as e:
            print(f"       Warning: Could not check existing coverage: {e}")
            return None

    def discover_and_validate_sources(self, provider: str) -> List[Dict[str, Any]]:
        """
        Discover sources using multi-query Tavily search, score with Harvester,
        and validate against existing DB.

        Multi-stage pipeline:
        1. Check RSS feed freshness (baseline timestamp)
        2. Run multiple Tavily queries (5+ per provider)
        3. Aggregate URLs (deduplicate, track which queries found them)
        4. Score each URL (Tavily relevance + recency + Harvester significance + multi-query bonus)
        5. Compare against existing DB (skip if already covered)
        6. Return high-scoring, validated URLs

        Args:
            provider: Provider name

        Returns:
            List of scored and validated sources ready for extraction:
                [
                    {
                        'url': str,
                        'combined_score': float,
                        'harvested_content': HarvestedContent,
                        ...
                    }
                ]
        """
        print(f"\n{'=' * 80}")
        print(f"DISCOVERING SOURCES: {provider}")
        print(f"{'=' * 80}")

        # Step 1: Check RSS feed freshness (optional validation)
        rss_feeds = self.config['agents']['source_scout'].get('rss_feeds', {})
        provider_key = provider.lower().replace(' ', '')

        rss_last_updated = None
        if provider_key in rss_feeds:
            rss_url = rss_feeds[provider_key]
            print(f"\n1. 📡 Checking RSS feed...")
            rss_last_updated = self.check_rss_freshness(rss_url)

            if rss_last_updated:
                print(f"   ✓ RSS last updated: {rss_last_updated.strftime('%Y-%m-%d %H:%M')}")
            else:
                print(f"   ⚠️  Could not determine RSS freshness")

        # Step 2: Run multiple Tavily queries and aggregate
        print(f"\n2. 🔍 Running multi-query Tavily discovery...")
        aggregated_urls = self.run_multi_query_discovery(provider, days_back=30)

        if not aggregated_urls:
            print(f"\n   ⚠️  No sources discovered for {provider}")
            return []

        # Step 3: Score each URL with Harvester
        print(f"\n3. 📊 Scoring {len(aggregated_urls)} discovered URLs...")
        scored_urls = []

        for url, aggregated_data in aggregated_urls.items():
            score_data = self.score_discovered_url(url, provider, aggregated_data)
            if score_data:
                scored_urls.append(score_data)

        if not scored_urls:
            print(f"\n   ⚠️  No URLs passed scoring")
            return []

        # Sort by combined score (highest first)
        scored_urls.sort(key=lambda x: x['combined_score'], reverse=True)

        # Step 4: Filter by score threshold
        min_score = self.config['agents']['source_scout']['scoring'].get('min_combined_score', 5.0)
        high_priority_threshold = self.config['agents']['source_scout']['scoring'].get('high_priority_threshold', 7.0)

        print(f"\n4. ✅ Filtering by score threshold (min: {min_score}/10)...")

        filtered_urls = []
        for score_data in scored_urls:
            score = score_data['combined_score']

            if score >= high_priority_threshold:
                print(f"   🔥 HIGH PRIORITY ({score:.1f}/10): {score_data['title'][:60]}")
                filtered_urls.append(score_data)
            elif score >= min_score:
                print(f"   ✓ PASS ({score:.1f}/10): {score_data['title'][:60]}")
                filtered_urls.append(score_data)
            else:
                print(f"   ✗ SKIP ({score:.1f}/10): {score_data['title'][:60]}")

        # Step 5: Check against existing DB
        print(f"\n5. 🔍 Checking against existing events...")

        validated_sources = []
        for score_data in filtered_urls:
            existing_event_id = self.check_if_already_covered(
                score_data['url'],
                score_data['harvested_content']
            )

            if existing_event_id:
                print(f"   ⊙ SKIP (already covered): {score_data['title'][:60]}")
            else:
                print(f"   ✓ NEW: {score_data['title'][:60]}")
                validated_sources.append(score_data)

        # Summary
        print(f"\n{'=' * 80}")
        print(f"DISCOVERY SUMMARY: {provider}")
        print(f"{'=' * 80}")
        print(f"  Total URLs discovered: {len(aggregated_urls)}")
        print(f"  Passed scoring: {len(filtered_urls)}")
        print(f"  New content: {len(validated_sources)}")
        print(f"  Already covered: {len(filtered_urls) - len(validated_sources)}")

        return validated_sources

    # ========================================================================
    # MAIN INGESTION
    # ========================================================================

    def ingest_from_sources(
        self,
        sources: Optional[List[str]] = None,
        providers: Optional[List[str]] = None,
        use_ensemble: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest from multiple sources with deduplication.

        Args:
            sources: List of source URLs (if provided, skips discovery)
            providers: List of providers to discover sources for (default: all configured)
            use_ensemble: Whether to use ensemble extraction

        Returns:
            Dict with:
                - total_fetched: Number of events fetched
                - new_events: Number of new events created
                - updated_events: Number of events updated
                - skipped_duplicates: Number of duplicates skipped
                - errors: List of errors
        """
        print("=" * 80)
        print("LIVE SOURCE INGESTION - Tavily Discovery + RSS Validation")
        print("=" * 80)

        # If sources not provided, discover them
        if sources is None:
            if providers is None:
                # Default to all configured providers
                providers = ['OpenAI', 'Anthropic', 'Google', 'Microsoft', 'Meta']

            print(f"\n{'=' * 80}")
            print(f"PHASE 1: MULTI-QUERY DISCOVERY & SCORING")
            print(f"{'=' * 80}")
            print(f"\nProviders: {', '.join(providers)}")

            scored_sources = []
            for provider in providers:
                discovered = self.discover_and_validate_sources(provider)
                scored_sources.extend(discovered)

            print(f"\n✓ Total sources validated: {len(scored_sources)}")

            # Convert scored sources to the format expected by ingestion
            # (We already have harvested content, so we skip the harvest step)
            sources = scored_sources
        else:
            print(f"\n1. Using provided sources: {len(sources)}")
            # If sources provided directly, they're just URLs
            # We'll need to convert them to the right format

        if not sources:
            print("\n⚠️  No sources to ingest")
            return {
                'total_fetched': 0,
                'new_events': 0,
                'updated_events': 0,
                'skipped_duplicates': 0,
                'errors': ['No sources discovered']
            }

        results = {
            'total_fetched': 0,
            'new_events': 0,
            'updated_events': 0,
            'skipped_duplicates': 0,
            'errors': []
        }

        print(f"\n{'=' * 80}")
        print(f"PHASE 2: SIGNAL EXTRACTION")
        print(f"{'=' * 80}")

        # Process each scored source
        for i, source_data in enumerate(sources, 1):
            print(f"\n{'-' * 80}")

            # Check if this is a scored source (from discovery) or raw URL
            if isinstance(source_data, dict) and 'harvested_content' in source_data:
                print(f"SOURCE {i}/{len(sources)} (Score: {source_data.get('combined_score', 0):.1f}/10)")
                print(f"{'-' * 80}")
                # Already scored and harvested - go directly to extraction
                source_url = source_data['url']
                provider = source_data['provider']
                harvested = source_data['harvested_content']

                print(f"URL: {source_url}")
                print(f"Provider: {provider}")
                print(f"Categories: {', '.join(set(source_data['categories']))}")
                print(f"\n✓ Content already harvested (skipping harvest step)")

                # Extract signal using the extractor
                from ..agents import SignalExtractor, EnsembleSignalExtractor

                try:
                    extractor = EnsembleSignalExtractor(self.llm, self.db, self.config) if use_ensemble else SignalExtractor(self.llm, self.db, self.config)

                    print(f"Extracting signal...")
                    event = extractor.extract(
                        content=harvested.filtered_content,
                        provider=provider,
                        source_url=source_url,
                        source_type=harvested.source_type,
                        published_at=harvested.published_at,
                        metadata=harvested.metadata
                    )

                    if event:
                        print(f"✓ Extracted event: {event.event_id}")

                        # Process with deduplication
                        dup_result = self._check_and_merge_duplicate(event)

                        if dup_result['action'] == 'new':
                            results['new_events'] += 1
                        elif dup_result['action'] == 'updated':
                            results['updated_events'] += 1
                        elif dup_result['action'] == 'skipped':
                            results['skipped_duplicates'] += 1

                        results['total_fetched'] += 1
                    else:
                        print(f"✗ Extraction failed")

                except Exception as e:
                    error_msg = f"Error extracting from {source_url}: {str(e)}"
                    print(f"✗ {error_msg}")
                    results['errors'].append(error_msg)

            else:
                # Raw URL - use full workflow
                source_url = source_data if isinstance(source_data, str) else source_data.get('url')
                print(f"SOURCE {i}/{len(sources)}")
                print(f"{'-' * 80}")
                print(f"URL: {source_url}")

                try:
                    provider = self._infer_provider_from_url(source_url)
                    print(f"Provider: {provider}")

                    # Ingest from source using workflow
                    from ..workflows import ingest_from_url

                    print(f"\nIngesting from source...")
                    ingestion_result = ingest_from_url(
                        url=source_url,
                        provider=provider,
                        source_type='official_blog',
                        llm_provider=self.llm,
                        database=self.db,
                        vector_store=self.vector_store,
                        config=self.config,
                        use_ensemble=use_ensemble
                    )

                    # Check result
                    if ingestion_result.get('status') == 'completed':
                        fetched_events = ingestion_result.get('events', [])
                        print(f"✓ Fetched {len(fetched_events)} events")

                        # Process each event with deduplication
                        for event in fetched_events:
                            dup_result = self._check_and_merge_duplicate(event)

                            if dup_result['action'] == 'new':
                                results['new_events'] += 1
                            elif dup_result['action'] == 'updated':
                                results['updated_events'] += 1
                            elif dup_result['action'] == 'skipped':
                                results['skipped_duplicates'] += 1

                        results['total_fetched'] += len(fetched_events)

                    else:
                        error_msg = f"Ingestion failed: {ingestion_result.get('errors', [])}"
                        print(f"✗ {error_msg}")
                        results['errors'].append(error_msg)

                except Exception as e:
                    error_msg = f"Error processing {source_url}: {str(e)}"
                    print(f"✗ {error_msg}")
                    results['errors'].append(error_msg)

        # Summary
        print("\n" + "=" * 80)
        print("INGESTION COMPLETE")
        print("=" * 80)
        print(f"\nTotal fetched: {results['total_fetched']}")
        print(f"New events: {results['new_events']}")
        print(f"Updated events: {results['updated_events']}")
        print(f"Skipped duplicates: {results['skipped_duplicates']}")
        if results['errors']:
            print(f"\nErrors: {len(results['errors'])}")

        return results

    # ========================================================================
    # DEDUPLICATION LOGIC
    # ========================================================================

    def _check_and_merge_duplicate(
        self,
        new_event: MarketSignalEvent
    ) -> Dict[str, Any]:
        """
        Check if event is duplicate and merge if needed.

        Returns:
            Dict with:
                - action: 'new', 'updated', or 'skipped'
                - event_id: Event ID (existing or new)
                - reason: Explanation of action taken
        """
        # Find potential duplicates using semantic search
        duplicates = self._find_potential_duplicates(new_event)

        if not duplicates:
            # No duplicates - store as new
            try:
                self.db.create_event(new_event)
                self.vector_store.add_event(new_event)
                print(f"   ✓ New event: {new_event.event_id}")
                return {
                    'action': 'new',
                    'event_id': new_event.event_id,
                    'reason': 'No duplicates found'
                }
            except Exception as e:
                print(f"   ✗ Failed to store: {e}")
                return {
                    'action': 'skipped',
                    'event_id': new_event.event_id,
                    'reason': f'Storage error: {str(e)}'
                }

        # Found duplicates - merge with best match
        best_match = duplicates[0]
        existing_event = self.db.get_event(best_match['event_id'])

        if not existing_event:
            # Existing event gone - store as new
            self.db.create_event(new_event)
            self.vector_store.add_event(new_event)
            return {
                'action': 'new',
                'event_id': new_event.event_id,
                'reason': 'Duplicate event not found in DB'
            }

        # Merge events
        merged_event = self._merge_events(existing_event, new_event)

        # Check if anything actually changed
        if self._events_are_identical(existing_event, merged_event):
            print(f"   ⊙ Duplicate (no new info): {existing_event.event_id}")
            return {
                'action': 'skipped',
                'event_id': existing_event.event_id,
                'reason': f'Duplicate of {existing_event.event_id} (similarity: {best_match["similarity"]:.2f})'
            }

        # Update the existing event
        try:
            self.db.update_event(merged_event.event_id, merged_event)
            self.vector_store.update_event(merged_event)
            print(f"   ↻ Updated: {merged_event.event_id} (enriched with new data)")
            return {
                'action': 'updated',
                'event_id': merged_event.event_id,
                'reason': f'Merged with {new_event.event_id}'
            }
        except Exception as e:
            print(f"   ✗ Failed to update: {e}")
            return {
                'action': 'skipped',
                'event_id': existing_event.event_id,
                'reason': f'Update error: {str(e)}'
            }

    def _find_potential_duplicates(
        self,
        event: MarketSignalEvent
    ) -> List[Dict[str, Any]]:
        """
        Find potential duplicate events using semantic search.

        Returns:
            List of dicts with 'event_id' and 'similarity' score
        """
        # Search for similar events
        query_text = f"{event.provider} {event.what_changed}"

        try:
            results = self.db.semantic_search_with_filters(
                query_text=query_text,
                vector_store=self.vector_store,
                provider=event.provider,
                start_date=event.published_at - timedelta(days=self.date_proximity_days),
                end_date=event.published_at + timedelta(days=self.date_proximity_days),
                limit=5
            )

            # Filter by similarity threshold
            duplicates = []
            for result in results:
                similarity = result.get('similarity_score', 0)
                print(f"     Candidate: {result['event_id']} (similarity: {similarity:.3f})")
                if similarity >= self.similarity_threshold:
                    duplicates.append({
                        'event_id': result['event_id'],
                        'similarity': similarity
                    })

            # Sort by similarity (highest first)
            duplicates.sort(key=lambda x: x['similarity'], reverse=True)

            if duplicates:
                print(f"     → Found {len(duplicates)} duplicates above threshold {self.similarity_threshold}")
            else:
                print(f"     → No duplicates above threshold {self.similarity_threshold}")

            return duplicates

        except Exception as e:
            print(f"   Warning: Duplicate search failed: {e}")
            return []

    def _merge_events(
        self,
        existing: MarketSignalEvent,
        new: MarketSignalEvent
    ) -> MarketSignalEvent:
        """
        Merge two events, keeping the most complete information.

        Strategy:
        - Keep longer/more detailed text fields
        - Merge pillar impacts (union)
        - Merge competitive effects (union)
        - Keep better source URL (prefer official sources)
        - Use more recent retrieval time
        """
        # Choose better values for each field
        merged_data = {
            'event_id': existing.event_id,  # Keep existing ID
            'provider': existing.provider,

            # Source info - prefer official sources
            'source_type': self._choose_better(
                existing.source_type, new.source_type,
                prefer='official_blog' in existing.source_type or 'official_blog' in new.source_type
            ),
            'source_url': existing.source_url if existing.source_url else new.source_url,

            # Dates
            'published_at': existing.published_at,  # Keep original
            'retrieved_at': max(existing.retrieved_at, new.retrieved_at),  # Most recent

            # Text fields - keep longer/more detailed
            'what_changed': self._choose_longer(existing.what_changed, new.what_changed),
            'why_it_matters': self._choose_longer(existing.why_it_matters, new.why_it_matters),
            'scope': existing.scope,

            # Complex fields - merge
            'pillars_impacted': self._merge_pillar_impacts(existing.pillars_impacted, new.pillars_impacted),
            'competitive_effects': self._merge_competitive_effects(existing.competitive_effects, new.competitive_effects),
            'temporal_context': existing.temporal_context,  # Keep existing (has context)

            # Other fields
            'alignment_implications': self._choose_longer(existing.alignment_implications, new.alignment_implications),
            'regulatory_signal': existing.regulatory_signal
        }

        # Create merged event
        return MarketSignalEvent(**merged_data)

    def _events_are_identical(self, event1: MarketSignalEvent, event2: MarketSignalEvent) -> bool:
        """Check if two events have identical content."""
        return (
            event1.what_changed == event2.what_changed and
            event1.why_it_matters == event2.why_it_matters and
            len(event1.pillars_impacted) == len(event2.pillars_impacted) and
            len(event1.competitive_effects.advantages_created) == len(event2.competitive_effects.advantages_created)
        )

    def _choose_longer(self, text1: str, text2: str) -> str:
        """Choose the longer/more detailed text."""
        if len(text2) > len(text1) * 1.2:  # 20% longer threshold
            return text2
        return text1

    def _choose_better(self, val1: Any, val2: Any, prefer: bool) -> Any:
        """Choose better value based on preference."""
        return val1 if prefer else val2

    def _merge_pillar_impacts(self, pillars1: List, pillars2: List) -> List:
        """Merge pillar impacts, keeping unique pillars with strongest signals."""
        # Create dict by pillar name
        merged = {}

        for impact in pillars1 + pillars2:
            pillar = impact.pillar_name
            if pillar not in merged:
                merged[pillar] = impact
            else:
                # Keep stronger signal
                existing = merged[pillar]
                if impact.relative_strength_signal.value > existing.relative_strength_signal.value:
                    merged[pillar] = impact

        return list(merged.values())

    def _merge_competitive_effects(self, effects1, effects2):
        """Merge competitive effects, taking union of advantages/barriers."""
        from ..models import CompetitiveEffects

        return CompetitiveEffects(
            advantages_created=list(set(effects1.advantages_created + effects2.advantages_created)),
            advantages_eroded=list(set(effects1.advantages_eroded + effects2.advantages_eroded)),
            new_barriers=list(set(effects1.new_barriers + effects2.new_barriers)),
            lock_in_or_openness_shift=effects1.lock_in_or_openness_shift  # Keep existing
        )

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _infer_provider_from_url(self, url: str) -> str:
        """Infer provider name from URL."""
        url_lower = url.lower()

        if 'openai.com' in url_lower:
            return 'OpenAI'
        elif 'anthropic.com' in url_lower:
            return 'Anthropic'
        elif 'google' in url_lower:
            return 'Google'
        elif 'microsoft.com' in url_lower:
            return 'Microsoft'
        elif 'meta.com' in url_lower or 'fb.com' in url_lower:
            return 'Meta'
        else:
            return 'Unknown'


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def ingest_live_sources(
    database: EventDatabase,
    vector_store: EventVectorStore,
    llm_provider: LLMProvider,
    config: Dict[str, Any],
    sources: Optional[List[str]] = None,
    providers: Optional[List[str]] = None,
    use_ensemble: bool = False
) -> Dict[str, Any]:
    """
    One-liner to ingest from live sources with deduplication.

    Uses Tavily to discover recent blog posts, validated against RSS feed freshness.

    Args:
        database: Event database
        vector_store: Vector store
        llm_provider: LLM provider
        config: Configuration dict
        sources: Optional list of source URLs (if provided, skips discovery)
        providers: Optional list of providers to discover (default: all)
        use_ensemble: Whether to use ensemble extraction

    Returns:
        Dict with ingestion results

    Example:
        # Auto-discover from all providers
        result = ingest_live_sources(
            database=db,
            vector_store=vector_store,
            llm_provider=llm,
            config=config
        )

        # Specific providers only
        result = ingest_live_sources(
            database=db,
            vector_store=vector_store,
            llm_provider=llm,
            config=config,
            providers=['OpenAI', 'Anthropic']
        )

        print(f"New: {result['new_events']}, Updated: {result['updated_events']}")
    """
    manager = LiveIngestionManager(database, vector_store, llm_provider, config)
    return manager.ingest_from_sources(sources, providers, use_ensemble)
