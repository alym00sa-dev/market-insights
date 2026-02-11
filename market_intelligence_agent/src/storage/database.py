"""
SQLite Database Layer

Manages structured storage of Market Signal Events, sources, and content fetches.

Design decisions:
- JSON columns for nested data (pillars_impacted, competitive_effects) - flexible schema
- Indices on common query patterns (provider, pillar, date)
- Append-only for events (maintain history)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
from contextlib import contextmanager

from ..models import MarketSignalEvent, Pillar


class EventDatabase:
    """
    SQLite database for Market Signal Events.

    Why not an ORM like SQLAlchemy?
    - Simpler for MVP (direct SQL is clearer)
    - Easier to optimize queries
    - Less abstraction = easier to debug
    """

    def __init__(self, db_path: str = "./data/events.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize schema on first run
        self._init_schema()

    @contextmanager
    def _get_connection(self):
        """
        Context manager for database connections.

        Why context manager?
        - Ensures connections are always closed
        - Handles transactions (commit on success, rollback on error)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """
        Create database tables if they don't exist.

        Design decisions:
        - events table: Stores MarketSignalEvent as JSON + key fields for indexing
        - sources table: Tracks reliability of sources over time
        - content_fetches table: Raw content + significance scores
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # ================================================================
            # EVENTS TABLE - Core market signal events
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    source_type TEXT,
                    source_url TEXT,
                    published_at TEXT NOT NULL,
                    retrieved_at TEXT NOT NULL,

                    -- Event description (for quick access without parsing JSON)
                    what_changed TEXT NOT NULL,
                    why_it_matters TEXT NOT NULL,
                    scope TEXT,

                    -- Pillar tracking (for fast filtering without parsing JSON)
                    pillars TEXT,  -- Comma-separated list: "TECHNICAL_CAPABILITIES,DATA_PIPELINES"

                    -- Full event data as JSON (includes all nested structures)
                    event_json TEXT NOT NULL,

                    -- Metadata
                    extraction_confidence REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                    -- Indices will be created separately
                    CHECK (extraction_confidence IS NULL OR (extraction_confidence >= 0 AND extraction_confidence <= 1))
                )
            """)

            # Index for common queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_provider
                ON events(provider)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_published_at
                ON events(published_at DESC)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_provider_date
                ON events(provider, published_at DESC)
            """)

            # Index for pillar-based queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_events_pillars
                ON events(pillars)
            """)

            # ================================================================
            # TEMPORAL_CHAINS TABLE - Event relationships
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS temporal_chains (
                    from_event_id TEXT NOT NULL,
                    to_event_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,  -- 'preceded_by' or 'likely_to_trigger'
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,

                    PRIMARY KEY (from_event_id, to_event_id, relationship_type),
                    FOREIGN KEY (from_event_id) REFERENCES events(event_id),
                    FOREIGN KEY (to_event_id) REFERENCES events(event_id),
                    CHECK (relationship_type IN ('preceded_by', 'likely_to_trigger'))
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_temporal_from
                ON temporal_chains(from_event_id, relationship_type)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_temporal_to
                ON temporal_chains(to_event_id, relationship_type)
            """)

            # ================================================================
            # SOURCES TABLE - Track source reliability
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    url TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    source_type TEXT,
                    first_discovered TEXT,
                    last_checked TEXT,
                    total_fetches INTEGER DEFAULT 0,
                    successful_events INTEGER DEFAULT 0,
                    reliability_score REAL DEFAULT 0.0,
                    enabled BOOLEAN DEFAULT 1,

                    CHECK (reliability_score >= 0 AND reliability_score <= 1)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sources_provider
                ON sources(provider)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sources_reliability
                ON sources(reliability_score DESC)
            """)

            # ================================================================
            # CONTENT_FETCHES TABLE - Raw content + significance analysis
            # ================================================================
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS content_fetches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    content_hash TEXT,

                    -- Raw and filtered content
                    raw_content TEXT,
                    filtered_content TEXT,

                    -- Significance analysis (from Content Harvester LLM)
                    significance_score INTEGER,
                    content_type TEXT,
                    metadata_json TEXT,

                    -- Fetch status
                    fetch_successful BOOLEAN,
                    error_message TEXT,

                    FOREIGN KEY (url) REFERENCES sources(url)
                )
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_url_hash
                ON content_fetches(url, content_hash)
            """)

            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_content_significance
                ON content_fetches(significance_score DESC)
            """)

    # ========================================================================
    # EVENT CRUD OPERATIONS
    # ========================================================================

    def create_event(self, event: MarketSignalEvent) -> bool:
        """
        Store a new Market Signal Event.

        Args:
            event: MarketSignalEvent to store

        Returns:
            True if created, False if event_id already exists

        Why store as JSON?
        - Flexible schema (can add fields without migrations)
        - Preserves all nested structures (pillars_impacted, competitive_effects)
        - Key fields extracted for indexing/querying
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            try:
                # Extract pillars for indexing
                pillars_list = [p.pillar_name.value for p in event.pillars_impacted]
                pillars_str = ",".join(pillars_list) if pillars_list else ""

                # Get extraction confidence (may not exist on model)
                extraction_confidence = getattr(event, 'extraction_confidence', None)

                cursor.execute("""
                    INSERT INTO events (
                        event_id, provider, source_type, source_url,
                        published_at, retrieved_at,
                        what_changed, why_it_matters, scope,
                        pillars, event_json, extraction_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.provider,
                    event.source_type,
                    event.source_url,
                    event.published_at.isoformat(),
                    event.retrieved_at.isoformat(),
                    event.what_changed,
                    event.why_it_matters,
                    event.scope,
                    pillars_str,
                    event.model_dump_json(),  # Pydantic serialization
                    extraction_confidence
                ))

                # Store temporal chains
                self._store_temporal_chains(event)

                return True
            except sqlite3.IntegrityError:
                # event_id already exists
                return False

    def update_event(self, event_id: str, event: MarketSignalEvent) -> bool:
        """
        Update an existing Market Signal Event.

        Args:
            event_id: Event ID to update
            event: Updated MarketSignalEvent object

        Returns:
            True if updated, False if event not found

        Use case:
        - Enriching events with additional information from multiple sources
        - Merging duplicate events while keeping the most complete data
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Check if event exists
            cursor.execute("SELECT event_id FROM events WHERE event_id = ?", (event_id,))
            if not cursor.fetchone():
                return False

            # Extract pillars for indexing
            pillars_list = [p.pillar_name.value for p in event.pillars_impacted]
            pillars_str = ",".join(pillars_list) if pillars_list else ""

            # Get extraction confidence
            extraction_confidence = getattr(event, 'extraction_confidence', None)

            # Update event
            cursor.execute("""
                UPDATE events
                SET provider = ?,
                    source_type = ?,
                    source_url = ?,
                    published_at = ?,
                    retrieved_at = ?,
                    what_changed = ?,
                    why_it_matters = ?,
                    scope = ?,
                    pillars = ?,
                    event_json = ?,
                    extraction_confidence = ?
                WHERE event_id = ?
            """, (
                event.provider,
                event.source_type,
                event.source_url,
                event.published_at.isoformat(),
                event.retrieved_at.isoformat(),
                event.what_changed,
                event.why_it_matters,
                event.scope,
                pillars_str,
                event.model_dump_json(),
                extraction_confidence,
                event_id
            ))

            # Delete old temporal chains
            cursor.execute("DELETE FROM temporal_chains WHERE from_event_id = ?", (event_id,))

            # Store new temporal chains
            self._store_temporal_chains(event)

            return True

    def get_event(self, event_id: str) -> Optional[MarketSignalEvent]:
        """
        Retrieve event by ID.

        Returns:
            MarketSignalEvent or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_json FROM events WHERE event_id = ?
            """, (event_id,))

            row = cursor.fetchone()
            if row:
                return MarketSignalEvent.model_validate_json(row['event_json'])
            return None

    def get_events_by_provider(
        self,
        provider: str,
        limit: Optional[int] = None
    ) -> List[MarketSignalEvent]:
        """
        Get all events for a specific provider.

        Args:
            provider: Provider name (e.g., "OpenAI")
            limit: Max number of events to return (most recent first)

        Returns:
            List of MarketSignalEvent objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT event_json FROM events
                WHERE provider = ?
                ORDER BY published_at DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query, (provider,))

            return [
                MarketSignalEvent.model_validate_json(row['event_json'])
                for row in cursor.fetchall()
            ]

    def get_events_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        provider: Optional[str] = None
    ) -> List[MarketSignalEvent]:
        """
        Get events within a date range.

        Args:
            start_date: Start of range (inclusive)
            end_date: End of range (inclusive)
            provider: Optional provider filter

        Returns:
            List of MarketSignalEvent objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT event_json FROM events
                WHERE published_at BETWEEN ? AND ?
            """
            params = [start_date.isoformat(), end_date.isoformat()]

            if provider:
                query += " AND provider = ?"
                params.append(provider)

            query += " ORDER BY published_at DESC"

            cursor.execute(query, params)

            return [
                MarketSignalEvent.model_validate_json(row['event_json'])
                for row in cursor.fetchall()
            ]

    def get_events_by_pillar(
        self,
        pillar: Pillar,
        provider: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[MarketSignalEvent]:
        """
        Get events that impact a specific pillar.

        Note: This requires parsing JSON, so it's slower than indexed queries.
        For MVP this is fine; can optimize later with a separate pillars table.

        Args:
            pillar: Pillar to filter by
            provider: Optional provider filter
            limit: Max number of events

        Returns:
            List of MarketSignalEvent objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT event_json FROM events
                WHERE json_extract(event_json, '$.pillars_impacted') LIKE ?
            """
            params = [f'%{pillar.value}%']

            if provider:
                query += " AND provider = ?"
                params.append(provider)

            query += " ORDER BY published_at DESC"

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query, params)

            return [
                MarketSignalEvent.model_validate_json(row['event_json'])
                for row in cursor.fetchall()
            ]

    def search_events(
        self,
        provider: Optional[str] = None,
        pillar: Optional[Pillar] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_confidence: Optional[float] = None,
        limit: Optional[int] = 50
    ) -> List[MarketSignalEvent]:
        """
        Flexible event search with multiple filters.

        This is the main query function used by Competitive Reasoning Agent.

        Args:
            provider: Filter by provider
            pillar: Filter by pillar
            start_date: Filter by date range (start)
            end_date: Filter by date range (end)
            min_confidence: Minimum extraction confidence
            limit: Max results

        Returns:
            List of MarketSignalEvent objects matching filters
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT event_json FROM events WHERE 1=1"
            params = []

            if provider:
                query += " AND provider = ?"
                params.append(provider)

            if pillar:
                query += " AND json_extract(event_json, '$.pillars_impacted') LIKE ?"
                params.append(f'%{pillar.value}%')

            if start_date:
                query += " AND published_at >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND published_at <= ?"
                params.append(end_date.isoformat())

            if min_confidence is not None:
                query += " AND extraction_confidence >= ?"
                params.append(min_confidence)

            query += " ORDER BY published_at DESC"

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query, params)

            return [
                MarketSignalEvent.model_validate_json(row['event_json'])
                for row in cursor.fetchall()
            ]

    def get_all_providers(self) -> List[str]:
        """Get list of all providers in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT provider FROM events ORDER BY provider")
            return [row['provider'] for row in cursor.fetchall()]

    def get_event_count(
        self,
        provider: Optional[str] = None,
        pillar: Optional[Pillar] = None
    ) -> int:
        """
        Count events matching filters.

        Used for statistics and validation.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = "SELECT COUNT(*) as count FROM events WHERE 1=1"
            params = []

            if provider:
                query += " AND provider = ?"
                params.append(provider)

            if pillar:
                query += " AND json_extract(event_json, '$.pillars_impacted') LIKE ?"
                params.append(f'%{pillar.value}%')

            cursor.execute(query, params)
            return cursor.fetchone()['count']

    # ========================================================================
    # SOURCE TRACKING OPERATIONS
    # ========================================================================

    def add_or_update_source(
        self,
        url: str,
        provider: str,
        source_type: str
    ) -> None:
        """
        Add a new source or update if exists.

        Called by Source Scout when discovering sources.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO sources (url, provider, source_type, first_discovered)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    last_checked = CURRENT_TIMESTAMP
            """, (url, provider, source_type, datetime.now().isoformat()))

    def update_source_reliability(self, url: str, successful: bool) -> None:
        """
        Update source reliability after Content Harvester fetches.

        Args:
            url: Source URL
            successful: Whether fetch resulted in a valid event

        Updates:
            - total_fetches (increment)
            - successful_events (increment if successful)
            - reliability_score (recalculate: successful_events / total_fetches)
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Increment counters
            cursor.execute("""
                UPDATE sources
                SET
                    total_fetches = total_fetches + 1,
                    successful_events = successful_events + ?,
                    last_checked = ?
                WHERE url = ?
            """, (1 if successful else 0, datetime.now().isoformat(), url))

            # Recalculate reliability score
            cursor.execute("""
                UPDATE sources
                SET reliability_score = CAST(successful_events AS REAL) / NULLIF(total_fetches, 0)
                WHERE url = ?
            """, (url,))

    def get_sources_by_reliability(
        self,
        provider: Optional[str] = None,
        min_reliability: float = 0.0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get sources ranked by reliability score.

        Used by Source Scout to prioritize high-quality sources.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT url, provider, source_type, reliability_score,
                       total_fetches, successful_events
                FROM sources
                WHERE reliability_score >= ? AND enabled = 1
            """
            params = [min_reliability]

            if provider:
                query += " AND provider = ?"
                params.append(provider)

            query += " ORDER BY reliability_score DESC, total_fetches DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            return [dict(row) for row in cursor.fetchall()]

    # ========================================================================
    # PILLAR-BASED QUERIES
    # ========================================================================

    def get_events_by_pillar(
        self,
        pillar: Pillar,
        provider: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[MarketSignalEvent]:
        """
        Get events that impact a specific I³ pillar.

        Uses the indexed pillars column for fast filtering.

        Args:
            pillar: Pillar to filter by
            provider: Optional provider filter
            start_date: Optional start date
            end_date: Optional end date
            limit: Max results

        Returns:
            List of MarketSignalEvent objects
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = """
                SELECT event_json FROM events
                WHERE pillars LIKE ?
            """
            params = [f'%{pillar.value}%']

            if provider:
                query += " AND provider = ?"
                params.append(provider)

            if start_date:
                query += " AND published_at >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND published_at <= ?"
                params.append(end_date.isoformat())

            query += " ORDER BY published_at DESC"

            if limit:
                query += f" LIMIT {limit}"

            cursor.execute(query, params)

            return [
                MarketSignalEvent.model_validate_json(row['event_json'])
                for row in cursor.fetchall()
            ]

    # ========================================================================
    # TEMPORAL CHAIN OPERATIONS
    # ========================================================================

    def _store_temporal_chains(self, event: MarketSignalEvent) -> None:
        """
        Store temporal relationships from event's temporal_context.

        Extracts preceded_by and likely_to_trigger relationships
        and stores them in the temporal_chains table.

        Args:
            event: Event with temporal_context
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Store preceded_by relationships
            if event.temporal_context and event.temporal_context.preceded_by_events:
                for preceded_event in event.temporal_context.preceded_by_events:
                    # Only store if it looks like an event_id (not a description)
                    if preceded_event.startswith('evt_'):
                        try:
                            cursor.execute("""
                                INSERT OR IGNORE INTO temporal_chains
                                (from_event_id, to_event_id, relationship_type)
                                VALUES (?, ?, 'preceded_by')
                            """, (preceded_event, event.event_id))
                        except sqlite3.IntegrityError:
                            # Foreign key constraint - event doesn't exist yet
                            pass

            # Store likely_to_trigger relationships
            if event.temporal_context and event.temporal_context.likely_to_trigger_events:
                for triggered_event in event.temporal_context.likely_to_trigger_events:
                    # These are predictions, not actual event_ids yet
                    # Store as descriptions for now (will link later if events created)
                    pass  # Skip for now - these are predictions, not actual links

    def get_events_preceded_by(self, event_id: str) -> List[MarketSignalEvent]:
        """
        Get events that preceded this event (causal ancestors).

        Args:
            event_id: Event to find predecessors for

        Returns:
            List of events that came before and influenced this one
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT e.event_json
                FROM events e
                JOIN temporal_chains tc ON e.event_id = tc.from_event_id
                WHERE tc.to_event_id = ? AND tc.relationship_type = 'preceded_by'
                ORDER BY e.published_at ASC
            """, (event_id,))

            return [
                MarketSignalEvent.model_validate_json(row['event_json'])
                for row in cursor.fetchall()
            ]

    def get_events_triggered_by(self, event_id: str) -> List[MarketSignalEvent]:
        """
        Get events that were triggered by this event (causal descendants).

        Args:
            event_id: Event to find descendants for

        Returns:
            List of events that followed and were influenced by this one
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT e.event_json
                FROM events e
                JOIN temporal_chains tc ON e.event_id = tc.to_event_id
                WHERE tc.from_event_id = ? AND tc.relationship_type = 'preceded_by'
                ORDER BY e.published_at ASC
            """, (event_id,))

            return [
                MarketSignalEvent.model_validate_json(row['event_json'])
                for row in cursor.fetchall()
            ]

    def get_event_chain(
        self,
        event_id: str,
        direction: Literal['predecessors', 'successors', 'both'] = 'both',
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Get the full causal chain for an event.

        Recursively traverses temporal_chains to build a graph of related events.

        Args:
            event_id: Starting event
            direction: Which direction to traverse
            max_depth: Maximum depth to traverse (prevents infinite loops)

        Returns:
            Dict with:
                - center_event: The starting event
                - predecessors: Events that came before (if direction includes predecessors)
                - successors: Events that came after (if direction includes successors)
                - graph_depth: How many levels were traversed
        """
        center_event = self.get_event(event_id)
        if not center_event:
            return {'error': f'Event {event_id} not found'}

        result = {
            'center_event': center_event,
            'predecessors': [],
            'successors': [],
            'graph_depth': 0
        }

        if direction in ['predecessors', 'both']:
            result['predecessors'] = self._traverse_chain(
                event_id, 'backwards', max_depth, set()
            )

        if direction in ['successors', 'both']:
            result['successors'] = self._traverse_chain(
                event_id, 'forwards', max_depth, set()
            )

        result['graph_depth'] = max(
            len(result['predecessors']),
            len(result['successors'])
        )

        return result

    def _traverse_chain(
        self,
        event_id: str,
        direction: Literal['backwards', 'forwards'],
        max_depth: int,
        visited: set
    ) -> List[MarketSignalEvent]:
        """
        Recursively traverse temporal chain.

        Args:
            event_id: Current event
            direction: 'backwards' (predecessors) or 'forwards' (successors)
            max_depth: Remaining depth to traverse
            visited: Set of visited event_ids (cycle detection)

        Returns:
            List of events in the chain
        """
        if max_depth == 0 or event_id in visited:
            return []

        visited.add(event_id)

        # Get related events
        if direction == 'backwards':
            related = self.get_events_preceded_by(event_id)
        else:
            related = self.get_events_triggered_by(event_id)

        # Recursively traverse
        all_events = list(related)
        for event in related:
            all_events.extend(
                self._traverse_chain(event.event_id, direction, max_depth - 1, visited)
            )

        return all_events

    # ========================================================================
    # HYBRID SEARCH (SQL + Vector)
    # ========================================================================

    def semantic_search_with_filters(
        self,
        query_text: str,
        vector_store,  # EventVectorStore instance
        provider: Optional[str] = None,
        pillar: Optional[Pillar] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: Semantic similarity + SQL filters.

        Flow:
        1. Vector store finds semantically similar events (top 100)
        2. SQL filters by provider, pillar, date (from those 100)
        3. Return top N results

        Args:
            query_text: Natural language query
            vector_store: EventVectorStore instance
            provider: Optional provider filter
            pillar: Optional pillar filter
            start_date: Optional start date
            end_date: Optional end date
            limit: Max results

        Returns:
            List of dicts with event data + similarity scores

        Example:
            results = db.semantic_search_with_filters(
                "context window improvements",
                vector_store,
                pillar=Pillar.TECHNICAL_CAPABILITIES,
                limit=5
            )
        """
        # Step 1: Semantic search (vector store)
        semantic_results = vector_store.semantic_search(
            query=query_text,
            n_results=100  # Get more candidates for filtering
        )

        # Extract event_ids from semantic results
        candidate_ids = [r['event_id'] for r in semantic_results]

        if not candidate_ids:
            return []

        # Step 2: SQL filtering
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Build query with filters
            placeholders = ','.join(['?'] * len(candidate_ids))
            query = f"""
                SELECT event_id, event_json, what_changed, why_it_matters,
                       provider, published_at, pillars
                FROM events
                WHERE event_id IN ({placeholders})
            """
            params = candidate_ids

            if provider:
                query += " AND provider = ?"
                params.append(provider)

            if pillar:
                query += " AND pillars LIKE ?"
                params.append(f'%{pillar.value}%')

            if start_date:
                query += " AND published_at >= ?"
                params.append(start_date.isoformat())

            if end_date:
                query += " AND published_at <= ?"
                params.append(end_date.isoformat())

            cursor.execute(query, params)
            filtered_events = {row['event_id']: dict(row) for row in cursor.fetchall()}

        # Step 3: Combine results (preserve similarity order)
        results = []
        for semantic_result in semantic_results:
            event_id = semantic_result['event_id']
            if event_id in filtered_events:
                # Convert distance to similarity (ChromaDB returns distance, lower = more similar)
                # similarity = 1 - distance (normalized to 0-1 range)
                distance = semantic_result.get('distance', 1.0)
                similarity = max(0.0, 1.0 - distance)  # Clamp to 0-1

                result = {
                    **filtered_events[event_id],
                    'similarity_score': similarity,
                    'distance': distance
                }
                results.append(result)

                if len(results) >= limit:
                    break

        return results
