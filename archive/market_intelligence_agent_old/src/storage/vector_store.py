"""
Vector Store Layer using ChromaDB

Enables semantic search on Market Signal Events.

Why ChromaDB?
- Persistent storage (survives restarts)
- Built-in metadata filtering
- Simple API
- Good for MVP scale (millions of vectors)

Use cases:
- Timeline Analysis: "Find events about governance of agents"
- Similar Events: "What other events are like this one?"
- Pattern Detection: Cluster events to find strategic themes
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..models import MarketSignalEvent, Pillar


class EventVectorStore:
    """
    ChromaDB vector store for semantic search on events.

    Design decisions:
    - Embed: what_changed + why_it_matters + evidence (semantic content)
    - Metadata: provider, pillar, date (for filtering)
    - IDs: event_id (links back to SQL database)
    """

    def __init__(
        self,
        persist_directory: str = "./data/vector_store",
        collection_name: str = "market_events",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Where to store ChromaDB data
            collection_name: Name of the collection
            embedding_model: OpenAI embedding model to use
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Ensure directory exists
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False  # Disable telemetry
            )
        )

        # Get or create collection
        # OpenAI embedding function will be used
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Market Signal Events for competitive intelligence"}
        )

    def _create_embedding_text(self, event: MarketSignalEvent) -> str:
        """
        Create text to embed for an event.

        What to embed?
        - what_changed: Core factual content
        - why_it_matters: Competitive significance
        - Evidence from pillars: Specific details

        Why this combination?
        - Captures semantic meaning
        - Includes competitive context
        - Enough detail for similarity matching

        Returns:
            Combined text for embedding
        """
        # Start with core description
        text_parts = [
            f"Provider: {event.provider}",
            f"Change: {event.what_changed}",
            f"Significance: {event.why_it_matters}",
        ]

        # Add pillar evidence (rich semantic content)
        for impact in event.pillars_impacted:
            text_parts.append(
                f"{impact.pillar_name.value}: {impact.evidence}"
            )

        # Add competitive effects (key for similarity)
        if event.competitive_effects.advantages_created:
            text_parts.append(
                f"Advantages: {'; '.join(event.competitive_effects.advantages_created)}"
            )

        return " | ".join(text_parts)

    def _create_metadata(self, event: MarketSignalEvent) -> Dict[str, Any]:
        """
        Create metadata for filtering.

        ChromaDB allows filtering by metadata during queries.
        Useful for: "Find similar events from OpenAI" or
                   "Find events about DATA_PIPELINES"

        Returns:
            Metadata dict
        """
        return {
            "provider": event.provider,
            "published_at": event.published_at.isoformat(),
            "primary_pillar": event.get_primary_pillar().value,
            "is_first_move": event.is_first_move(),
            "total_impact_score": event.total_impact_score(),
            "source_url": event.source_url
        }

    def add_event(self, event: MarketSignalEvent) -> None:
        """
        Add an event to the vector store.

        Called after event is added to SQL database.

        Args:
            event: MarketSignalEvent to embed and store
        """
        # Create embedding text
        embedding_text = self._create_embedding_text(event)

        # Create metadata
        metadata = self._create_metadata(event)

        # Add to collection
        # ChromaDB will automatically generate embeddings using OpenAI
        self.collection.add(
            ids=[event.event_id],
            documents=[embedding_text],
            metadatas=[metadata]
        )

    def add_events_batch(self, events: List[MarketSignalEvent]) -> None:
        """
        Add multiple events at once (more efficient).

        Used during seed data import.

        Args:
            events: List of MarketSignalEvent objects
        """
        if not events:
            return

        ids = [event.event_id for event in events]
        documents = [self._create_embedding_text(event) for event in events]
        metadatas = [self._create_metadata(event) for event in events]

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

    def update_event(self, event: MarketSignalEvent) -> None:
        """
        Update an existing event in the vector store.

        Args:
            event: Updated MarketSignalEvent

        Note:
            ChromaDB update is done by deleting and re-adding since the
            embedding text may have changed (longer what_changed, etc.)
        """
        # Delete old version
        try:
            self.collection.delete(ids=[event.event_id])
        except Exception:
            # Event might not exist yet, that's OK
            pass

        # Add updated version
        self.add_event(event)

    def semantic_search(
        self,
        query: str,
        n_results: int = 10,
        provider: Optional[str] = None,
        pillar: Optional[Pillar] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for events matching a query.

        Use case: Timeline Analysis
        Example: "Find events about memory portability"

        Args:
            query: Natural language query
            n_results: Number of results to return
            provider: Filter by provider
            pillar: Filter by pillar

        Returns:
            List of dicts with:
                - event_id: ID of matching event
                - distance: Similarity score (lower = more similar)
                - metadata: Event metadata
        """
        # Build metadata filter
        where = {}
        if provider:
            where["provider"] = provider
        if pillar:
            where["primary_pillar"] = pillar.value

        # Query collection
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=where if where else None
        )

        # Format results
        formatted_results = []
        if results['ids'] and results['ids'][0]:
            for i, event_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    "event_id": event_id,
                    "distance": results['distances'][0][i] if results['distances'] else None,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {}
                })

        return formatted_results

    def find_similar_events(
        self,
        event_id: str,
        n_results: int = 5,
        exclude_same_provider: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Find events similar to a given event.

        Use case: Event Impact Analysis
        Example: "What other events are like GPT-4 200K context release?"

        Args:
            event_id: Event to find similar matches for
            n_results: Number of similar events
            exclude_same_provider: If True, only return events from other providers

        Returns:
            List of similar events with similarity scores
        """
        # Get the event's embedding by querying with its ID
        result = self.collection.get(
            ids=[event_id],
            include=['embeddings', 'metadatas']
        )

        if not result['ids']:
            return []

        # Get metadata for filtering
        where = None
        if exclude_same_provider and result['metadatas']:
            provider = result['metadatas'][0].get('provider')
            if provider:
                # ChromaDB uses $ne for "not equal"
                where = {"provider": {"$ne": provider}}

        # Search using the event's embedding
        # Note: We ask for n_results + 1 because the event itself will be included
        similar = self.collection.query(
            query_embeddings=result['embeddings'],
            n_results=n_results + 1,
            where=where
        )

        # Format and filter out the query event itself
        formatted_results = []
        if similar['ids'] and similar['ids'][0]:
            for i, similar_id in enumerate(similar['ids'][0]):
                if similar_id != event_id:  # Exclude the event itself
                    formatted_results.append({
                        "event_id": similar_id,
                        "distance": similar['distances'][0][i] if similar['distances'] else None,
                        "metadata": similar['metadatas'][0][i] if similar['metadatas'] else {}
                    })

        return formatted_results[:n_results]

    def get_event_clusters(
        self,
        provider: Optional[str] = None,
        pillar: Optional[Pillar] = None,
        min_cluster_size: int = 3
    ) -> List[List[str]]:
        """
        Cluster similar events to detect strategic themes.

        Use case: Pattern detection in Competitive Reasoning
        Example: "What are the major strategic themes for Microsoft?"

        Note: This is a simplified clustering approach for MVP.
        Production version could use proper clustering algorithms (K-means, DBSCAN).

        Args:
            provider: Filter by provider
            pillar: Filter by pillar
            min_cluster_size: Minimum events per cluster

        Returns:
            List of clusters (each cluster is a list of event_ids)
        """
        # Build metadata filter
        where = {}
        if provider:
            where["provider"] = provider
        if pillar:
            where["primary_pillar"] = pillar.value

        # Get all events matching filters
        results = self.collection.get(
            where=where if where else None,
            include=['embeddings', 'metadatas']
        )

        if not results['ids'] or len(results['ids']) < min_cluster_size:
            return []

        # Simple clustering: for each event, find its nearest neighbors
        # Events that are mutual nearest neighbors form a cluster
        # (This is a heuristic for MVP; can be improved)

        clusters = []
        processed = set()

        for i, event_id in enumerate(results['ids']):
            if event_id in processed:
                continue

            # Find similar events
            similar = self.find_similar_events(
                event_id,
                n_results=min_cluster_size - 1
            )

            if len(similar) >= min_cluster_size - 1:
                cluster = [event_id] + [s['event_id'] for s in similar[:min_cluster_size-1]]
                clusters.append(cluster)
                processed.update(cluster)

        return clusters

    def count_events(
        self,
        provider: Optional[str] = None,
        pillar: Optional[Pillar] = None
    ) -> int:
        """
        Count events in vector store matching filters.

        Used for validation and statistics.
        """
        where = {}
        if provider:
            where["provider"] = provider
        if pillar:
            where["primary_pillar"] = pillar.value

        result = self.collection.count()
        return result

    def delete_event(self, event_id: str) -> None:
        """
        Delete an event from the vector store.

        Args:
            event_id: Event ID to delete
        """
        self.collection.delete(ids=[event_id])

    def clear_all(self) -> None:
        """
        Clear all events from the vector store.

        Warning: This is destructive! Used for testing or complete resets.
        """
        # Delete and recreate collection
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Market Signal Events for competitive intelligence"}
        )
