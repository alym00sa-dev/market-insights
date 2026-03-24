"""
Seed Data Loader - Part 1: HTML Parser

Extracts events from market_demo.html and loads them into the database.

Data sources in HTML:
1. releases[] array (30 items) - Technical features with metadata
2. memoryProductsByYear{} object (41 items) - Product announcements with sources

Mapping strategy:
- featureType → I³ Pillars
- HTML fields → MarketSignalEvent fields
- Generate realistic competitive analysis
"""

import os
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from bs4 import BeautifulSoup

from ..models import (
    MarketSignalEvent,
    Pillar,
    DirectionOfChange,
    RelativeStrength,
    PillarImpact,
    CompetitiveEffects,
    TemporalContext
)
from ..storage import EventDatabase, EventVectorStore
from ..llm import LLMProvider


class SeedDataLoader:
    """
    Loads seed data from market_demo.html into the database.

    Two-phase approach:
    Phase 1: Parse HTML and extract JavaScript data
    Phase 2: Transform and enrich with LLM (optional)
    """

    def __init__(
        self,
        html_path: str,
        database: EventDatabase,
        vector_store: EventVectorStore,
        llm_provider: Optional[LLMProvider] = None
    ):
        """
        Initialize seed data loader.

        Args:
            html_path: Path to market_demo.html
            database: Event database
            vector_store: Vector store
            llm_provider: Optional LLM for enrichment
        """
        self.html_path = html_path
        self.db = database
        self.vector_store = vector_store
        self.llm = llm_provider

        # Feature type to pillar mapping
        self.feature_type_to_pillars = {
            'memory': [Pillar.DATA_PIPELINES, Pillar.TECHNICAL_CAPABILITIES],
            'agent': [Pillar.TECHNICAL_CAPABILITIES],
            'identity': [Pillar.MARKET_SHAPING],  # Identity lock-in
            'api': [Pillar.DATA_PIPELINES, Pillar.TECHNICAL_CAPABILITIES],
            'safety': [Pillar.ALIGNMENT],
            'governance': [Pillar.ALIGNMENT]
        }

    # ========================================================================
    # PHASE 1: PARSE HTML
    # ========================================================================

    def load_from_html(self, use_llm_enrichment: bool = False) -> Dict[str, Any]:
        """
        Load all events from HTML.

        Args:
            use_llm_enrichment: Whether to use LLM for enriching events

        Returns:
            Dict with:
                - events_created: Number of events created
                - event_ids: List of event IDs
                - errors: List of errors encountered
        """
        print("=" * 80)
        print("SEED DATA LOADER - Part 1: HTML Parser")
        print("=" * 80)

        # Parse HTML
        print(f"\n1. Parsing HTML from: {self.html_path}")
        raw_data = self._parse_html()
        print(f"   ✓ Extracted {len(raw_data['releases'])} releases")
        print(f"   ✓ Extracted {len(raw_data['products'])} product announcements")

        # Transform to events
        print("\n2. Transforming to MarketSignalEvent objects...")
        events = self._transform_to_events(raw_data, use_llm_enrichment)
        print(f"   ✓ Created {len(events)} events")

        # Store in database
        print("\n3. Storing events in database...")
        result = self._store_events(events)

        print("\n" + "=" * 80)
        print(f"COMPLETE: {result['events_created']} events loaded")
        print("=" * 80)

        return result

    def _parse_html(self) -> Dict[str, Any]:
        """
        Parse market_demo.html and extract JavaScript data.

        Returns:
            Dict with 'releases' and 'products' arrays
        """
        with open(self.html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Extract releases array (with nested structures)
        # Find the releases array section
        releases_start = html_content.find('releases: [')
        if releases_start != -1:
            # Find the matching closing bracket
            bracket_count = 0
            releases_end = releases_start + len('releases: [')
            in_array = True

            for i in range(releases_end, len(html_content)):
                if html_content[i] == '[':
                    bracket_count += 1
                elif html_content[i] == ']':
                    if bracket_count == 0:
                        releases_end = i
                        break
                    bracket_count -= 1

            releases_text = html_content[releases_start:releases_end + 1]
        else:
            releases_text = ''

        releases = []
        # Split by '}, {' pattern but handle nested arrays
        # Look for objects that start with 'id:' as markers
        id_matches = list(re.finditer(r"id:\s*'(r\d+)'", releases_text))

        for i, match in enumerate(id_matches):
            obj_start = releases_text.rfind('{', 0, match.start())
            # Find end of this object
            if i < len(id_matches) - 1:
                obj_end = releases_text.find('}', match.end())
                # Make sure we get to the right closing brace
                while obj_end < id_matches[i + 1].start() - 10:
                    obj_end = releases_text.find('}', obj_end + 1)
            else:
                # Last object
                obj_end = releases_text.rfind('}')

            obj_text = releases_text[obj_start:obj_end + 1]
            release = self._parse_js_object(obj_text)
            if release:
                releases.append(release)

        # Extract memoryProductsByYear
        products = []
        for year in [2023, 2024, 2025]:
            year_pattern = rf'{year}:\s*\[(.*?)\],'
            year_match = re.search(year_pattern, html_content, re.DOTALL)

            if year_match:
                year_text = year_match.group(1)
                product_objects = re.findall(r'\{[^}]+?\}(?=\s*(?:,|\]))', year_text)

                for obj_text in product_objects:
                    product = self._parse_js_object(obj_text)
                    if product:
                        product['year'] = year
                        products.append(product)

        return {
            'releases': releases,
            'products': products
        }

    def _parse_js_object(self, obj_text: str) -> Optional[Dict[str, Any]]:
        """
        Parse a JavaScript object string into a Python dict.

        Simplified parser for the specific format in market_demo.html.
        """
        try:
            obj = {}

            # Extract key-value pairs
            # Pattern: key: value (where value can be string, number, array)
            patterns = [
                (r"id:\s*'([^']+)'", 'id'),
                (r"provider:\s*'([^']+)'", 'provider'),
                (r"featureType:\s*'([^']+)'", 'featureType'),
                (r"date:\s*'([^']+)'", 'date'),
                (r"feature:\s*'([^']+)'", 'feature'),
                (r"description:\s*'([^']*?)'(?=,|\})", 'description'),
                (r"details:\s*'([^']*?)'(?=,|\})", 'details'),
                (r"impact:\s*'([^']+)'", 'impact'),
                (r"product:\s*'([^']+)'", 'product'),
                (r"category:\s*'([^']+)'", 'category'),
                (r"source:\s*'([^']*?)'(?=,|\})", 'source'),
                (r"sourceUrl:\s*'([^']+)'", 'sourceUrl'),
                (r"tokens:\s*(\d+)", 'tokens'),
                (r"adoptionRate:\s*(\d+)", 'adoptionRate'),
            ]

            for pattern, key in patterns:
                match = re.search(pattern, obj_text)
                if match:
                    value = match.group(1)
                    # Convert numbers
                    if key in ['tokens', 'adoptionRate']:
                        obj[key] = int(value)
                    else:
                        obj[key] = value

            return obj if obj else None

        except Exception as e:
            print(f"     Warning: Failed to parse object: {str(e)[:50]}")
            return None

    # ========================================================================
    # PHASE 2: TRANSFORM TO EVENTS
    # ========================================================================

    def _transform_to_events(
        self,
        raw_data: Dict[str, Any],
        use_llm_enrichment: bool
    ) -> List[MarketSignalEvent]:
        """
        Transform raw HTML data into MarketSignalEvent objects.

        Args:
            raw_data: Parsed HTML data
            use_llm_enrichment: Whether to use LLM for enrichment

        Returns:
            List of MarketSignalEvent objects
        """
        events = []

        # Process releases (more structured data)
        for release in raw_data['releases']:
            event = self._create_event_from_release(release, use_llm_enrichment)
            if event:
                events.append(event)

        # Process products (has source URLs, more detail)
        for product in raw_data['products']:
            event = self._create_event_from_product(product, use_llm_enrichment)
            if event:
                events.append(event)

        return events

    def _create_event_from_release(
        self,
        release: Dict[str, Any],
        use_llm: bool
    ) -> Optional[MarketSignalEvent]:
        """Create MarketSignalEvent from a release object."""
        try:
            # Generate event ID
            event_id = f"evt_{release['provider'].lower().replace(' ', '_')}_{release.get('id', 'unknown')}"

            # Parse date
            try:
                published_at = datetime.fromisoformat(release['date'])
            except:
                published_at = datetime.now()

            # Map feature type to pillars
            feature_type = release.get('featureType', 'memory')
            pillars = self.feature_type_to_pillars.get(feature_type, [Pillar.TECHNICAL_CAPABILITIES])

            # Create pillar impacts
            pillars_impacted = []
            for pillar in pillars:
                pillars_impacted.append(PillarImpact(
                    pillar_name=pillar,
                    direction_of_change=DirectionOfChange.ADVANCE,
                    relative_strength_signal=self._map_impact_to_strength(release.get('impact', 'medium')),
                    evidence=release.get('details', release.get('description', ''))
                ))

            # Create competitive effects
            competitive_effects = self._generate_competitive_effects(release, feature_type)

            # Create temporal context
            temporal_context = TemporalContext(
                preceded_by_events=[],
                likely_to_trigger_events=[],
                time_horizon='medium'
            )

            # Create event
            event = MarketSignalEvent(
                event_id=event_id,
                provider=release['provider'],
                source_type='official_blog',
                source_url=f"https://example.com/{release['provider'].lower()}/{release.get('id', 'release')}",
                published_at=published_at,
                retrieved_at=datetime.now(),
                what_changed=f"{release['provider']} released {release['feature']}: {release.get('description', '')}",
                why_it_matters=release.get('details', release.get('description', '')),
                scope='product_release',
                pillars_impacted=pillars_impacted,
                competitive_effects=competitive_effects,
                temporal_context=temporal_context,
                alignment_implications='Standard compliance and safety measures',
                regulatory_signal='none'
            )

            return event

        except Exception as e:
            print(f"     Error creating event from release {release.get('id', 'unknown')}: {e}")
            return None

    def _create_event_from_product(
        self,
        product: Dict[str, Any],
        use_llm: bool
    ) -> Optional[MarketSignalEvent]:
        """Create MarketSignalEvent from a product announcement."""
        try:
            # Generate event ID
            provider_slug = product['provider'].lower().replace(' ', '_')
            product_slug = product.get('product', 'product').lower().replace(' ', '_')[:30]
            event_id = f"evt_{provider_slug}_{product_slug}_{product.get('year', 2024)}"

            # Parse date
            try:
                published_at = datetime.fromisoformat(product['date'])
            except:
                published_at = datetime(product.get('year', 2024), 1, 1)

            # Infer pillar from category
            category = product.get('category', '').lower()
            pillars = self._infer_pillars_from_category(category)

            # Create pillar impacts
            pillars_impacted = []
            for pillar in pillars:
                evidence_text = product.get('description', 'Product announcement')
                # Ensure evidence meets min_length requirement
                if len(evidence_text) < 10:
                    evidence_text = f"{category.title()} advancement in frontier AI"

                pillars_impacted.append(PillarImpact(
                    pillar_name=pillar,
                    direction_of_change=DirectionOfChange.ADVANCE,
                    relative_strength_signal=RelativeStrength.MODERATE,
                    evidence=evidence_text
                ))

            # Create competitive effects
            competitive_effects = CompetitiveEffects(
                advantages_created=[f"Enhanced {category} capabilities"],
                advantages_eroded=[],
                new_barriers=[],
                lock_in_or_openness_shift='neutral'
            )

            # Create temporal context
            temporal_context = TemporalContext(
                preceded_by_events=[],
                likely_to_trigger_events=[],
                time_horizon='immediate'
            )

            # Create event
            event = MarketSignalEvent(
                event_id=event_id,
                provider=product['provider'],
                source_type='official_blog',
                source_url=product.get('sourceUrl', f"https://example.com/{provider_slug}"),
                published_at=published_at,
                retrieved_at=datetime.now(),
                what_changed=f"{product['provider']} announced {product.get('product', 'product')}: {product.get('description', '')}",
                why_it_matters=f"This {category} advancement affects competitive positioning in frontier AI markets.",
                scope='product_release',
                pillars_impacted=pillars_impacted,
                competitive_effects=competitive_effects,
                temporal_context=temporal_context,
                alignment_implications='Standard compliance and safety measures',
                regulatory_signal='none'
            )

            return event

        except Exception as e:
            print(f"     Error creating event from product: {e}")
            return None

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _map_impact_to_strength(self, impact: str) -> RelativeStrength:
        """Map impact level to RelativeStrength enum."""
        mapping = {
            'high': RelativeStrength.STRONG,
            'medium': RelativeStrength.MODERATE,
            'low': RelativeStrength.WEAK
        }
        return mapping.get(impact.lower(), RelativeStrength.MODERATE)

    def _infer_pillars_from_category(self, category: str) -> List[Pillar]:
        """Infer I³ pillars from product category."""
        category_lower = category.lower()

        if 'memory' in category_lower or 'persistent' in category_lower:
            return [Pillar.DATA_PIPELINES, Pillar.TECHNICAL_CAPABILITIES]
        elif 'agent' in category_lower or 'workflow' in category_lower:
            return [Pillar.TECHNICAL_CAPABILITIES]
        elif 'safety' in category_lower or 'governance' in category_lower:
            return [Pillar.ALIGNMENT]
        elif 'education' in category_lower or 'learning' in category_lower:
            return [Pillar.EDUCATION_INFLUENCE]
        elif 'graph' in category_lower or 'organizational' in category_lower:
            return [Pillar.DATA_PIPELINES, Pillar.MARKET_SHAPING]
        else:
            return [Pillar.TECHNICAL_CAPABILITIES]

    def _generate_competitive_effects(
        self,
        release: Dict[str, Any],
        feature_type: str
    ) -> CompetitiveEffects:
        """Generate competitive effects based on feature type."""
        feature = release.get('feature', '')
        description = release.get('description', '')

        advantages_created = []
        advantages_eroded = []
        new_barriers = []
        lock_in_shift = 'neutral'

        if feature_type == 'memory':
            advantages_created.append(f"Enhanced memory capacity ({release.get('tokens', 0)} tokens)")
            advantages_created.append("Improved user experience with persistent context")
            new_barriers.append("Proprietary memory format may create switching costs")
            lock_in_shift = 'increased_lock_in'

        elif feature_type == 'agent':
            advantages_created.append("New agent capabilities")
            advantages_created.append("Enhanced automation potential")

        elif feature_type == 'identity':
            advantages_created.append("Deeper identity integration")
            new_barriers.append("Identity binding creates exit barriers")
            lock_in_shift = 'increased_lock_in'

        elif feature_type == 'api':
            advantages_created.append(f"Expanded API surface ({release.get('apiEndpoints', 0)} new endpoints)")
            lock_in_shift = 'increased_openness'

        elif feature_type == 'safety':
            advantages_created.append("Enhanced compliance and safety controls")
            advantages_created.append("Reduced regulatory risk")

        return CompetitiveEffects(
            advantages_created=advantages_created,
            advantages_eroded=advantages_eroded,
            new_barriers=new_barriers,
            lock_in_or_openness_shift=lock_in_shift
        )

    # ========================================================================
    # PHASE 3: STORE EVENTS
    # ========================================================================

    def _store_events(self, events: List[MarketSignalEvent]) -> Dict[str, Any]:
        """
        Store events in database and vector store.

        Returns:
            Dict with events_created, event_ids, and errors
        """
        stored_count = 0
        event_ids = []
        errors = []

        for event in events:
            try:
                # Check if event already exists
                existing = self.db.get_event(event.event_id)
                if existing:
                    print(f"   ⊙ Skipping duplicate: {event.event_id}")
                    continue

                # Store in database
                self.db.create_event(event)

                # Store in vector store
                self.vector_store.add_event(event)

                stored_count += 1
                event_ids.append(event.event_id)
                print(f"   ✓ Stored: {event.event_id}")

            except Exception as e:
                error_msg = f"Failed to store {event.event_id}: {str(e)}"
                errors.append(error_msg)
                print(f"   ✗ {error_msg}")

        return {
            'events_created': stored_count,
            'event_ids': event_ids,
            'errors': errors
        }


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def load_seed_data(
    html_path: str,
    database: EventDatabase,
    vector_store: EventVectorStore,
    llm_provider: Optional[LLMProvider] = None,
    use_llm_enrichment: bool = False
) -> Dict[str, Any]:
    """
    One-liner to load seed data from HTML.

    Args:
        html_path: Path to market_demo.html
        database: Event database
        vector_store: Vector store
        llm_provider: Optional LLM provider
        use_llm_enrichment: Whether to use LLM enrichment (slower, better quality)

    Returns:
        Dict with events_created, event_ids, and errors

    Example:
        result = load_seed_data(
            html_path='./market_demo.html',
            database=db,
            vector_store=vector_store
        )
        print(f"Loaded {result['events_created']} events")
    """
    loader = SeedDataLoader(html_path, database, vector_store, llm_provider)
    return loader.load_from_html(use_llm_enrichment)
