"""
Provider Memory Management

Manages persistent provider profiles (the system's "memory" about each hyperscaler).

Why separate from database.py?
- Different access patterns (profiles updated less frequently than events queried)
- Simpler JSON file storage (don't need SQL for 4-5 profiles)
- Easy to inspect/edit manually if needed

Design: Store profiles as JSON files in data/profiles/
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..models import ProviderProfile, Pillar, DEFAULT_PROVIDERS


class ProviderMemory:
    """
    Manages provider profiles (persistent memory about hyperscalers).

    Why JSON files instead of SQL?
    - Only ~5 providers (not thousands)
    - Infrequent updates (after ingesting events, not per-query)
    - Human-readable for debugging
    - Simpler than SQL for this use case
    """

    def __init__(self, profiles_directory: str = "./data/profiles"):
        """
        Initialize provider memory.

        Args:
            profiles_directory: Where to store provider profile JSON files
        """
        self.profiles_directory = Path(profiles_directory)
        self.profiles_directory.mkdir(parents=True, exist_ok=True)

        # In-memory cache of profiles
        self._cache: Dict[str, ProviderProfile] = {}

        # Load existing profiles on init
        self._load_all_profiles()

    def _get_profile_path(self, provider_name: str) -> Path:
        """Get file path for a provider's profile."""
        # Sanitize provider name for filename
        safe_name = provider_name.lower().replace(" ", "_")
        return self.profiles_directory / f"{safe_name}.json"

    def _load_all_profiles(self) -> None:
        """
        Load all existing provider profiles from disk.

        Called on init to populate cache.
        """
        for profile_file in self.profiles_directory.glob("*.json"):
            try:
                with open(profile_file, 'r') as f:
                    profile_data = json.load(f)
                    profile = ProviderProfile.model_validate(profile_data)
                    self._cache[profile.provider_name] = profile
            except Exception as e:
                print(f"Warning: Could not load profile {profile_file}: {e}")

        # If no profiles exist, initialize defaults
        if not self._cache:
            self._initialize_default_profiles()

    def _initialize_default_profiles(self) -> None:
        """
        Create default provider profiles on first run.

        Uses DEFAULT_PROVIDERS from models/provider.py
        """
        for provider_name, profile in DEFAULT_PROVIDERS.items():
            self._cache[provider_name] = profile
            self._save_profile(profile)

    def _save_profile(self, profile: ProviderProfile) -> None:
        """
        Save a profile to disk.

        Args:
            profile: ProviderProfile to save
        """
        profile_path = self._get_profile_path(profile.provider_name)

        with open(profile_path, 'w') as f:
            # Use Pydantic's model_dump for serialization
            json.dump(
                profile.model_dump(mode='json'),
                f,
                indent=2,
                default=str  # Handle datetime serialization
            )

    def get_profile(self, provider_name: str) -> ProviderProfile:
        """
        Get a provider's profile.

        If profile doesn't exist, creates a new one.

        Args:
            provider_name: Provider name (e.g., "OpenAI")

        Returns:
            ProviderProfile for the provider
        """
        if provider_name not in self._cache:
            # Create new profile
            from ..models.provider import get_or_create_provider
            profile = get_or_create_provider(provider_name)
            self._cache[provider_name] = profile
            self._save_profile(profile)

        return self._cache[provider_name]

    def update_profile(
        self,
        provider_name: str,
        **updates
    ) -> ProviderProfile:
        """
        Update a provider's profile.

        Args:
            provider_name: Provider to update
            **updates: Fields to update (e.g., total_events=10)

        Returns:
            Updated ProviderProfile

        Example:
            memory.update_profile(
                "OpenAI",
                total_events=42,
                last_event_date=datetime.now()
            )
        """
        profile = self.get_profile(provider_name)

        # Update fields
        for key, value in updates.items():
            if hasattr(profile, key):
                setattr(profile, key, value)

        profile.last_updated = datetime.now()

        # Save to disk
        self._save_profile(profile)

        return profile

    def update_from_event(
        self,
        provider_name: str,
        event_date: datetime
    ) -> None:
        """
        Update profile based on a new event being ingested.

        Called by event ingestion workflow.

        Args:
            provider_name: Provider name
            event_date: When the event was published
        """
        profile = self.get_profile(provider_name)
        profile.increment_event_count(event_date)
        self._save_profile(profile)

    def add_pillar_insight(
        self,
        provider_name: str,
        pillar: Pillar,
        insight: str
    ) -> None:
        """
        Add or update pillar-specific insight for a provider.

        Called by Competitive Reasoning Agent when it generates insights.

        Args:
            provider_name: Provider name
            pillar: Which pillar
            insight: Insight description

        Example:
            memory.add_pillar_insight(
                "Microsoft",
                Pillar.DATA_PIPELINES,
                "Strong focus on GDPR compliance and portability"
            )
        """
        profile = self.get_profile(provider_name)
        profile.update_pillar_strength(pillar, insight)
        self._save_profile(profile)

    def add_strategy(
        self,
        provider_name: str,
        strategy: str
    ) -> None:
        """
        Add a strategic pattern observation for a provider.

        Args:
            provider_name: Provider name
            strategy: Strategy description

        Example:
            memory.add_strategy(
                "OpenAI",
                "Walled garden approach - prioritizes capability over portability"
            )
        """
        profile = self.get_profile(provider_name)
        profile.add_strategy(strategy)
        self._save_profile(profile)

    def add_behavior_pattern(
        self,
        provider_name: str,
        pattern: str
    ) -> None:
        """
        Add a behavioral pattern observation.

        Args:
            provider_name: Provider name
            pattern: Pattern description

        Example:
            memory.add_behavior_pattern(
                "Anthropic",
                "Tends to lead with governance disclosures before feature releases"
            )
        """
        profile = self.get_profile(provider_name)
        profile.add_behavior_pattern(pattern)
        self._save_profile(profile)

    def get_all_providers(self) -> List[str]:
        """
        Get list of all known providers.

        Returns:
            List of provider names
        """
        return list(self._cache.keys())

    def get_all_profiles(self) -> Dict[str, ProviderProfile]:
        """
        Get all provider profiles.

        Returns:
            Dict mapping provider name to ProviderProfile
        """
        return self._cache.copy()

    def get_pillar_leaders(
        self,
        pillar: Pillar
    ) -> List[tuple[str, str]]:
        """
        Get providers with known strengths in a pillar.

        Returns:
            List of (provider_name, strength_description) tuples

        Example:
            leaders = memory.get_pillar_leaders(Pillar.DATA_PIPELINES)
            # [("Microsoft", "Strong GDPR compliance"), ...]
        """
        leaders = []

        for provider_name, profile in self._cache.items():
            strength = profile.strengths_by_pillar.get(pillar)
            if strength:
                leaders.append((provider_name, strength))

        return leaders

    def export_summary(self) -> Dict[str, Any]:
        """
        Export a summary of all provider knowledge.

        Useful for debugging and reporting.

        Returns:
            Dict with summary statistics
        """
        summary = {
            "total_providers": len(self._cache),
            "providers": {}
        }

        for provider_name, profile in self._cache.items():
            summary["providers"][provider_name] = {
                "total_events": profile.total_events,
                "first_event": profile.first_event_date.isoformat() if profile.first_event_date else None,
                "last_event": profile.last_event_date.isoformat() if profile.last_event_date else None,
                "known_strategies": len(profile.known_strategies),
                "pillar_strengths": len(profile.strengths_by_pillar),
                "behavior_patterns": len(profile.historical_behavior_patterns)
            }

        return summary
