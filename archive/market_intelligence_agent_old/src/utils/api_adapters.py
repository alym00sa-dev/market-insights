"""
External API Adapters

Adapters for fetching structured data from external APIs.

Why separate from Content Harvester?
- APIs return structured data (JSON), not HTML to parse
- Different error handling (API errors vs HTTP errors)
- Rate limiting specific to each API
- Easier to test/mock

Design:
- Each API gets its own adapter class
- All adapters implement common interface
- Content Harvester calls appropriate adapter based on source_type
"""

import os
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


class ArtificialAnalysisAdapter:
    """
    Adapter for Artificial Analysis API.

    Why this API matters:
    - Track model releases (new models appear in API)
    - Track pricing changes (competitive signal for TECHNICAL_CAPABILITIES)
    - Track performance improvements (speed, quality benchmarks)
    - Track competitive rankings (ELO scores)

    Use cases:
    - "OpenAI released new model" → appears in /data/llms/models
    - "Anthropic reduced pricing" → pricing field changes
    - "Google's Gemini Pro improved speed" → output_speed increases
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Artificial Analysis adapter.

        Args:
            config: Config dict from config.yaml['external_apis']['artificial_analysis']
        """
        self.base_url = config['base_url']
        self.endpoints = config['endpoints']
        self.api_key = os.getenv('ARTIFICIAL_ANALYSIS_API_KEY')

        if not self.api_key:
            print("Warning: ARTIFICIAL_ANALYSIS_API_KEY not set. API calls will fail.")

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make authenticated request to Artificial Analysis API.

        Args:
            endpoint: Endpoint path (e.g., '/data/llms/models')
            params: Optional query parameters

        Returns:
            Response JSON

        Raises:
            Exception if request fails
        """
        url = f"{self.base_url}{endpoint}"

        headers = {
            'x-api-key': self.api_key
        }

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Artificial Analysis API request failed: {e}")

    def fetch_llm_models(self) -> Dict[str, Any]:
        """
        Fetch LLM model benchmarks.

        Returns data about all tracked models including:
        - Model name, provider, release date
        - Pricing (input/output tokens)
        - Performance (speed, latency)
        - Quality benchmarks (MMLU, GPQA, etc.)

        Use case: Detect new model releases, pricing changes, performance improvements

        Returns:
            Dict with 'data' (list of models), 'retrieved_at', 'source_url'
        """
        data = self._make_request(self.endpoints['llms'])

        return {
            'data': data,
            'retrieved_at': datetime.now(),
            'source_url': f"{self.base_url}{self.endpoints['llms']}",
            'source_type': 'api_llm_benchmarks'
        }

    def fetch_text_to_image_models(self, include_categories: bool = False) -> Dict[str, Any]:
        """
        Fetch text-to-image model ELO rankings.

        Returns:
            Dict with 'data', 'retrieved_at', 'source_url'
        """
        params = {'include_categories': 'true'} if include_categories else None
        data = self._make_request(self.endpoints['text_to_image'], params=params)

        return {
            'data': data,
            'retrieved_at': datetime.now(),
            'source_url': f"{self.base_url}{self.endpoints['text_to_image']}",
            'source_type': 'api_image_gen_rankings'
        }

    def fetch_text_to_speech_models(self) -> Dict[str, Any]:
        """Fetch text-to-speech model ELO rankings."""
        data = self._make_request(self.endpoints['text_to_speech'])

        return {
            'data': data,
            'retrieved_at': datetime.now(),
            'source_url': f"{self.base_url}{self.endpoints['text_to_speech']}",
            'source_type': 'api_tts_rankings'
        }

    def fetch_text_to_video_models(self, include_categories: bool = False) -> Dict[str, Any]:
        """Fetch text-to-video model ELO rankings."""
        params = {'include_categories': 'true'} if include_categories else None
        data = self._make_request(self.endpoints['text_to_video'], params=params)

        return {
            'data': data,
            'retrieved_at': datetime.now(),
            'source_url': f"{self.base_url}{self.endpoints['text_to_video']}",
            'source_type': 'api_video_gen_rankings'
        }

    def detect_changes(
        self,
        current_data: List[Dict[str, Any]],
        previous_data: List[Dict[str, Any]],
        key_field: str = 'model_id'
    ) -> Dict[str, Any]:
        """
        Detect changes between current and previous API snapshots.

        This is where competitive signals are identified:
        - New models (in current, not in previous)
        - Removed models (in previous, not in current)
        - Changed pricing (same model, different price)
        - Performance improvements (same model, better speed/quality)

        Args:
            current_data: Current API response
            previous_data: Previous API response (from last fetch)
            key_field: Field to use as unique identifier (e.g., 'model_id')

        Returns:
            Dict with:
                - new_models: List of new models
                - removed_models: List of removed models
                - price_changes: List of models with price changes
                - performance_changes: List of models with performance changes

        Example:
            changes = adapter.detect_changes(current, previous)
            if changes['new_models']:
                # Generate event: "OpenAI released GPT-5"
            if changes['price_changes']:
                # Generate event: "Anthropic reduced Claude pricing 40%"
        """
        # Build lookup dictionaries
        previous_lookup = {item[key_field]: item for item in previous_data if key_field in item}
        current_lookup = {item[key_field]: item for item in current_data if key_field in item}

        changes = {
            'new_models': [],
            'removed_models': [],
            'price_changes': [],
            'performance_changes': []
        }

        # Detect new models
        for model_id, model_data in current_lookup.items():
            if model_id not in previous_lookup:
                changes['new_models'].append(model_data)
            else:
                # Model exists in both - check for changes
                prev = previous_lookup[model_id]
                curr = model_data

                # Price change detection
                if 'price_input' in prev and 'price_input' in curr:
                    if prev['price_input'] != curr['price_input'] or prev.get('price_output') != curr.get('price_output'):
                        changes['price_changes'].append({
                            'model': model_id,
                            'previous_price': {
                                'input': prev.get('price_input'),
                                'output': prev.get('price_output')
                            },
                            'current_price': {
                                'input': curr.get('price_input'),
                                'output': curr.get('price_output')
                            }
                        })

                # Performance change detection (e.g., speed improvements)
                if 'output_speed' in prev and 'output_speed' in curr:
                    speed_change = ((curr['output_speed'] - prev['output_speed']) / prev['output_speed']) * 100
                    if abs(speed_change) > 10:  # 10% change threshold
                        changes['performance_changes'].append({
                            'model': model_id,
                            'metric': 'output_speed',
                            'previous': prev['output_speed'],
                            'current': curr['output_speed'],
                            'change_percent': speed_change
                        })

        # Detect removed models
        for model_id in previous_lookup:
            if model_id not in current_lookup:
                changes['removed_models'].append(previous_lookup[model_id])

        return changes


class APIAdapterFactory:
    """
    Factory for creating API adapters.

    Usage:
        factory = APIAdapterFactory(config)
        adapter = factory.get_adapter('artificial_analysis')
        data = adapter.fetch_llm_models()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full config dict from config.yaml
        """
        self.config = config

    def get_adapter(self, api_name: str):
        """
        Get adapter for a specific API.

        Args:
            api_name: Name of API (e.g., 'artificial_analysis')

        Returns:
            Adapter instance

        Raises:
            ValueError if API not configured or unknown
        """
        if api_name == 'artificial_analysis':
            api_config = self.config.get('external_apis', {}).get('artificial_analysis')
            if not api_config or not api_config.get('enabled'):
                raise ValueError("Artificial Analysis API not enabled in config")
            return ArtificialAnalysisAdapter(api_config)
        else:
            raise ValueError(f"Unknown API: {api_name}")

    def get_enabled_apis(self) -> List[str]:
        """
        Get list of enabled API names.

        Returns:
            List of enabled API names (e.g., ['artificial_analysis'])
        """
        enabled = []
        apis = self.config.get('external_apis', {})

        for api_name, api_config in apis.items():
            if isinstance(api_config, dict) and api_config.get('enabled'):
                enabled.append(api_name)

        return enabled
