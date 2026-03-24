"""
Multi-Provider LLM Wrapper

Abstracts away differences between Anthropic (Claude) and OpenAI (GPT-4).
Provides consistent interface for all agents.

Design decisions:
- Task complexity determines model choice (complex → Claude, simple → GPT-4)
- Fallback logic (if primary model fails, try secondary)
- Retry with exponential backoff
- Load config from config.yaml and .env
"""

import os
import time
import yaml
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
from dotenv import load_dotenv

from anthropic import Anthropic, APIError as AnthropicAPIError
from openai import OpenAI, APIError as OpenAIAPIError

# Load environment variables
load_dotenv()


class LLMProvider:
    """
    Multi-provider LLM wrapper supporting Claude and GPT-4.

    Why wrap instead of using clients directly?
    - Consistent interface across models
    - Easy model switching (config change, not code change)
    - Centralized error handling and retries
    - Cost tracking (can add later)
    """

    def __init__(self, config_path: str = "./config/config.yaml"):
        """
        Initialize LLM provider with configuration.

        Args:
            config_path: Path to config.yaml
        """
        self.config = self._load_config(config_path)

        # Initialize clients
        self.anthropic_client = Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )

        # Extract model configs
        self.complex_model_config = self.config['llm']['complex_model']
        self.simple_model_config = self.config['llm']['simple_model']
        self.fallback_enabled = self.config['llm'].get('fallback_enabled', True)
        self.max_retries = self.config['llm'].get('retry_attempts', 3)

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def generate(
        self,
        messages: List[Dict[str, str]],
        task_complexity: Literal["simple", "complex"] = "complex",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response using the appropriate model.

        This is the main entry point for all LLM calls.

        Args:
            messages: List of message dicts with 'role' and 'content'
            task_complexity: "simple" (use GPT-4) or "complex" (use Claude)
            temperature: Override default temperature
            max_tokens: Override default max tokens
            tools: Tool definitions (for structured output)
            tool_choice: Tool choice specification

        Returns:
            Response dict with 'content', 'model', 'usage', etc.

        Why separate simple vs complex?
        - Cost optimization: Claude is expensive, GPT-4o-mini is cheap
        - Speed: GPT-4o-mini is faster for simple tasks
        - Quality: Claude is better for complex reasoning

        Example simple tasks: Intent classification, parameter extraction
        Example complex tasks: Signal extraction, competitive analysis
        """
        # Select model config
        if task_complexity == "complex":
            model_config = self.complex_model_config
            provider = model_config['provider']
        else:
            model_config = self.simple_model_config
            provider = model_config['provider']

        # Override defaults if specified
        temperature = temperature if temperature is not None else model_config.get('temperature', 0.7)
        max_tokens = max_tokens if max_tokens is not None else model_config.get('max_tokens', 4096)

        # Generate with retries
        for attempt in range(self.max_retries):
            try:
                if provider == "anthropic":
                    return self._generate_anthropic(
                        messages=messages,
                        model=model_config['model'],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        tool_choice=tool_choice
                    )
                elif provider == "openai":
                    return self._generate_openai(
                        messages=messages,
                        model=model_config['model'],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        tool_choice=tool_choice
                    )
                else:
                    raise ValueError(f"Unknown provider: {provider}")

            except (AnthropicAPIError, OpenAIAPIError) as e:
                # Log error
                print(f"LLM API error (attempt {attempt + 1}/{self.max_retries}): {e}")

                # If this is the last attempt and fallback is enabled, try fallback
                if attempt == self.max_retries - 1 and self.fallback_enabled:
                    return self._try_fallback(
                        messages=messages,
                        failed_provider=provider,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        tools=tools,
                        tool_choice=tool_choice
                    )

                # Otherwise, wait and retry (exponential backoff)
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(wait_time)

        raise Exception(f"Failed to generate response after {self.max_retries} attempts")

    def _generate_anthropic(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate using Anthropic Claude.

        Returns standardized response format.
        """
        # Separate system message if present
        system_message = None
        api_messages = []

        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                api_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })

        # Build API call parameters
        params = {
            'model': model,
            'messages': api_messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }

        if system_message:
            params['system'] = system_message

        if tools:
            params['tools'] = tools

        if tool_choice:
            params['tool_choice'] = tool_choice

        # Call API
        response = self.anthropic_client.messages.create(**params)

        # Standardize response format
        return {
            'provider': 'anthropic',
            'model': model,
            'content': self._extract_anthropic_content(response),
            'tool_calls': self._extract_anthropic_tool_calls(response),
            'usage': {
                'input_tokens': response.usage.input_tokens,
                'output_tokens': response.usage.output_tokens
            },
            'raw_response': response
        }

    def _generate_openai(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate using OpenAI GPT-4.

        Returns standardized response format.
        """
        # Build API call parameters
        params = {
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }

        if tools:
            # Convert to OpenAI tools format if needed
            params['tools'] = tools

        if tool_choice:
            params['tool_choice'] = tool_choice

        # Call API
        response = self.openai_client.chat.completions.create(**params)

        # Standardize response format
        return {
            'provider': 'openai',
            'model': model,
            'content': response.choices[0].message.content,
            'tool_calls': self._extract_openai_tool_calls(response),
            'usage': {
                'input_tokens': response.usage.prompt_tokens,
                'output_tokens': response.usage.completion_tokens
            },
            'raw_response': response
        }

    def _extract_anthropic_content(self, response) -> str:
        """
        Extract text content from Anthropic response.

        Anthropic can return multiple content blocks (text + tool use).
        We extract just the text.
        """
        text_parts = []
        for block in response.content:
            if block.type == 'text':
                text_parts.append(block.text)
        return '\n'.join(text_parts)

    def _extract_anthropic_tool_calls(self, response) -> List[Dict[str, Any]]:
        """
        Extract tool calls from Anthropic response.

        Tool calls are used for structured output.
        """
        tool_calls = []
        for block in response.content:
            if block.type == 'tool_use':
                tool_calls.append({
                    'id': block.id,
                    'name': block.name,
                    'input': block.input
                })
        return tool_calls

    def _extract_openai_tool_calls(self, response) -> List[Dict[str, Any]]:
        """Extract tool calls from OpenAI response."""
        tool_calls = []
        message = response.choices[0].message

        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    'id': tc.id,
                    'name': tc.function.name,
                    'input': eval(tc.function.arguments)  # JSON string to dict
                })

        return tool_calls

    def _try_fallback(
        self,
        messages: List[Dict[str, str]],
        failed_provider: str,
        temperature: float,
        max_tokens: int,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Try fallback provider when primary fails.

        If Claude fails → try GPT-4
        If GPT-4 fails → try Claude

        Why fallback?
        - API outages happen
        - Rate limits can be hit
        - Better to get an answer than fail completely
        """
        print(f"Attempting fallback from {failed_provider}...")

        # Determine fallback provider
        if failed_provider == "anthropic":
            fallback_config = self.simple_model_config  # GPT-4
            fallback_provider = "openai"
        else:
            fallback_config = self.complex_model_config  # Claude
            fallback_provider = "anthropic"

        try:
            if fallback_provider == "anthropic":
                return self._generate_anthropic(
                    messages=messages,
                    model=fallback_config['model'],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice
                )
            else:
                return self._generate_openai(
                    messages=messages,
                    model=fallback_config['model'],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice
                )
        except Exception as e:
            raise Exception(f"Fallback also failed: {e}")

    def get_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding for text.

        Used by EventVectorStore for semantic search.

        Args:
            text: Text to embed
            model: Embedding model (default from config)

        Returns:
            Embedding vector (list of floats)
        """
        if model is None:
            model = self.config['storage']['vector_store']['embedding_model']

        # OpenAI embeddings
        response = self.openai_client.embeddings.create(
            input=text,
            model=model
        )

        return response.data[0].embedding
