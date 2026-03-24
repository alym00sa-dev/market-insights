from functools import lru_cache
from llm.provider import LLMProvider
from config.settings import settings


@lru_cache(maxsize=4)
def get_llm(provider: str = None) -> LLMProvider:
    provider = (provider or settings.LLM_PROVIDER).lower()
    if provider == "claude":
        from llm.claude_provider import ClaudeProvider
        return ClaudeProvider()
    elif provider == "openai":
        from llm.openai_provider import OpenAIProvider
        return OpenAIProvider()
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider!r}. Set LLM_PROVIDER to 'claude' or 'openai'."
        )


def set_active_model(provider: str, model: str):
    """Switch the active LLM provider and model at runtime. Clears the cache."""
    settings.LLM_PROVIDER = provider
    if provider == "claude":
        settings.CLAUDE_MODEL = model
    elif provider == "openai":
        settings.OPENAI_MODEL = model
    get_llm.cache_clear()
