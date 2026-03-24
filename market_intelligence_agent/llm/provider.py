from abc import ABC, abstractmethod


class LLMProvider(ABC):

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return a plain text completion."""
        ...

    @abstractmethod
    def complete_structured(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        """Return a JSON-parsed structured response conforming to schema."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...
