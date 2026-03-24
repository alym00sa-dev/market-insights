import json
import anthropic
from llm.provider import LLMProvider
from config.settings import settings


class ClaudeProvider(LLMProvider):

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    @property
    def model_name(self) -> str:
        return settings.CLAUDE_MODEL

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        message = self._client.messages.create(
            model=self.model_name,
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text

    def complete_structured(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        full_system = (
            f"{system_prompt}\n\n"
            f"You must respond with valid JSON only, conforming to this schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )
        message = self._client.messages.create(
            model=self.model_name,
            max_tokens=8096,
            system=full_system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{") or part.startswith("["):
                    raw = part
                    break
        return json.loads(raw.strip())
