import json
from openai import OpenAI
from llm.provider import LLMProvider
from config.settings import settings


class OpenAIProvider(LLMProvider):

    def __init__(self):
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)

    @property
    def model_name(self) -> str:
        return settings.OPENAI_MODEL

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4096,
        )
        return response.choices[0].message.content

    def complete_structured(self, system_prompt: str, user_prompt: str, schema: dict) -> dict:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_tokens=8096,
        )
        return json.loads(response.choices[0].message.content)
