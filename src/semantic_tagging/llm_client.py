from __future__ import annotations

from abc import ABC, abstractmethod
import os
from pathlib import Path
from typing import Any

import requests

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", encoding="utf-8-sig")


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        max_tokens: int = 3000,
        temperature: float = 0.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

    def generate(self, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        if "gpt-oss" in self.model:
            payload["reasoning_effort"] = "low"
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(f"LLM request failed: {response.status_code} {response.text[:1000]}") from exc
        result = response.json()
        message = result["choices"][0]["message"]
        content = message.get("content")
        if content is None:
            raise RuntimeError("LLM returned empty content.")
        return content
