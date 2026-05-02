from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env", encoding="utf-8-sig")


@dataclass(frozen=True)
class LLMConfig:
    provider: str = os.environ.get("SEMANTIC_TAGGER_PROVIDER", "openai")
    model: str = os.environ.get("SEMANTIC_TAGGER_MODEL", "gpt-4o-mini")
    api_key: str | None = os.environ.get("OPENAI_API_KEY")
    nvidia_api_key: str | None = os.environ.get("NVIDIA_API_KEY")
    base_url: str = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    nvidia_base_url: str = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    timeout: int = int(os.environ.get("SEMANTIC_TAGGER_TIMEOUT", "60"))
    temperature: float = float(os.environ.get("SEMANTIC_TAGGER_TEMPERATURE", "0"))
    max_tokens: int = int(os.environ.get("SEMANTIC_TAGGER_MAX_TOKENS", "700"))
    dry_run: bool = os.environ.get("SEMANTIC_TAGGER_DRY_RUN", "").lower() in {"1", "true", "yes"}


DEFAULT_MAX_RETRIES = 2
