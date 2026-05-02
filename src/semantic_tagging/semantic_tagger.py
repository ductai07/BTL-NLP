from __future__ import annotations

import json
import logging
import re
from typing import Any

from .llm_client import LLMClient, OpenAICompatibleClient


SCHEMA_FIELDS = [
    "named_entities",
    "dates",
    "industries",
    "domains",
    "sectors",
    "organizations",
    "partnerships",
    "partners",
    "dividends",
    "products",
    "locations",
]

EMPTY_TAGS = {field: [] for field in SCHEMA_FIELDS}
PLACEHOLDERS = {"", "...", "etc.", "etc", "and more", "various", "other", "others"}

PROMPT_TEMPLATE = """You are a semantic tagger for financial reports.

Given metadata and a text chunk from a financial filing, extract semantic tags according to the tag categories described in the paper "Multi-Document Financial Question Answering using LLMs".

Return JSON only. Always include every field in the schema. Do not add any extra fields.

Schema:
{
  "named_entities": [],
  "dates": [],
  "industries": [],
  "domains": [],
  "sectors": [],
  "organizations": [],
  "partnerships": [],
  "partners": [],
  "dividends": [],
  "products": [],
  "locations": []
}

Rules:
- Return JSON only. No markdown. No explanations.
- Always include every field from the schema.
- If a field has no value, return [].
- Only extract information explicitly present in the metadata or chunk text.
- Do not infer facts.
- Do not answer questions.
- Do not summarize.
- Do not add any field outside the schema.
- Do not use placeholders such as "...", "etc.", "and more", "various", "other".
- Keep tags short and normalized.
- Preserve official names for companies, products, organizations, laws, and locations.
- A tag can appear in more than one field only when appropriate. For example, "Apple Inc." can be in named_entities; "Nasdaq" can be in named_entities and organizations.
- Do not create financial metric tags unless they fit one of the schema fields. For example, do not create "financial_metrics".
- Do not create section/content tags. For example, do not create "section_tags", "chunk_role", or "evidence_type".
- Do not put products or service categories into industries, domains, or sectors unless the chunk explicitly describes them as an industry, domain, or sector.
- Do not put laws, statutes, report names, filing forms, or filing types into organizations.
- "Private Securities Litigation Reform Act of 1995" belongs in named_entities only, not organizations.
- "Form 10-K" and "Annual Report on Form 10-K" belong in named_entities only, not organizations.

Field guidance:
- named_entities: named companies, laws, reports, legal entities, named organizations, named people, named products, or named business units explicitly mentioned.
- dates: explicit dates, years, fiscal years, quarters, or periods explicitly mentioned.
- industries: explicitly mentioned industries.
- domains: explicitly mentioned domains or business/technology domains.
- sectors: explicitly mentioned sectors.
- organizations: explicitly mentioned companies, organizations, regulators, exchanges, auditors, customers, suppliers, or institutions. Do not include laws, statutes, report names, or filing forms.
- partnerships: explicitly mentioned partnership relationships.
- partners: explicitly mentioned partner entities.
- dividends: explicitly mentioned dividend-related terms or dividend facts.
- products: explicitly mentioned products or services.
- locations: explicitly mentioned geographic locations.

Metadata:
{metadata_json}

Chunk text:
{chunk_text}
"""


class SemanticTagger:
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        model: str | None = None,
        max_retries: int = 2,
        dry_run: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.llm_client = llm_client or OpenAICompatibleClient(model=model)
        self.max_retries = max_retries
        self.dry_run = dry_run
        self.logger = logger or logging.getLogger(__name__)

    def build_prompt(self, chunk: dict[str, Any]) -> str:
        metadata = {key: value for key, value in chunk.items() if key not in {"text", "embed_text"}}
        chunk_text = chunk.get("text") or chunk.get("embed_text") or ""
        return (
            PROMPT_TEMPLATE.replace("{metadata_json}", json.dumps(metadata, ensure_ascii=False, indent=2))
            .replace("{chunk_text}", str(chunk_text))
        )

    def tag_chunk(self, chunk: dict[str, Any]) -> dict[str, list[str]]:
        if self.dry_run:
            return empty_schema()

        prompt = self.build_prompt(chunk)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.llm_client.generate(prompt)
                return normalize_tags(parse_json_object(raw))
            except Exception as exc:
                last_error = exc
                self.logger.warning(
                    "semantic tag attempt %s failed for %s: %s",
                    attempt + 1,
                    chunk.get("chunk_id") or chunk.get("id"),
                    exc,
                )

        self.logger.error("LLM semantic tagging failed for %s: %s", chunk.get("chunk_id") or chunk.get("id"), last_error)
        return empty_schema()


def empty_schema() -> dict[str, list[str]]:
    return {field: [] for field in SCHEMA_FIELDS}


def normalize_tags(tags: dict[str, Any]) -> dict[str, list[str]]:
    normalized = empty_schema()
    for field in SCHEMA_FIELDS:
        value = tags.get(field)
        if value is None:
            values: list[Any] = []
        elif isinstance(value, list):
            values = value
        else:
            values = [value]
        normalized[field] = clean_list(values)
    product_keys = {value.lower() for value in normalized["products"]}
    for field in ("industries", "domains", "sectors"):
        normalized[field] = [value for value in normalized[field] if value.lower() not in product_keys]
    normalized["organizations"] = [
        value for value in normalized["organizations"] if not looks_like_non_organization(value)
    ]
    return normalized


def clean_list(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = re.sub(r"\s+", " ", str(value)).strip()
        key = item.lower()
        if key in PLACEHOLDERS or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def looks_like_non_organization(value: str) -> bool:
    lowered = value.lower()
    return any(
        marker in lowered
        for marker in (
            "act of ",
            "reform act",
            "securities litigation",
            "form 10-k",
            "form 10-q",
            "form 8-k",
            "annual report",
            "quarterly report",
            "proxy statement",
        )
    )


def parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM output is not a JSON object.")
    return value
