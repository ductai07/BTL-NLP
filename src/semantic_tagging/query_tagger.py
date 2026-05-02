from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from semantic_tagging.llm_client import LLMClient, OpenAICompatibleClient
from semantic_tagging.semantic_tagger import empty_schema, normalize_tags, parse_json_object


PROMPT_TEMPLATE = """You are a semantic tagger for financial questions.

Given a financial question, extract semantic tags according to the same tag categories used for financial report chunks in the paper "Multi-Document Financial Question Answering using LLMs".

Return JSON only. Always include every field in the schema. Do not add extra fields.

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
- Always include every field.
- If a field has no value, return [].
- Only extract information explicitly mentioned in the question.
- Do not answer the question.
- Do not infer facts.
- Do not add fields outside the schema.
- Keep tags short and normalized.
- Laws, acts, regulations, and report types go in named_entities, not organizations.
- organizations only includes companies, regulators, exchanges, auditors, customers, suppliers, institutions, or government agencies.
- products includes product/service names or product categories explicitly mentioned.
- dates includes years, fiscal years, quarters, or explicit dates mentioned.
- Do not infer industries, domains, or sectors from company names. Only extract them if explicitly mentioned in the question.
- Do not put topic words such as "risks" or "tariffs" into named_entities unless they are named legal, product, company, or organization entities.

Question:
{question}
"""


class QueryTagger:
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

    def build_prompt(self, question: str) -> str:
        return PROMPT_TEMPLATE.replace("{question}", question)

    def tag_query(self, question: str) -> dict[str, list[str]]:
        if self.dry_run:
            return empty_schema()

        prompt = self.build_prompt(question)
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                raw = self.llm_client.generate(prompt)
                return self.clean_question_tags(normalize_tags(parse_json_object(raw)), question)
            except Exception as exc:
                last_error = exc
                self.logger.warning("query tag attempt %s failed: %s", attempt + 1, exc)

        self.logger.error("LLM query tagging failed: %s", last_error)
        return empty_schema()

    def clean_question_tags(self, tags: dict[str, list[str]], question: str) -> dict[str, list[str]]:
        lowered = question.lower()
        for field in ("industries", "domains", "sectors"):
            tags[field] = [value for value in tags[field] if value.lower() in lowered]
        tags["named_entities"] = [
            value for value in tags["named_entities"] if value.lower() not in {"risk", "risks", "tariff", "tariffs"}
        ]
        for organization in tags["organizations"]:
            if organization not in tags["named_entities"]:
                tags["named_entities"].append(organization)
        return tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag a financial question using paper semantic schema.")
    parser.add_argument("--question", required=True)
    parser.add_argument("--model")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    tags = QueryTagger(model=args.model, dry_run=args.dry_run).tag_query(args.question)
    print(json.dumps(tags, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
