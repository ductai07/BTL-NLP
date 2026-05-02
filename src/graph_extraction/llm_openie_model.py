from __future__ import annotations

import ast
import json
import logging
import re
from itertools import chain
from typing import Any

import numpy as np

from semantic_tagging.llm_client import LLMClient, OpenAICompatibleClient

from .base_openie_model import BaseOPENIEModel
from .prompts import NER_PROMPT_TEMPLATE, OPENIE_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)
CHANGE_KEYWORDS = (
    "increased",
    "increase",
    "decreased",
    "decrease",
    "reduced",
    "declined",
    "grew",
    "rose",
    "fell",
    "improved",
    "worsened",
)

CANONICAL_RELATIONS = {
    "located in": "located_in",
    "is located in": "located_in",
    "headquartered in": "located_in",
    "has trading symbol": "has_ticker",
    "trading symbol": "has_ticker",
    "ticker": "has_ticker",
    "is listed on": "listed_on",
    "listed on": "listed_on",
    "is registered with": "listed_on",
    "registered with": "listed_on",
    "files": "files_report",
    "filed": "files_report",
    "filed form": "files_report",
    "has address": "has_address",
    "address": "has_address",
    "has phone": "has_phone",
    "offers": "offers",
    "owns": "owns",
    "acquired": "acquired",
    "partnered with": "partnered_with",
    "partners with": "partnered_with",
    "is": "is",
    "registered": "listed_on",
    "has_a_telephone_number": "has_phone",
    "has_telephone_number": "has_phone",
    "has_a_principal_executive_office_at": "has_address",
    "principal_executive_office": "has_address",
    "has_filed_all_reports": "files_report",
    "has_submitted_electronically_every_interactive_data_file": "files_report",
    "has_filed_a_report_on_and_attestation_to_its_managements_assessment_of_the_effectiveness_of_its_internal_control_over_financial_reporting": "files_report",
    "is_a_large_accelerated_filer": "filer_status",
    "is_a_shell_company": "shell_company_status",
    "increased": "increased",
    "increase": "increased",
    "grew": "increased",
    "rose": "increased",
    "expanded": "increased",
    "decreased": "decreased",
    "decrease": "decreased",
    "declined": "decreased",
    "fell": "decreased",
    "reduced": "decreased",
    "dropped": "decreased",
    "improved": "improved",
    "worsened": "worsened",
    "reported": "reported",
    "recorded": "reported",
    "generated": "generated",
    "incurred": "incurred",
    "guidance_for": "guidance_for",
    "expects": "expects",
    "projects": "projects",
    "filed_form": "files_report",
    "filed_with": "files_report",
    "merged_with": "merged_with",
    "invested_in": "invested_in",
}


class LLMOPENIEModel(BaseOPENIEModel):
    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        model: str | None = None,
        max_retries: int = 2,
        enable_delta_heuristic: bool = False,
    ) -> None:
        self.client = llm_client or OpenAICompatibleClient(model=model, max_tokens=2000)
        self.max_retries = max_retries
        self.enable_delta_heuristic = enable_delta_heuristic

    def ner(self, text: str) -> list[str]:
        prompt = NER_PROMPT_TEMPLATE.format(text=text)
        content = self._generate_with_retry(prompt)
        if content is None:
            return []
        try:
            data = parse_json_dict(content)
            values = data.get("named_entities", [])
            if not isinstance(values, list):
                return []
            return clean_str_list(values)
        except Exception as exc:
            logger.error("Loi parse named entities: %s", exc)
            return []

    def openie_post_ner_extract(self, text: str, entities: list[str]) -> dict[str, list[list[str]]]:
        named_entity_json = {"named_entities": entities}
        prompt = OPENIE_PROMPT_TEMPLATE.format(
            text=text,
            entities_json=json.dumps(named_entity_json, ensure_ascii=False),
        )
        content = self._generate_with_retry(prompt)
        if content is None:
            return {"triples": []}
        try:
            data = parse_json_dict(content)
            triples = data.get("triples", [])
            if not isinstance(triples, list):
                return {"triples": []}
            norm_triples: list[list[str]] = []
            for triple in triples:
                if not isinstance(triple, list) or len(triple) != 3:
                    continue
                s, p, o = (str(triple[0]).strip(), str(triple[1]).strip(), str(triple[2]).strip())
                if not s or not p or not o:
                    continue
                norm_triples.append([s, p, o])
            return {"triples": dedupe_triples(norm_triples)}
        except Exception as exc:
            logger.error("Loi parse triples: %s", exc)
            return {"triples": []}

    def __call__(self, text: str) -> dict[str, Any]:
        focused_text = select_focus_passage(text)
        res: dict[str, Any] = {
            "passage": focused_text,
            "extracted_entities": [],
            "extracted_triples": [],
            "clean_triples": [],
            "noisy_triples": [],
        }
        doc_entities = self.ner(focused_text)
        try:
            doc_entities = list(np.unique(doc_entities))
        except Exception as exc:
            logger.error("Loi unique entities: %s", exc)
            doc_entities = list(np.unique(list(chain.from_iterable(doc_entities))))  # type: ignore[arg-type]
        triples_data = self.openie_post_ner_extract(focused_text, doc_entities)
        clean_triples, noisy_triples = filter_and_normalize_triples(
            triples=triples_data.get("triples", []),
            entities=doc_entities,
            passage=focused_text,
        )
        if self.enable_delta_heuristic and len(clean_triples) <= 2:
            # Heuristic chỉ hỗ trợ khi output LLM quá nghèo thông tin.
            heuristic_delta_triples = extract_delta_triples_from_passage(focused_text, doc_entities)
            clean_triples = dedupe_triples(clean_triples + heuristic_delta_triples)
        res["extracted_entities"] = doc_entities
        res["extracted_triples"] = clean_triples
        res["clean_triples"] = clean_triples
        res["noisy_triples"] = noisy_triples
        return res

    def _generate_with_retry(self, prompt: str) -> str | None:
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                return self.client.generate(prompt)
            except Exception as exc:
                last_error = exc
        logger.error("LLM request failed: %s", last_error)
        return None


def parse_json_dict(raw: str) -> dict[str, Any]:
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
        body = match.group(0)
        try:
            value = json.loads(body)
        except json.JSONDecodeError:
            value = ast.literal_eval(body)
    if not isinstance(value, dict):
        raise ValueError("Output cua LLM khong phai JSON object")
    return value


def clean_str_list(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        item = re.sub(r"\s+", " ", str(value)).strip()
        if not item:
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(item)
    return cleaned


def dedupe_triples(triples: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[list[str]] = []
    for s, p, o in triples:
        key = (s.lower(), p.lower(), o.lower())
        if key in seen:
            continue
        seen.add(key)
        unique.append([s, p, o])
    return unique


def normalize_relation_name(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    normalized = re.sub(r"[^\w\s_]", "", normalized)
    if normalized.startswith("has_"):
        normalized = normalized.replace("_", " ")
    if normalized in CANONICAL_RELATIONS:
        return CANONICAL_RELATIONS[normalized]
    if "telephone" in normalized or "phone" in normalized:
        return "has_phone"
    if "address" in normalized or "office" in normalized:
        return "has_address"
    if "filed" in normalized or "report" in normalized:
        return "files_report"
    if "registered" in normalized or "listed" in normalized or "exchange" in normalized:
        return "listed_on"
    if any(token in normalized for token in ("increase", "grew", "rose", "expanded")):
        return "increased"
    if any(token in normalized for token in ("decrease", "declined", "fell", "reduced", "dropped")):
        return "decreased"
    if "improved" in normalized:
        return "improved"
    if "worsened" in normalized:
        return "worsened"
    if any(token in normalized for token in ("reported", "recorded")):
        return "reported"
    if "guidance" in normalized:
        return "guidance_for"
    if "expects" in normalized or "expect" in normalized:
        return "expects"
    if "projects" in normalized or "projected" in normalized:
        return "projects"
    normalized = normalized.replace(" ", "_")
    if len(normalized) > 36:
        return "related_to"
    if not normalized:
        return "related_to"
    return normalized


def phrase_in_passage(phrase: str, passage_lower: str) -> bool:
    cleaned = re.sub(r"\s+", " ", phrase.strip().lower())
    if not cleaned:
        return False
    if cleaned in passage_lower:
        return True
    compact = re.sub(r"[^\w\s]", "", cleaned)
    passage_compact = re.sub(r"[^\w\s]", "", passage_lower)
    return compact in passage_compact


def filter_and_normalize_triples(
    *,
    triples: list[list[str]],
    entities: list[str],
    passage: str,
) -> tuple[list[list[str]], list[dict[str, Any]]]:
    entity_keys = {re.sub(r"\s+", " ", str(ent).strip().lower()) for ent in entities}
    passage_lower = str(passage).lower()
    clean: list[list[str]] = []
    noisy: list[dict[str, Any]] = []

    for triple in triples:
        if not isinstance(triple, list) or len(triple) != 3:
            noisy.append({"triple": triple, "reason": "invalid_triple_shape"})
            continue
        subject = str(triple[0]).strip()
        relation_raw = str(triple[1]).strip()
        obj = str(triple[2]).strip()
        if not subject or not relation_raw or not obj:
            noisy.append({"triple": triple, "reason": "empty_element"})
            continue

        relation = normalize_relation_name(relation_raw)
        subject_key = re.sub(r"\s+", " ", subject.lower())
        object_key = re.sub(r"\s+", " ", obj.lower())
        subj_in_entities = subject_key in entity_keys
        obj_in_entities = object_key in entity_keys

        subj_in_passage = phrase_in_passage(subject, passage_lower)
        obj_in_passage = phrase_in_passage(obj, passage_lower)
        if not subj_in_passage or not obj_in_passage:
            noisy.append(
                {
                    "triple": [subject, relation, obj],
                    "reason": "entity_not_grounded_in_passage",
                    "subject_in_passage": subj_in_passage,
                    "object_in_passage": obj_in_passage,
                }
            )
            continue

        if not (subj_in_entities or obj_in_entities):
            noisy.append(
                {
                    "triple": [subject, relation, obj],
                    "reason": "neither_subject_nor_object_in_ner_entities",
                }
            )
            continue

        clean.append([subject, relation, obj])

    return dedupe_triples(clean), noisy


def select_focus_passage(text: str, max_chars: int = 6000) -> str:
    raw = re.sub(r"\s+", " ", str(text)).strip()
    if len(raw) <= max_chars:
        return raw

    sentences = re.split(r"(?<=[.!?])\s+", raw)
    if len(sentences) <= 1:
        return raw[:max_chars]

    picked: list[str] = []
    for idx, sentence in enumerate(sentences):
        lower = sentence.lower()
        if any(keyword in lower for keyword in CHANGE_KEYWORDS):
            left = sentences[idx - 1] if idx - 1 >= 0 else ""
            right = sentences[idx + 1] if idx + 1 < len(sentences) else ""
            block = " ".join(part for part in (left, sentence, right) if part)
            picked.append(block)

    if not picked:
        picked = sentences[: min(12, len(sentences))]

    merged = " ".join(picked)
    merged = re.sub(r"\s+", " ", merged).strip()
    return merged[:max_chars]


def extract_delta_triples_from_passage(passage: str, entities: list[str]) -> list[list[str]]:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", passage).strip())
    if not sentences:
        return []
    relation_map = {
        "increased": "increased",
        "increase": "increased",
        "grew": "increased",
        "rose": "increased",
        "expanded": "increased",
        "decreased": "decreased",
        "decrease": "decreased",
        "declined": "decreased",
        "fell": "decreased",
        "reduced": "decreased",
        "dropped": "decreased",
        "improved": "improved",
        "worsened": "worsened",
    }

    triples: list[list[str]] = []
    entities_sorted = sorted((str(e).strip() for e in entities if str(e).strip()), key=len, reverse=True)
    for sentence in sentences:
        sentence_lower = sentence.lower()
        matched_keyword = None
        for keyword in relation_map:
            if re.search(rf"\b{re.escape(keyword)}\b", sentence_lower):
                matched_keyword = keyword
                break
        if matched_keyword is None:
            continue

        relation = relation_map[matched_keyword]
        subject = None
        for entity in entities_sorted:
            if entity.lower() in sentence_lower:
                subject = entity
                break
        if subject is None and entities_sorted:
            subject = entities_sorted[0]
        if subject is None:
            continue

        obj = sentence.strip()
        if len(obj) > 240:
            obj = obj[:237] + "..."
        triples.append([subject, relation, obj])
    return triples
