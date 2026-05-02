from __future__ import annotations

from collections import defaultdict
from typing import Any


def _norm_set(values: Any) -> set[str]:
    if not isinstance(values, list):
        return set()
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _overlap(query_tags: dict[str, Any], chunk_tags: dict[str, Any], query_field: str, chunk_field: str | None = None) -> float:
    chunk_field = chunk_field or query_field
    query_values = _norm_set(query_tags.get(query_field))
    chunk_values = _norm_set(chunk_tags.get(chunk_field))
    if not query_values or not chunk_values:
        return 0.0
    return len(query_values & chunk_values) / len(query_values)


def tag_overlap_score(query_tags: dict[str, Any], chunk_tags: dict[str, Any]) -> float:
    chunk_role = str(chunk_tags.get("chunk_role", "unknown")).lower()
    is_navigation_only = bool(chunk_tags.get("is_navigation_only", False))
    metadata_roles = {"cover_page", "table_of_contents", "signature", "exhibit_list", "metadata"}
    evidence_roles = {"section_content", "financial_statement", "footnote", "risk_factor", "mdna"}

    if chunk_role in metadata_roles:
        return 0.0

    weighted_parts = [
        (0.28, _overlap(query_tags, chunk_tags, "financial_metrics")),
        (0.18, _overlap(query_tags, chunk_tags, "business_topics")),
        (0.15, _overlap(query_tags, chunk_tags, "risk_topics")),
        (0.18, _overlap(query_tags, chunk_tags, "expected_sections", "section_tags")),
        (0.12, _overlap(query_tags, chunk_tags, "retrieval_keywords")),
    ]
    score = sum(weight * value for weight, value in weighted_parts)

    if _overlap(query_tags, chunk_tags, "companies") > 0:
        score += 0.08
    if _overlap(query_tags, chunk_tags, "years") > 0:
        score += 0.06

    evidence_needed = str(query_tags.get("evidence_type_needed", "unknown")).lower()
    evidence_type = str(chunk_tags.get("evidence_type", "unknown")).lower()
    if evidence_needed == evidence_type and evidence_needed != "unknown":
        score += 0.08
    elif evidence_needed == "table" and evidence_type == "mixed":
        score += 0.06

    intent = str(query_tags.get("intent", "unknown")).lower()
    if intent in {"calculate", "extract_value"}:
        if evidence_type in {"table", "mixed"}:
            score += 0.08
        if _overlap(query_tags, chunk_tags, "financial_metrics") > 0:
            score += 0.08

    if chunk_role in evidence_roles:
        score += 0.05
    if is_navigation_only:
        score *= 0.25

    return min(max(score, 0.0), 1.0)


def combine_scores(dense_score: float, bm25_score: float, tag_score: float, alpha: float = 0.5, beta: float = 0.2) -> float:
    final_score = alpha * dense_score + (1 - alpha) * bm25_score + beta * tag_score
    return final_score


def rrf_fusion(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    query_tags: dict[str, Any],
    k: int = 60,
    tag_weight: float = 0.2,
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    scores: dict[str, float] = defaultdict(float)

    for rank, row in enumerate(bm25_results, start=1):
        chunk_id = str(row["chunk_id"])
        by_id.setdefault(chunk_id, dict(row))
        scores[chunk_id] += 1.0 / (k + rank)

    for rank, row in enumerate(dense_results, start=1):
        chunk_id = str(row["chunk_id"])
        by_id.setdefault(chunk_id, dict(row))
        scores[chunk_id] += 1.0 / (k + rank)

    fused: list[dict[str, Any]] = []
    for chunk_id, row in by_id.items():
        tag_score = tag_overlap_score(query_tags, row.get("semantic_tags") or {})
        final_score = scores[chunk_id] + tag_weight * tag_score
        out = dict(row)
        out["rrf_score"] = scores[chunk_id]
        out["tag_score"] = tag_score
        out["final_score"] = final_score
        fused.append(out)

    return sorted(fused, key=lambda item: item["final_score"], reverse=True)
