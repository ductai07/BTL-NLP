from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


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
SCHEMA_FIELD_SET = set(SCHEMA_FIELDS)
TAG_CONTAINER_CANDIDATES = ("semantic_tags", "tags", "query_semantic_tags")
OLD_FORBIDDEN_FIELDS = {
    "chunk_role",
    "evidence_type",
    "section_tags",
    "financial_metrics",
    "business_topics",
    "risk_topics",
    "retrieval_keywords",
}

ORG_SUSPICIOUS_RE = re.compile(
    r"\b("
    r"form\s+10-[kq]|form\s+8-k|annual\s+report|quarterly\s+report|"
    r"private\s+securities\s+litigation\s+reform\s+act|"
    r"act|regulation|rule|law|standard"
    r")\b",
    re.IGNORECASE,
)
DATE_LIKE_RE = re.compile(
    r"\b(\d{4}|q[1-4]|fy\s?\d{2,4}|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
FORM_REPORT_RE = re.compile(r"\b(form\s+10-[kq]|form\s+8-k|annual\s+report|quarterly\s+report)\b", re.IGNORECASE)
ORG_MARKER_RE = re.compile(
    r"\b(inc\.?|corp\.?|corporation|company|llc|ltd\.?|plc|commission|exchange|llp|bank|audit|auditor)\b",
    re.IGNORECASE,
)


def load_jsonl_safe(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    parse_errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors.append({"line": line_no, "error": str(exc), "raw_preview": line[:300]})
                continue
            if not isinstance(value, dict):
                parse_errors.append({"line": line_no, "error": "JSONL record is not an object", "raw_preview": line[:300]})
                continue
            rows.append((line_no, value))
    return rows, parse_errors


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def find_tag_container(row: dict[str, Any]) -> tuple[str | None, Any]:
    present = [field for field in TAG_CONTAINER_CANDIDATES if field in row]
    if not present:
        return None, None
    for preferred in TAG_CONTAINER_CANDIDATES:
        if preferred in present:
            return preferred, row.get(preferred)
    return present[0], row.get(present[0])


def record_id(row: dict[str, Any]) -> str | None:
    for key in ("id", "chunk_id", "query_id"):
        if row.get(key) is not None:
            return str(row[key])
    return None


def text_preview(row: dict[str, Any], dataset_kind: str, limit: int = 320) -> str:
    keys = ("question", "query", "text", "embed_text") if dataset_kind == "queries" else ("text", "embed_text", "question", "query")
    text = ""
    for key in keys:
        if row.get(key):
            text = str(row[key])
            break
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def validate_tags(
    tags: Any,
    *,
    source_file: Path,
    line_no: int,
    row: dict[str, Any],
    tag_field: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    details = {
        "missing_fields": [],
        "extra_fields": [],
        "non_list_fields": [],
        "non_string_items": [],
        "duplicate_items": [],
    }

    def add_issue(issue_type: str, **extra: Any) -> None:
        issues.append(
            {
                "file": str(source_file),
                "line": line_no,
                "id": record_id(row),
                "tag_field": tag_field,
                "issue_type": issue_type,
                **extra,
            }
        )

    if tag_field is None:
        add_issue("missing_tag_container")
        return issues, details
    if not isinstance(tags, dict):
        add_issue("tag_container_not_object", actual_type=type(tags).__name__)
        return issues, details

    keys = set(tags)
    missing = sorted(SCHEMA_FIELD_SET - keys)
    extra = sorted(keys - SCHEMA_FIELD_SET)
    if missing:
        details["missing_fields"] = missing
        add_issue("missing_schema_fields", fields=missing)
    if extra:
        details["extra_fields"] = extra
        add_issue("extra_schema_fields", fields=extra)
    forbidden = sorted(keys & OLD_FORBIDDEN_FIELDS)
    if forbidden:
        add_issue("forbidden_old_schema_fields", fields=forbidden)

    for field in SCHEMA_FIELDS:
        if field not in tags:
            continue
        value = tags[field]
        if not isinstance(value, list):
            details["non_list_fields"].append(field)
            add_issue("tag_field_not_list", field=field, actual_type=type(value).__name__)
            continue
        seen: dict[str, int] = {}
        for index, item in enumerate(value):
            if not isinstance(item, str):
                details["non_string_items"].append({"field": field, "index": index, "actual_type": type(item).__name__})
                add_issue("tag_item_not_string", field=field, index=index, actual_type=type(item).__name__)
                continue
            key = item.strip().casefold()
            if key in seen:
                duplicate = {"field": field, "item": item, "first_index": seen[key], "index": index}
                details["duplicate_items"].append(duplicate)
                add_issue("duplicate_tag_in_field", **duplicate)
            else:
                seen[key] = index
    return issues, details


def suspicious_wrong_field(tags: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for field in SCHEMA_FIELDS:
        values = tags.get(field)
        if not isinstance(values, list):
            continue
        for value in values:
            if not isinstance(value, str):
                continue
            if field != "dates" and DATE_LIKE_RE.search(value):
                findings.append({"field": field, "tag": value, "reason": "date_like_tag_outside_dates"})
            if field == "organizations" and ORG_SUSPICIOUS_RE.search(value):
                findings.append({"field": field, "tag": value, "reason": "organization_contains_law_form_or_report_pattern"})
            if field in {"industries", "domains", "sectors", "products", "locations"} and FORM_REPORT_RE.search(value):
                findings.append({"field": field, "tag": value, "reason": "form_or_report_type_in_content_field"})
            if field not in {"organizations", "named_entities", "partners"} and ORG_MARKER_RE.search(value):
                findings.append({"field": field, "tag": value, "reason": "organization_like_tag_outside_organization_fields"})
    return findings


def audit_file(path: Path, dataset_kind: str) -> dict[str, Any]:
    rows, parse_errors = load_jsonl_safe(path)
    stats: dict[str, Any] = {
        "dataset_kind": dataset_kind,
        "file": str(path),
        "total_records": len(rows) + len(parse_errors),
        "parsed_records": len(rows),
        "parse_error_count": len(parse_errors),
        "parse_errors": parse_errors,
        "missing_tag_records": 0,
        "tag_field_counts": Counter(),
        "schema_issue_count": 0,
        "records_with_schema_issues": 0,
        "records_all_tags_empty": 0,
        "records_organizations_empty": 0,
        "records_named_entities_empty": 0,
        "field_empty_counts": Counter(),
        "field_value_counts": {field: 0 for field in SCHEMA_FIELDS},
        "top_tags": {},
        "duplicate_tag_count": 0,
        "wrong_field_suspicion_count": 0,
        "records_with_any": {field: 0 for field in SCHEMA_FIELDS},
    }
    counters: dict[str, Counter[str]] = {field: Counter() for field in SCHEMA_FIELDS}
    schema_issues: list[dict[str, Any]] = []
    suspicious_orgs: list[dict[str, Any]] = []
    wrong_field_samples: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for line_no, row in rows:
        tag_field, tags = find_tag_container(row)
        if tag_field is None:
            stats["missing_tag_records"] += 1
        else:
            stats["tag_field_counts"][tag_field] += 1

        issues, details = validate_tags(tags, source_file=path, line_no=line_no, row=row, tag_field=tag_field)
        if issues:
            stats["records_with_schema_issues"] += 1
            schema_issues.extend(issues)
        stats["schema_issue_count"] += len(issues)
        stats["duplicate_tag_count"] += len(details["duplicate_items"])

        sample_issue_types = [issue["issue_type"] for issue in issues]
        if len(sample_rows) < 5:
            sample_rows.append(
                {
                    "dataset_kind": dataset_kind,
                    "file": str(path),
                    "line": line_no,
                    "id": record_id(row),
                    "text_preview": text_preview(row, dataset_kind),
                    "tag_field": tag_field,
                    "semantic_tags": tags if isinstance(tags, dict) else None,
                    "schema_issues": sample_issue_types,
                }
            )

        if not isinstance(tags, dict):
            continue

        all_empty = True
        for field in SCHEMA_FIELDS:
            values = tags.get(field)
            if not isinstance(values, list):
                stats["field_empty_counts"][field] += 1
                continue
            string_values = [value for value in values if isinstance(value, str)]
            if string_values:
                all_empty = False
                stats["records_with_any"][field] += 1
                stats["field_value_counts"][field] += len(string_values)
                counters[field].update(string_values)
            else:
                stats["field_empty_counts"][field] += 1

        if all_empty:
            stats["records_all_tags_empty"] += 1
        if not isinstance(tags.get("organizations"), list) or not any(isinstance(v, str) for v in tags.get("organizations", [])):
            stats["records_organizations_empty"] += 1
        if not isinstance(tags.get("named_entities"), list) or not any(isinstance(v, str) for v in tags.get("named_entities", [])):
            stats["records_named_entities_empty"] += 1

        for org in tags.get("organizations", []) if isinstance(tags.get("organizations"), list) else []:
            if isinstance(org, str) and ORG_SUSPICIOUS_RE.search(org):
                suspicious_orgs.append(
                    {
                        "dataset_kind": dataset_kind,
                        "file": str(path),
                        "line": line_no,
                        "id": record_id(row),
                        "organization": org,
                        "text_preview": text_preview(row, dataset_kind),
                    }
                )

        wrong = suspicious_wrong_field(tags)
        stats["wrong_field_suspicion_count"] += len(wrong)
        if wrong and len(wrong_field_samples) < 100:
            wrong_field_samples.append(
                {
                    "dataset_kind": dataset_kind,
                    "file": str(path),
                    "line": line_no,
                    "id": record_id(row),
                    "findings": wrong[:10],
                    "text_preview": text_preview(row, dataset_kind),
                }
            )

    parsed = max(stats["parsed_records"], 1)
    stats["tag_field_counts"] = dict(stats["tag_field_counts"])
    stats["field_empty_counts"] = dict(stats["field_empty_counts"])
    stats["empty_tag_rate_by_field"] = {
        field: stats["field_empty_counts"].get(field, 0) / parsed for field in SCHEMA_FIELDS
    }
    stats["record_presence_rate_by_field"] = {
        field: stats["records_with_any"][field] / parsed for field in SCHEMA_FIELDS
    }
    stats["top_tags"] = {field: counters[field].most_common(25) for field in SCHEMA_FIELDS}
    stats["schema_clean_record_rate"] = (stats["parsed_records"] - stats["records_with_schema_issues"]) / parsed
    stats["all_tags_empty_rate"] = stats["records_all_tags_empty"] / parsed
    stats["organizations_empty_rate"] = stats["records_organizations_empty"] / parsed
    stats["named_entities_empty_rate"] = stats["records_named_entities_empty"] / parsed
    stats["suspicious_organization_count"] = len(suspicious_orgs)
    return {
        "stats": stats,
        "schema_issues": schema_issues,
        "suspicious_organizations": suspicious_orgs,
        "wrong_field_samples": wrong_field_samples,
        "samples": sample_rows,
    }


def candidate_score(path: Path, sample_limit: int = 50) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []
    rows, parse_errors = load_jsonl_safe(path)
    sample = rows[:sample_limit]
    tag_fields = Counter()
    has_text_like = 0
    has_question_like = 0
    schema_exact = 0
    old_schema = 0
    for _, row in sample:
        tag_field, tags = find_tag_container(row)
        if tag_field:
            tag_fields[tag_field] += 1
            score += 8
        if row.get("text") or row.get("embed_text"):
            has_text_like += 1
            score += 2
        if row.get("question") or row.get("query"):
            has_question_like += 1
            score += 3
        if isinstance(tags, dict):
            keys = set(tags)
            if keys == SCHEMA_FIELD_SET:
                schema_exact += 1
                score += 12
            if keys & OLD_FORBIDDEN_FIELDS:
                old_schema += 1
                score -= 15
    if tag_fields:
        reasons.append(f"tag containers found: {dict(tag_fields)}")
    if schema_exact:
        reasons.append(f"{schema_exact}/{len(sample)} sampled records match the 11-field schema exactly")
    if old_schema:
        reasons.append(f"{old_schema}/{len(sample)} sampled records include old/forbidden tag fields")
    if has_text_like:
        reasons.append(f"{has_text_like}/{len(sample)} sampled records look like chunks")
    if has_question_like:
        reasons.append(f"{has_question_like}/{len(sample)} sampled records look like queries")
    if parse_errors:
        reasons.append(f"{len(parse_errors)} parse errors while scanning")
        score -= len(parse_errors)
    return {
        "path": str(path),
        "score": score,
        "sampled_records": len(sample),
        "tag_fields": dict(tag_fields),
        "schema_exact_sample_count": schema_exact,
        "old_schema_sample_count": old_schema,
        "text_like_sample_count": has_text_like,
        "question_like_sample_count": has_question_like,
        "reasons": reasons,
    }


def discover_candidates(root: Path) -> dict[str, list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    for path in root.rglob("*.jsonl"):
        lowered = str(path).lower()
        if any(skip in lowered for skip in ("audit_schema_issues", "suspicious_organizations", "sample_tags")):
            continue
        score = candidate_score(path)
        if score["score"] > 0:
            candidates.append(score)
    chunk_candidates = [
        item
        for item in candidates
        if item["text_like_sample_count"] > 0 and any(item["tag_fields"].get(f, 0) for f in TAG_CONTAINER_CANDIDATES)
    ]
    query_candidates = [
        item
        for item in candidates
        if item["question_like_sample_count"] > 0 and any(item["tag_fields"].get(f, 0) for f in TAG_CONTAINER_CANDIDATES)
    ]
    chunk_candidates.sort(key=lambda item: (item["schema_exact_sample_count"], item["score"]), reverse=True)
    query_candidates.sort(key=lambda item: (item["schema_exact_sample_count"], item["score"]), reverse=True)
    return {"chunks": chunk_candidates, "queries": query_candidates}


def choose_path(explicit: Path | None, candidates: list[dict[str, Any]], label: str) -> Path:
    if explicit is not None:
        return explicit
    if not candidates:
        raise FileNotFoundError(f"No tagged {label} JSONL candidate found. Pass --{label} explicitly.")
    return Path(candidates[0]["path"])


def compare_query_chunk_tags(chunk_stats: dict[str, Any], query_stats: dict[str, Any]) -> dict[str, Any]:
    fields = {}
    for field in SCHEMA_FIELDS:
        fields[field] = {
            "chunk_presence_rate": chunk_stats["record_presence_rate_by_field"][field],
            "query_presence_rate": query_stats["record_presence_rate_by_field"][field],
            "chunk_empty_rate": chunk_stats["empty_tag_rate_by_field"][field],
            "query_empty_rate": query_stats["empty_tag_rate_by_field"][field],
        }
    focus = {}
    for field in ("organizations", "dates", "products", "locations"):
        focus[field] = {
            "chunk_has_at_least_one_rate": chunk_stats["record_presence_rate_by_field"][field],
            "query_has_at_least_one_rate": query_stats["record_presence_rate_by_field"][field],
        }
    return {"field_presence_comparison": fields, "focus_presence_rates": focus}


def final_status(summary: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    datasets = [summary["chunks"]["stats"], summary["queries"]["stats"]]
    fail = False
    warn = False
    for stats in datasets:
        parsed = max(stats["parsed_records"], 1)
        if stats["parse_error_count"] > 0:
            fail = True
            reasons.append(f"{stats['dataset_kind']} has parse errors: {stats['parse_error_count']}")
        if stats["missing_tag_records"] / parsed > 0.10:
            fail = True
            reasons.append(f"{stats['dataset_kind']} has many records missing tag containers")
        if stats["schema_clean_record_rate"] < 0.80:
            fail = True
            reasons.append(f"{stats['dataset_kind']} schema clean rate is below 80%")
        if stats["schema_issue_count"] and stats["schema_clean_record_rate"] < 0.95:
            warn = True
            reasons.append(f"{stats['dataset_kind']} schema clean rate is below 95%")
        if stats["all_tags_empty_rate"] > 0.30:
            warn = True
            reasons.append(f"{stats['dataset_kind']} has many all-empty tag records")
        if stats["suspicious_organization_count"] > 0:
            warn = True
            reasons.append(f"{stats['dataset_kind']} has suspicious organization tags")
    if fail:
        return "FAIL", reasons
    if warn:
        return "WARN", reasons
    return "PASS", ["Most records match the 11-field schema and serious type/container issues are low."]


def print_candidates(candidates: dict[str, list[dict[str, Any]]], selected_chunks: Path, selected_queries: Path) -> None:
    for label in ("chunks", "queries"):
        print(f"\n{label.upper()} candidates:")
        for item in candidates[label][:8]:
            selected = " SELECTED" if Path(item["path"]) == (selected_chunks if label == "chunks" else selected_queries) else ""
            print(f"- {item['path']}{selected}")
            print(f"  score={item['score']} sampled={item['sampled_records']}")
            for reason in item["reasons"][:4]:
                print(f"  reason: {reason}")


def print_samples(title: str, samples: list[dict[str, Any]]) -> None:
    print(f"\n{title}:")
    for sample in samples[:5]:
        print(json.dumps(sample, ensure_ascii=False, indent=2))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Audit existing paper-schema semantic tags without retagging data.")
    parser.add_argument("--chunks", type=Path, help="Tagged chunks JSONL. Auto-detected when omitted.")
    parser.add_argument("--queries", type=Path, help="Tagged queries/questions JSONL. Auto-detected when omitted.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_tag_audit"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    candidates = discover_candidates(project_root)
    chunks_path = choose_path(args.chunks, candidates["chunks"], "chunks")
    queries_path = choose_path(args.queries, candidates["queries"], "queries")
    if not chunks_path.is_absolute():
        chunks_path = (project_root / chunks_path).resolve()
    if not queries_path.is_absolute():
        queries_path = (project_root / queries_path).resolve()

    print_candidates(candidates, selected_chunks=chunks_path, selected_queries=queries_path)
    print(f"\nSelected chunks file: {chunks_path}")
    print(f"Selected queries file: {queries_path}")

    chunk_result = audit_file(chunks_path, "chunks")
    query_result = audit_file(queries_path, "queries")
    comparison = compare_query_chunk_tags(chunk_result["stats"], query_result["stats"])

    summary = {
        "schema_fields": SCHEMA_FIELDS,
        "forbidden_old_fields": sorted(OLD_FORBIDDEN_FIELDS),
        "selected_files": {"chunks": str(chunks_path), "queries": str(queries_path)},
        "candidate_files": candidates,
        "chunks": {"stats": chunk_result["stats"]},
        "queries": {"stats": query_result["stats"]},
        "query_vs_chunk_comparison": comparison,
        "wrong_field_suspicion_samples": chunk_result["wrong_field_samples"] + query_result["wrong_field_samples"],
    }
    status, reasons = final_status(summary)
    summary["final_status"] = status
    summary["final_status_reasons"] = reasons

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    schema_issues = chunk_result["schema_issues"] + query_result["schema_issues"]
    suspicious_orgs = chunk_result["suspicious_organizations"] + query_result["suspicious_organizations"]
    samples = chunk_result["samples"] + query_result["samples"]

    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=json_default),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "audit_schema_issues.jsonl", schema_issues)
    write_jsonl(output_dir / "suspicious_organizations.jsonl", suspicious_orgs)
    write_jsonl(output_dir / "sample_tags.jsonl", samples)

    print_samples("Sample chunks", chunk_result["samples"])
    print_samples("Sample queries", query_result["samples"])

    print("\nAudit report files:")
    print(f"- {output_dir / 'audit_summary.json'}")
    print(f"- {output_dir / 'audit_schema_issues.jsonl'}")
    print(f"- {output_dir / 'suspicious_organizations.jsonl'}")
    print(f"- {output_dir / 'sample_tags.jsonl'}")

    print(f"\nCONCLUSION: {status}")
    for reason in reasons:
        print(f"- {reason}")


if __name__ == "__main__":
    main()
