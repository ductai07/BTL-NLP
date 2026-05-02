from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_METHODS = [
    ("BM25", Path("outputs/retrieval_benchmark_report_bm25/metrics_summary.json")),
    ("Text-only BGE", Path("outputs/retrieval_benchmark_report_text_only/metrics_summary.json")),
    ("Dense BGE", Path("outputs/retrieval_benchmark_report_all/metrics_summary.json")),
    ("Proposed", Path("outputs/retrieval_benchmark_report_method/metrics_summary.json")),
    ("Proposed + Tag Boost", Path("outputs/retrieval_benchmark_report_method/metrics_summary.json")),
    ("Proposed + Tag Filter", Path("outputs/retrieval_benchmark_report_method/metrics_summary.json")),
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.3f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize retrieval method metrics into report tables.")
    parser.add_argument("--output", type=Path, default=Path("outputs/retrieval_benchmark_report_summary.md"))
    args = parser.parse_args()

    summaries = []
    for name, path in DEFAULT_METHODS:
        if not path.exists():
            continue
        summaries.append((name, load_json(path)))

    method_rows = []
    for name, summary in summaries:
        metrics = summary["metrics"]["all"]
        if name == "Proposed + Tag Boost":
            if not summary.get("tag_aware"):
                continue
            method_rows.append([
                name,
                fmt(metrics.get("tag_boost_recall@5")),
                fmt(metrics.get("tag_boost_recall@10")),
                fmt(metrics.get("tag_boost_mrr@10")),
            ])
        elif name == "Proposed + Tag Filter":
            if not summary.get("tag_aware"):
                continue
            method_rows.append([
                name,
                fmt(metrics.get("tag_filter_recall@5")),
                fmt(metrics.get("tag_filter_recall@10")),
                fmt(metrics.get("tag_filter_mrr@10")),
            ])
        else:
            method_rows.append([
                name,
                fmt(metrics.get("recall@5")),
                fmt(metrics.get("recall@10")),
                fmt(metrics.get("mrr@10")),
            ])

    type_rows = []
    proposed = next((summary for name, summary in summaries if name == "Proposed"), None)
    if proposed:
        for qtype in ("text", "table", "image", "multimodal", "all"):
            metrics = proposed["metrics"].get(qtype)
            if not metrics:
                continue
            type_rows.append(
                [
                    qtype,
                    str(metrics.get("queries", "")),
                    fmt(metrics.get("recall@5")),
                    fmt(metrics.get("recall@10")),
                    fmt(metrics.get("mrr@10")),
                ]
            )

    content = "\n\n".join(
        [
            "# Retrieval Benchmark Summary",
            "## Method Comparison",
            markdown_table(["Method", "Recall@5", "Recall@10", "MRR@10"], method_rows),
            "## Proposed Method by Query Type",
            markdown_table(["Type", "#Queries", "Recall@5", "Recall@10", "MRR@10"], type_rows),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
