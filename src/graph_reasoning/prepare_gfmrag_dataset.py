from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_GRAPH_DIR = Path("data/graph")
DEFAULT_QUERIES = Path("data/benchmark_report/queries_tagged.jsonl")
DEFAULT_OUTPUT_ROOT = Path("data/gfmrag_reasoning")
DEFAULT_DATA_NAME = "multifin_graph"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_attributes(value: Any) -> dict[str, Any]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return {}
    if isinstance(value, dict):
        return value
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def normalize_entity_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip().lower()
    text = re.sub(r"[^\w\s&.-]", "", text)
    return text


def load_node_indexes(nodes_csv: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    nodes_df = pd.read_csv(nodes_csv, keep_default_na=False)
    nodes_df["attributes"] = nodes_df["attributes"].apply(parse_attributes)

    node_attrs: dict[str, dict[str, Any]] = {}
    entity_by_norm: dict[str, list[str]] = defaultdict(list)
    for row in nodes_df.to_dict(orient="records"):
        uid = str(row["uid"])
        attrs = dict(row["attributes"])
        attrs["name"] = row.get("name")
        attrs["type"] = row.get("type")
        node_attrs[uid] = attrs
        if row.get("type") == "entity":
            name = str(row.get("name") or "")
            entity_by_norm[normalize_entity_name(name)].append(uid)
    return node_attrs, dict(entity_by_norm)


def query_terms(item: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    tags = item.get("query_semantic_tags")
    if isinstance(tags, dict):
        for value in tags.values():
            if isinstance(value, list):
                terms.extend(str(v) for v in value if str(v).strip())
            elif value:
                terms.append(str(value))
    for key in ("ticker", "company", "organization"):
        if item.get(key):
            terms.append(str(item[key]))
    return list(dict.fromkeys(terms))


def resolve_start_entities(
    item: dict[str, Any],
    entity_by_norm: dict[str, list[str]],
) -> list[str]:
    starts: list[str] = []
    for term in query_terms(item):
        norm = normalize_entity_name(term)
        if norm in entity_by_norm:
            starts.extend(entity_by_norm[norm])
            continue

        # Conservative fuzzy fallback for cases like "Apple" vs "Apple Inc.".
        if len(norm) >= 3:
            for entity_norm, entity_ids in entity_by_norm.items():
                if norm == entity_norm or norm in entity_norm.split():
                    starts.extend(entity_ids)
    return sorted(set(starts))


def evidence_chunk_ids(item: dict[str, Any]) -> list[str]:
    raw_ids = item.get("evidence_chunk_ids") or item.get("chunk_ids") or []
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    return [f"chunk:{cid}" for cid in raw_ids if str(cid).strip()]


def build_documents(node_attrs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    docs: dict[str, dict[str, Any]] = {}
    for uid, attrs in node_attrs.items():
        if attrs.get("type") == "chunk":
            docs[uid] = attrs
    return docs


def split_samples(
    samples: list[dict[str, Any]],
    *,
    train_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not samples:
        return [], []
    cutoff = max(1, int(len(samples) * train_ratio))
    if cutoff >= len(samples):
        cutoff = max(1, len(samples) - 1)
    return samples[:cutoff], samples[cutoff:]


def prepare_dataset(
    *,
    graph_dir: Path,
    queries_path: Path,
    output_root: Path,
    data_name: str,
    train_ratio: float,
) -> dict[str, Any]:
    data_dir = output_root / data_name
    raw_dir = data_dir / "raw"
    stage1_dir = data_dir / "processed" / "stage1"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stage1_dir.mkdir(parents=True, exist_ok=True)

    for name in ("nodes.csv", "relations.csv", "edges.csv"):
        src = graph_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing graph file: {src}")
        shutil.copy2(src, stage1_dir / name)

    node_attrs, entity_by_norm = load_node_indexes(graph_dir / "nodes.csv")
    documents = build_documents(node_attrs)
    (raw_dir / "documents.json").write_text(
        json.dumps(documents, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    samples: list[dict[str, Any]] = []
    skipped_no_start = 0
    skipped_no_target = 0
    for idx, item in enumerate(read_jsonl(queries_path)):
        start_nodes = resolve_start_entities(item, entity_by_norm)
        target_nodes = [uid for uid in evidence_chunk_ids(item) if uid in node_attrs]
        if not start_nodes:
            skipped_no_start += 1
            continue
        if not target_nodes:
            skipped_no_target += 1
            continue
        samples.append(
            {
                "id": item.get("query_id") or item.get("id") or idx,
                "question": item["question"],
                "start_nodes": {"entity": start_nodes},
                "target_nodes": {"chunk": sorted(set(target_nodes))},
                "metadata": {
                    "source_pdf": item.get("source_pdf"),
                    "type": item.get("type"),
                    "answer": item.get("answer"),
                },
            }
        )

    train_samples, test_samples = split_samples(samples, train_ratio=train_ratio)
    (stage1_dir / "train.json").write_text(
        json.dumps(train_samples, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    (stage1_dir / "test.json").write_text(
        json.dumps(test_samples, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    summary = {
        "graph_dir": str(graph_dir).replace("\\", "/"),
        "queries_path": str(queries_path).replace("\\", "/"),
        "output_root": str(output_root).replace("\\", "/"),
        "data_name": data_name,
        "documents": len(documents),
        "samples_total": len(samples),
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "skipped_no_start": skipped_no_start,
        "skipped_no_target": skipped_no_target,
        "stage1_dir": str(stage1_dir).replace("\\", "/"),
    }
    (stage1_dir / "prepare_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a GFM-RAG stage1 dataset from the local graph CSV files."
    )
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-name", default=DEFAULT_DATA_NAME)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    args = parser.parse_args()

    summary = prepare_dataset(
        graph_dir=args.graph_dir,
        queries_path=args.queries,
        output_root=args.output_root,
        data_name=args.data_name,
        train_ratio=args.train_ratio,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
