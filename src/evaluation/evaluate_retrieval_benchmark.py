from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.bge_embedder import DEFAULT_BGE_MODEL, load_bge_model  # noqa: E402


SCHEMA_FIELDS = [
    "named_entities", "dates", "industries", "domains", "sectors",
    "organizations", "partnerships", "partners", "dividends", "products", "locations",
]
TAG_FIELDS_FOR_AWARE_MODES = ("organizations", "dates", "products", "locations", "named_entities")


def require_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing faiss-cpu. Install it with: python -m pip install faiss-cpu") from exc
    return faiss


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def rerank_score(query: dict[str, Any], chunk: dict[str, Any], dense_score: float) -> float:
    score = float(dense_score)
    qtype = query.get("type")
    modality = chunk.get("modality")
    if qtype == modality:
        score += 0.08
    if qtype == "multimodal" and modality in {"text", "table", "image"}:
        score += 0.03
    source_pdf = query.get("source_pdf")
    if source_pdf and chunk.get("source_pdf") == source_pdf:
        score += 0.06
    return score


def _normalize_tag_value(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"\b(incorporated|inc\.?|corporation|corp\.?|company|co\.?|llc|ltd\.?)\b", "", value)
    value = value.replace("&", "and")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def _normalize_tags(tags: Any) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {f: [] for f in SCHEMA_FIELDS}
    if not isinstance(tags, dict):
        return normalized
    for field in SCHEMA_FIELDS:
        values = tags.get(field) or []
        if not isinstance(values, list):
            values = [values]
        seen: set[str] = set()
        for v in values:
            if not isinstance(v, str):
                continue
            item = re.sub(r"\s+", " ", v).strip()
            if not item:
                continue
            key = _normalize_tag_value(item)
            if key in seen:
                continue
            seen.add(key)
            normalized[field].append(item)
    return normalized


def _tag_overlap_count(qtags: dict[str, list[str]], ctags: dict[str, list[str]]) -> int:
    count = 0
    for field in TAG_FIELDS_FOR_AWARE_MODES:
        q_keys = {_normalize_tag_value(v) for v in qtags.get(field, [])}
        c_keys = {_normalize_tag_value(v) for v in ctags.get(field, [])}
        count += len(q_keys & c_keys)
    return count


def _get_chunk_tags(record: dict[str, Any]) -> dict[str, list[str]]:
    if isinstance(record.get("semantic_tags"), dict):
        return _normalize_tags(record["semantic_tags"])
    meta = record.get("metadata")
    if isinstance(meta, dict):
        return _normalize_tags(meta.get("semantic_tags"))
    return _normalize_tags(None)


def _get_query_tags(query: dict[str, Any]) -> dict[str, list[str]]:
    for key in ("query_semantic_tags", "semantic_tags"):
        if key in query:
            return _normalize_tags(query[key])
    return _normalize_tags(None)


def evaluate(
    benchmark_dir: Path,
    chunks_path: Path,
    index_path: Path,
    ids_path: Path,
    output_dir: Path,
    top_k: int,
    k_values: list[int],
    model_name: str,
    batch_size: int,
    device: str | None,
    filter_source: bool,
    rerank: bool,
    tagged_index_dir: Path | None = None,
    queries_tagged_path: Path | None = None,
    tag_boost_weight: float = 0.03,
) -> dict[str, Any]:
    faiss = require_faiss()

    # Dùng queries tagged nếu có, fallback sang queries thường
    queries_path = queries_tagged_path if (queries_tagged_path and queries_tagged_path.exists()) else benchmark_dir / "queries.jsonl"
    queries = load_jsonl(queries_path)
    qrels = load_jsonl(benchmark_dir / "qrels.jsonl")

    chunks = load_jsonl(chunks_path)
    chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
    index_ids = json.loads(ids_path.read_text(encoding="utf-8"))

    # Load tagged index nếu có
    tagged_ids: list[str] = []
    tagged_records_by_id: dict[str, dict[str, Any]] = {}
    tagged_index = None
    if tagged_index_dir and tagged_index_dir.exists():
        tagged_index = faiss.read_index(str(tagged_index_dir / "chunks.faiss"))
        tagged_ids = json.loads((tagged_index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
        for rec in load_jsonl(tagged_index_dir / "records.jsonl"):
            cid = str(rec.get("chunk_id") or rec.get("id") or "")
            if cid:
                tagged_records_by_id[cid] = rec

    relevant_by_query: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        relevant_by_query[row["query_id"]].add(row["chunk_id"])

    index = faiss.read_index(str(index_path))
    embedder = load_bge_model(model_name=model_name, batch_size=batch_size, device=device)
    query_vectors = embedder.encode_queries([row["question"] for row in queries])

    search_k = min(max(top_k * 30, top_k), len(index_ids))
    scores, indices = index.search(query_vectors, search_k)

    # Tag-aware search trên tagged index
    # Áp dụng CÙNG điều kiện với Proposed: filter_source + rerank + tag scoring
    tagged_scores_map: dict[str, dict[str, tuple[float, float]]] = {}  # qid → {chunk_id: (boost_score, filter_score)}
    if tagged_index is not None:
        tag_search_k = min(max(top_k * 30, top_k), len(tagged_ids))
        t_scores, t_indices = tagged_index.search(query_vectors, tag_search_k)
        for query, tsc, tidx in zip(queries, t_scores, t_indices):
            qid = query["query_id"]
            source_pdf = query.get("source_pdf")
            qtags = _get_query_tags(query)
            query_has_useful_tags = any(qtags.get(f) for f in TAG_FIELDS_FOR_AWARE_MODES)
            boost_map: dict[str, tuple[float, float]] = {}
            for sc, idx in zip(tsc, tidx):
                if idx < 0:
                    continue
                cid = tagged_ids[int(idx)]
                rec = tagged_records_by_id.get(cid, {})
                rec_meta = rec.get("metadata") or {}
                # Áp dụng filter_source giống Proposed
                if filter_source and source_pdf and rec_meta.get("source_pdf") != source_pdf:
                    continue
                # Áp dụng rerank bonus giống Proposed
                base_score = rerank_score(query, rec_meta, float(sc)) if rerank else float(sc)
                ctags = _get_chunk_tags(rec)
                overlap = _tag_overlap_count(qtags, ctags)
                has_tag_match = query_has_useful_tags and overlap > 0
                # tag_boost: rerank_score + tag overlap bonus
                boost_score = base_score + tag_boost_weight * overlap
                # tag_filter: chỉ giữ chunk có overlap (hoặc query không có tags → fallback)
                filter_score = base_score if (has_tag_match or not query_has_useful_tags) else -1.0
                boost_map[cid] = (boost_score, filter_score)
            tagged_scores_map[qid] = boost_map

    by_question = []
    retrieval_rows = []
    # tag-aware rows lưu riêng để tổng hợp sau
    tag_boost_rows: dict[str, list[str]] = {}
    tag_filter_rows: dict[str, list[str]] = {}

    for query, query_scores, query_indices in zip(queries, scores, indices):
        qid = query["query_id"]
        source_pdf = query.get("source_pdf")
        relevant_ids = relevant_by_query.get(qid, set())
        retrieved_ids: list[str] = []

        candidates = []
        for score, idx in zip(query_scores, query_indices):
            if idx < 0:
                continue
            chunk_id = index_ids[int(idx)]
            chunk = chunks_by_id.get(chunk_id, {})
            if filter_source and source_pdf and chunk.get("source_pdf") != source_pdf:
                continue
            final_score = rerank_score(query, chunk, float(score)) if rerank else float(score)
            candidates.append((final_score, float(score), chunk_id, chunk))

        candidates.sort(key=lambda row: row[0], reverse=True)
        for final_score, dense_score, chunk_id, chunk in candidates:
            retrieved_ids.append(chunk_id)
            retrieval_rows.append({
                "query_id": qid,
                "rank": len(retrieved_ids),
                "chunk_id": chunk_id,
                "score": final_score,
                "dense_score": dense_score,
                "is_relevant": chunk_id in relevant_ids,
                "query_type": query.get("type"),
                "chunk_modality": chunk.get("modality"),
                "source_pdf": chunk.get("source_pdf"),
                "page": chunk.get("page"),
            })
            if len(retrieved_ids) >= top_k:
                break

        row = {
            "query_id": qid,
            "type": query.get("type", "unknown"),
            "question": query.get("question"),
            "num_relevant": len(relevant_ids),
            "retrieved_ids": retrieved_ids[:top_k],
        }
        for k in k_values:
            top_ids = retrieved_ids[:k]
            hits = len(set(top_ids) & relevant_ids)
            row[f"hit@{k}"] = 1.0 if hits else 0.0
            row[f"recall@{k}"] = hits / len(relevant_ids) if relevant_ids else 0.0
            row[f"precision@{k}"] = hits / k
            row[f"mrr@{k}"] = reciprocal_rank(top_ids, relevant_ids, k)

        # Tag-aware metrics
        if qid in tagged_scores_map:
            boost_map = tagged_scores_map[qid]
            # tag_boost: re-sort theo boost_score
            boost_sorted = sorted(boost_map.items(), key=lambda x: x[1][0], reverse=True)
            tag_boost_rows[qid] = [cid for cid, _ in boost_sorted[:top_k]]
            # tag_filter: chỉ giữ chunk có overlap
            filter_sorted = sorted(
                [(cid, sc) for cid, (_, sc) in boost_map.items() if sc >= 0],
                key=lambda x: x[1], reverse=True,
            )
            tag_filter_rows[qid] = [cid for cid, _ in filter_sorted[:top_k]]

            for k in k_values:
                # tag_boost
                tb_ids = tag_boost_rows[qid][:k]
                tb_hits = len(set(tb_ids) & relevant_ids)
                row[f"tag_boost_hit@{k}"] = 1.0 if tb_hits else 0.0
                row[f"tag_boost_recall@{k}"] = tb_hits / len(relevant_ids) if relevant_ids else 0.0
                row[f"tag_boost_mrr@{k}"] = reciprocal_rank(tb_ids, relevant_ids, k)
                # tag_filter
                tf_ids = tag_filter_rows[qid][:k]
                tf_hits = len(set(tf_ids) & relevant_ids)
                row[f"tag_filter_hit@{k}"] = 1.0 if tf_hits else 0.0
                row[f"tag_filter_recall@{k}"] = tf_hits / len(relevant_ids) if relevant_ids else 0.0
                row[f"tag_filter_mrr@{k}"] = reciprocal_rank(tf_ids, relevant_ids, k)

        by_question.append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_question:
        grouped[row["type"]].append(row)
        grouped["all"].append(row)

    has_tags = bool(tagged_scores_map)
    summary = {
        "benchmark_dir": str(benchmark_dir).replace("\\", "/"),
        "queries": len(queries),
        "qrels": len(qrels),
        "top_k": top_k,
        "filter_source": filter_source,
        "rerank": rerank,
        "tag_aware": has_tags,
        "metrics": {},
    }
    base_metrics = ["hit", "recall", "precision", "mrr"]
    tag_metrics = ["tag_boost_hit", "tag_boost_recall", "tag_boost_mrr",
                   "tag_filter_hit", "tag_filter_recall", "tag_filter_mrr"]

    for qtype, rows in grouped.items():
        summary["metrics"][qtype] = {"queries": len(rows)}
        for k in k_values:
            for metric in base_metrics:
                key = f"{metric}@{k}"
                summary["metrics"][qtype][key] = float(np.mean([row[key] for row in rows])) if rows else 0.0
            if has_tags:
                for metric in tag_metrics:
                    key = f"{metric}@{k}"
                    vals = [row[key] for row in rows if key in row]
                    summary["metrics"][qtype][key] = float(np.mean(vals)) if vals else 0.0

    write_json(output_dir / "metrics_summary.json", summary)
    write_jsonl(output_dir / "metrics_by_question.jsonl", by_question)
    write_jsonl(output_dir / "retrieval_results.jsonl", retrieval_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate chunk-level retrieval benchmark.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark_hard"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks/all_chunks.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/index_bge/all.faiss"))
    parser.add_argument("--ids", type=Path, default=Path("data/index_bge/all_chunk_ids.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/retrieval_benchmark_hard"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--model", default=DEFAULT_BGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--filter-source", action="store_true")
    parser.add_argument("--rerank", action="store_true", help="Apply a lightweight source/modality-aware reranker.")
    parser.add_argument("--tagged-index-dir", type=Path, default=None)
    parser.add_argument("--queries-tagged", type=Path, default=None)
    parser.add_argument("--tag-boost-weight", type=float, default=0.03)
    args = parser.parse_args()

    summary = evaluate(
        benchmark_dir=args.benchmark_dir,
        chunks_path=args.chunks,
        index_path=args.index,
        ids_path=args.ids,
        output_dir=args.output_dir,
        top_k=args.top_k,
        k_values=args.k_values,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        filter_source=args.filter_source,
        rerank=args.rerank,
        tagged_index_dir=args.tagged_index_dir,
        queries_tagged_path=args.queries_tagged,
        tag_boost_weight=args.tag_boost_weight,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
