from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.bge_embedder import DEFAULT_BGE_MODEL, load_bge_model  # noqa: E402


def require_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing faiss-cpu. Install it with: python -m pip install faiss-cpu") from exc
    return faiss


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9._%$-]*", str(text or "").lower())


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


class BM25Index:
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        self.docs = [tokenize(chunk.get("text")) for chunk in chunks]
        self.doc_lens = np.array([len(doc) for doc in self.docs], dtype="float32")
        self.avgdl = float(np.mean(self.doc_lens)) if len(self.doc_lens) else 0.0
        self.term_freqs = [Counter(doc) for doc in self.docs]
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        n_docs = len(self.docs)
        self.idf = {
            term: math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int, source_pdf: str | None = None, filter_source: bool = False) -> list[str]:
        query_terms = tokenize(query)
        scores = np.zeros(len(self.docs), dtype="float32")
        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf in enumerate(self.term_freqs):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * self.doc_lens[idx] / max(self.avgdl, 1e-9))
                scores[idx] += idf * (freq * (self.k1 + 1.0) / denom)

        out: list[str] = []
        for idx in np.argsort(-scores):
            if scores[idx] <= 0:
                break
            chunk = self.chunks[int(idx)]
            if filter_source and source_pdf and chunk.get("source_pdf") != source_pdf:
                continue
            out.append(chunk["chunk_id"])
            if len(out) >= top_k:
                break
        return out


def rrf_score(rank: int, rrf_k: int) -> float:
    return 1.0 / (rrf_k + rank)


def rerank_bonus(query: dict[str, Any], chunk: dict[str, Any]) -> float:
    score = 0.0
    if query.get("type") == chunk.get("modality"):
        score += 0.08
    if query.get("type") == "multimodal" and chunk.get("modality") in {"text", "table", "image"}:
        score += 0.03
    if query.get("source_pdf") and query.get("source_pdf") == chunk.get("source_pdf"):
        score += 0.06
    return score


def evaluate(
    *,
    benchmark_dir: Path,
    index_path: Path,
    ids_path: Path,
    output_dir: Path,
    top_k: int,
    candidate_k: int,
    k_values: list[int],
    rrf_k: int,
    model_name: str,
    batch_size: int,
    device: str | None,
    filter_source: bool,
    rerank: bool,
) -> dict[str, Any]:
    faiss = require_faiss()
    queries = load_jsonl(benchmark_dir / "queries.jsonl")
    qrels = load_jsonl(benchmark_dir / "qrels.jsonl")
    corpus = load_jsonl(benchmark_dir / "corpus.jsonl")
    corpus_by_id = {row["chunk_id"]: row for row in corpus}

    relevant_by_query: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        relevant_by_query[row["query_id"]].add(row["chunk_id"])

    bm25 = BM25Index(corpus)
    dense_index = faiss.read_index(str(index_path))
    dense_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    embedder = load_bge_model(model_name=model_name, batch_size=batch_size, device=device)
    query_vectors = embedder.encode_queries([row["question"] for row in queries])
    dense_scores, dense_indices = dense_index.search(query_vectors, min(candidate_k, len(dense_ids)))

    by_question: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []

    for query, dscores, didx in zip(queries, dense_scores, dense_indices):
        qid = query["query_id"]
        source_pdf = query.get("source_pdf")
        relevant_ids = relevant_by_query.get(qid, set())

        bm25_ids = bm25.search(query["question"], candidate_k, source_pdf=source_pdf, filter_source=filter_source)
        bm25_rank = {cid: rank for rank, cid in enumerate(bm25_ids, start=1)}

        dense_rank: dict[str, int] = {}
        for rank, raw_idx in enumerate(didx, start=1):
            if raw_idx < 0:
                continue
            cid = dense_ids[int(raw_idx)]
            chunk = corpus_by_id.get(cid)
            if not chunk:
                continue
            if filter_source and source_pdf and chunk.get("source_pdf") != source_pdf:
                continue
            dense_rank[cid] = rank

        fused: dict[str, float] = {}
        for cid, rank in bm25_rank.items():
            fused[cid] = fused.get(cid, 0.0) + rrf_score(rank, rrf_k)
        for cid, rank in dense_rank.items():
            fused[cid] = fused.get(cid, 0.0) + rrf_score(rank, rrf_k)

        candidates = []
        for cid, score in fused.items():
            chunk = corpus_by_id.get(cid, {})
            final_score = score + (rerank_bonus(query, chunk) if rerank else 0.0)
            candidates.append((final_score, score, cid, chunk))
        candidates.sort(key=lambda row: row[0], reverse=True)
        retrieved_ids = [cid for _, _, cid, _ in candidates[:top_k]]

        for rank, (final_score, fused_score, cid, chunk) in enumerate(candidates[:top_k], start=1):
            retrieval_rows.append(
                {
                    "query_id": qid,
                    "rank": rank,
                    "chunk_id": cid,
                    "score": final_score,
                    "rrf_score": fused_score,
                    "is_relevant": cid in relevant_ids,
                    "query_type": query.get("type"),
                    "chunk_modality": chunk.get("modality"),
                    "source_pdf": chunk.get("source_pdf"),
                    "page": chunk.get("page"),
                }
            )

        row = {
            "query_id": qid,
            "type": query.get("type", "unknown"),
            "question": query.get("question"),
            "num_relevant": len(relevant_ids),
            "retrieved_ids": retrieved_ids,
        }
        for k in k_values:
            top_ids = retrieved_ids[:k]
            hits = len(set(top_ids) & relevant_ids)
            row[f"hit@{k}"] = 1.0 if hits else 0.0
            row[f"recall@{k}"] = hits / len(relevant_ids) if relevant_ids else 0.0
            row[f"precision@{k}"] = hits / k
            row[f"mrr@{k}"] = reciprocal_rank(top_ids, relevant_ids, k)
        by_question.append(row)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in by_question:
        grouped[row["type"]].append(row)
        grouped["all"].append(row)

    summary = {
        "benchmark_dir": str(benchmark_dir).replace("\\", "/"),
        "queries": len(queries),
        "qrels": len(qrels),
        "top_k": top_k,
        "candidate_k": candidate_k,
        "rrf_k": rrf_k,
        "method": "hybrid_rrf_bm25_dense",
        "filter_source": filter_source,
        "rerank": rerank,
        "metrics": {},
    }
    for qtype, rows in grouped.items():
        summary["metrics"][qtype] = {"queries": len(rows)}
        for k in k_values:
            for metric in ("hit", "recall", "precision", "mrr"):
                key = f"{metric}@{k}"
                summary["metrics"][qtype][key] = float(np.mean([row[key] for row in rows])) if rows else 0.0

    write_json(output_dir / "metrics_summary.json", summary)
    write_jsonl(output_dir / "metrics_by_question.jsonl", by_question)
    write_jsonl(output_dir / "retrieval_results.jsonl", retrieval_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hybrid RRF(BM25 + Dense BGE) retrieval.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark_report"))
    parser.add_argument("--index", type=Path, default=Path("data/index_bge/all.faiss"))
    parser.add_argument("--ids", type=Path, default=Path("data/index_bge/all_chunk_ids.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/retrieval_benchmark_report_hybrid_rrf"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--model", default=DEFAULT_BGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--filter-source", action="store_true")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    summary = evaluate(
        benchmark_dir=args.benchmark_dir,
        index_path=args.index,
        ids_path=args.ids,
        output_dir=args.output_dir,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        k_values=args.k_values,
        rrf_k=args.rrf_k,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        filter_source=args.filter_source,
        rerank=args.rerank,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
