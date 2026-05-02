from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.bge_embedder import DEFAULT_BGE_MODEL, load_bge_model


DEFAULT_FINANCEBENCH = Path("benchmark/financebench/data/financebench_open_source.jsonl")
DEFAULT_CHUNKS = Path("data/chunks/all_chunks.jsonl")
DEFAULT_INDEX = Path("data/index_bge/all.faiss")
DEFAULT_IDS = Path("data/index_bge/all_chunk_ids.json")
DEFAULT_OUT_DIR = Path("data/eval/financebench_retrieval")


def require_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing faiss-cpu. Install it with: python -m pip install faiss-cpu") from exc
    return faiss


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("\u00a0", " ")
    text = text.replace("â€”", "-").replace("—", "-").replace("–", "-")
    text = re.sub(r"[^a-z0-9$%.\-()]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text))


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9$%.\-()]+", normalize_text(text)))


def token_recall(needle: str, haystack: str) -> float:
    needle_tokens = token_set(needle)
    if not needle_tokens:
        return 0.0
    return len(needle_tokens & token_set(haystack)) / len(needle_tokens)


def sequence_ratio(needle: str, haystack: str, max_haystack_chars: int = 12000) -> float:
    needle_norm = compact_text(needle)
    haystack_norm = compact_text(haystack)
    if not needle_norm or not haystack_norm:
        return 0.0
    if len(haystack_norm) > max_haystack_chars:
        # Center a cheap candidate window around any rare-ish evidence token if possible.
        tokens = sorted(token_set(needle_norm), key=len, reverse=True)
        start = 0
        for token in tokens:
            pos = haystack_norm.find(token)
            if pos >= 0:
                start = max(0, pos - max_haystack_chars // 3)
                break
        haystack_norm = haystack_norm[start : start + max_haystack_chars]
    return SequenceMatcher(None, needle_norm, haystack_norm).ratio()


def chunk_text(chunk: dict) -> str:
    return str(chunk.get("text") or chunk.get("summary") or "")


def doc_key(value: str | None) -> str:
    if not value:
        return ""
    stem = Path(str(value).replace("\\", "/")).stem
    return stem.lower()


def chunk_doc_name(chunk: dict) -> str:
    for key in ("doc_name", "source_doc", "source_pdf", "source_html"):
        value = chunk.get(key)
        if value:
            return doc_key(str(value))
    return ""


def evidence_doc_name(sample: dict, evidence: dict) -> str:
    return doc_key(evidence.get("evidence_doc_name") or evidence.get("doc_name") or sample.get("doc_name"))


def page_range(chunk: dict) -> tuple[int | None, int | None]:
    start = chunk.get("page_start", chunk.get("page"))
    end = chunk.get("page_end", chunk.get("page"))
    try:
        start_i = int(start) if start is not None else None
        end_i = int(end) if end is not None else start_i
    except (TypeError, ValueError):
        return None, None
    return start_i, end_i


def page_overlaps(chunk: dict, one_indexed_page: int | None) -> bool:
    if one_indexed_page is None:
        return False
    start, end = page_range(chunk)
    if start is None or end is None:
        return False
    return start <= one_indexed_page <= end


@dataclass
class MatchConfig:
    fuzzy_token_threshold: float = 0.6
    fuzzy_sequence_threshold: float = 0.18
    max_fuzzy_per_evidence: int = 5
    use_page_fallback: bool = True


def match_evidence_to_chunks(
    sample: dict,
    evidence: dict,
    chunks: list[dict],
    chunks_by_doc: dict[str, list[dict]],
    config: MatchConfig,
) -> list[dict]:
    ev_text = evidence.get("evidence_text") or evidence.get("evidence_text_full_page") or ""
    ev_doc = evidence_doc_name(sample, evidence)
    ev_page_raw = evidence.get("evidence_page_num")
    ev_page = int(ev_page_raw) + 1 if ev_page_raw is not None else None

    candidates = chunks_by_doc.get(ev_doc) or chunks
    exact_matches = []
    ev_compact = compact_text(ev_text)
    if ev_compact:
        for chunk in candidates:
            if ev_compact in compact_text(chunk_text(chunk)):
                exact_matches.append(
                    {
                        "chunk_id": chunk["id"],
                        "match_method": "exact_text",
                        "score": 1.0,
                        "evidence_page": ev_page,
                        "evidence_doc_name": ev_doc,
                    }
                )
    if exact_matches:
        return exact_matches

    fuzzy = []
    if ev_compact:
        for chunk in candidates:
            text = chunk_text(chunk)
            t_recall = token_recall(ev_text, text)
            s_ratio = sequence_ratio(ev_text, text)
            score = (0.75 * t_recall) + (0.25 * s_ratio)
            if t_recall >= config.fuzzy_token_threshold or s_ratio >= config.fuzzy_sequence_threshold:
                fuzzy.append(
                    {
                        "chunk_id": chunk["id"],
                        "match_method": "fuzzy_text",
                        "score": round(score, 6),
                        "token_recall": round(t_recall, 6),
                        "sequence_ratio": round(s_ratio, 6),
                        "evidence_page": ev_page,
                        "evidence_doc_name": ev_doc,
                    }
                )
    if fuzzy:
        return sorted(fuzzy, key=lambda row: row["score"], reverse=True)[: config.max_fuzzy_per_evidence]

    if config.use_page_fallback:
        page_matches = [
            {
                "chunk_id": chunk["id"],
                "match_method": "page_fallback",
                "score": 1.0,
                "evidence_page": ev_page,
                "evidence_doc_name": ev_doc,
            }
            for chunk in candidates
            if page_overlaps(chunk, ev_page)
        ]
        if page_matches:
            return page_matches

    return []


def build_qrels(samples: list[dict], chunks: list[dict], config: MatchConfig) -> tuple[list[dict], list[dict]]:
    chunks_by_doc: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_doc[chunk_doc_name(chunk)].append(chunk)

    qrels = []
    misses = []
    for sample in samples:
        qid = str(sample["financebench_id"])
        evidence_rows = sample.get("evidence") or []
        sample_matches = []
        for ev_idx, evidence in enumerate(evidence_rows):
            matches = match_evidence_to_chunks(sample, evidence, chunks, chunks_by_doc, config)
            for match in matches:
                row = {
                    "qid": qid,
                    "chunk_id": match["chunk_id"],
                    "relevance": 1,
                    "match_method": match["match_method"],
                    "match_score": match["score"],
                    "evidence_index": ev_idx,
                    "evidence_doc_name": match["evidence_doc_name"],
                    "evidence_page": match["evidence_page"],
                }
                if "token_recall" in match:
                    row["token_recall"] = match["token_recall"]
                    row["sequence_ratio"] = match["sequence_ratio"]
                qrels.append(row)
                sample_matches.append(row)
        if not sample_matches:
            misses.append(
                {
                    "qid": qid,
                    "doc_name": sample.get("doc_name"),
                    "question": sample.get("question"),
                    "evidence_count": len(evidence_rows),
                    "evidence_pages": [ev.get("evidence_page_num") for ev in evidence_rows],
                }
            )
    return qrels, misses


def retrieve(
    samples: list[dict],
    chunks: list[dict],
    index_path: Path,
    ids_path: Path,
    top_k: int,
    model_name: str,
    batch_size: int,
    device: str | None,
) -> dict[str, list[dict]]:
    faiss = require_faiss()
    index = faiss.read_index(str(index_path))
    index_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
    embedder = load_bge_model(model_name=model_name, batch_size=batch_size, device=device)
    queries = [sample["question"] for sample in samples]
    query_vectors = embedder.encode_queries(queries)
    search_k = min(max(top_k, 1), len(index_ids))
    scores, indices = index.search(query_vectors, search_k)

    runs: dict[str, list[dict]] = {}
    for sample, sample_scores, sample_indices in zip(samples, scores, indices):
        qid = str(sample["financebench_id"])
        rows = []
        for rank, (score, idx) in enumerate(zip(sample_scores, sample_indices), start=1):
            if idx < 0:
                continue
            chunk_id = index_ids[int(idx)]
            chunk = chunks_by_id.get(chunk_id, {})
            rows.append(
                {
                    "qid": qid,
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "score": float(score),
                    "doc_name": chunk_doc_name(chunk),
                    "page_start": page_range(chunk)[0],
                    "page_end": page_range(chunk)[1],
                    "modality": chunk.get("modality") or chunk.get("chunk_type"),
                    "preview": chunk_text(chunk)[:500],
                }
            )
        runs[qid] = rows
    return runs


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate(
    samples: list[dict],
    qrels: list[dict],
    runs: dict[str, list[dict]],
    chunks_by_id: dict[str, dict],
    k_values: list[int],
) -> tuple[list[dict], list[dict]]:
    qrels_by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in qrels:
        qrels_by_qid[row["qid"]].append(row)

    per_question = []
    for sample in samples:
        qid = str(sample["financebench_id"])
        relevant_rows = qrels_by_qid.get(qid, [])
        relevant_ids = {row["chunk_id"] for row in relevant_rows}
        evidence_pages = {row.get("evidence_page") for row in relevant_rows if row.get("evidence_page") is not None}
        evidence_doc_names = {row.get("evidence_doc_name") for row in relevant_rows if row.get("evidence_doc_name")}
        text_relevant_ids = {
            row["chunk_id"] for row in relevant_rows if row.get("match_method") in {"exact_text", "fuzzy_text"}
        }
        retrieved = runs.get(qid, [])
        retrieved_ids = [row["chunk_id"] for row in retrieved]

        base = {
            "qid": qid,
            "company": sample.get("company"),
            "doc_name": sample.get("doc_name"),
            "question_type": sample.get("question_type"),
            "question_reasoning": sample.get("question_reasoning"),
            "question": sample.get("question"),
            "answer": sample.get("answer"),
            "num_relevant_chunks": len(relevant_ids),
            "qrel_methods": ",".join(sorted({row["match_method"] for row in relevant_rows})),
        }

        for k in k_values:
            top_ids = retrieved_ids[:k]
            hits = [chunk_id for chunk_id in top_ids if chunk_id in relevant_ids]
            page_hit = False
            evidence_hit = False
            for chunk_id in top_ids:
                chunk = chunks_by_id.get(chunk_id, {})
                same_doc = not evidence_doc_names or chunk_doc_name(chunk) in evidence_doc_names
                if same_doc and any(page_overlaps(chunk, page) for page in evidence_pages):
                    page_hit = True
                if chunk_id in text_relevant_ids:
                    evidence_hit = True

            denom = len(relevant_ids) if relevant_ids else 0
            base[f"precision@{k}"] = len(hits) / k
            base[f"recall@{k}"] = len(set(hits)) / denom if denom else math.nan
            base[f"hit@{k}"] = 1.0 if hits else 0.0
            base[f"mrr@{k}"] = reciprocal_rank(retrieved_ids, relevant_ids, k)
            base[f"page_recall@{k}"] = 1.0 if page_hit else 0.0
            base[f"evidence_recall@{k}"] = 1.0 if evidence_hit else 0.0
        per_question.append(base)

    aggregate = []
    for k in k_values:
        row = {"k": k, "questions": len(samples)}
        for metric in ("precision", "recall", "hit", "mrr", "page_recall", "evidence_recall"):
            values = [q[f"{metric}@{k}"] for q in per_question]
            values = [value for value in values if not (isinstance(value, float) and math.isnan(value))]
            row[f"{metric}@{k}"] = float(np.mean(values)) if values else math.nan
        aggregate.append(row)
    return aggregate, per_question


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_runs(runs: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for qid, retrieved in runs.items():
        rows.extend(retrieved)
    return rows


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.4f}" if not math.isnan(value) else "nan"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    aggregate: list[dict],
    per_question: list[dict],
    qrels: list[dict],
    qrel_misses: list[dict],
    args: argparse.Namespace,
) -> None:
    qrel_methods = Counter(row["match_method"] for row in qrels)
    aggregate_display = []
    for row in aggregate:
        k = row["k"]
        aggregate_display.append(
            {
                "k": k,
                "precision": row.get(f"precision@{k}"),
                "recall": row.get(f"recall@{k}"),
                "hit": row.get(f"hit@{k}"),
                "mrr": row.get(f"mrr@{k}"),
                "page_recall": row.get(f"page_recall@{k}"),
                "evidence_recall": row.get(f"evidence_recall@{k}"),
            }
        )
    no_hit_10 = sorted(
        per_question,
        key=lambda row: (row.get("hit@10", 0), row.get("recall@10", 0)),
    )[:20]
    lines = [
        "# FinanceBench Retrieval Evaluation",
        "",
        "## Config",
        "",
        f"- benchmark: `{args.benchmark}`",
        f"- chunks: `{args.chunks}`",
        f"- index: `{args.index}`",
        f"- ids: `{args.ids}`",
        f"- model: `{args.model}`",
        f"- k: `{','.join(map(str, args.k_values))}`",
        "",
        "## Aggregate Metrics",
        "",
        markdown_table(aggregate_display, ["k", "precision", "recall", "hit", "mrr", "page_recall", "evidence_recall"]),
        "",
        "## Qrel Match Methods",
        "",
        markdown_table([{"method": key, "count": value} for key, value in qrel_methods.items()], ["method", "count"]),
        "",
        f"Questions without any qrel match: {len(qrel_misses)}",
        "",
        "## Error Analysis Sample",
        "",
        markdown_table(
            [
                {
                    "qid": row["qid"],
                    "doc_name": row["doc_name"],
                    "hit@10": row.get("hit@10", ""),
                    "recall@10": row.get("recall@10", ""),
                    "qrel_methods": row.get("qrel_methods", ""),
                    "question": str(row.get("question", ""))[:120],
                }
                for row in no_hit_10
            ],
            ["qid", "doc_name", "hit@10", "recall@10", "qrel_methods", "question"],
        ),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_k_values(value: str) -> list[int]:
    values = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not values or any(k <= 0 for k in values):
        raise argparse.ArgumentTypeError("--k-values must contain positive integers")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval on FinanceBench by mapping evidence_text/pages to local chunks."
    )
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_FINANCEBENCH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default=DEFAULT_BGE_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--k-values", type=parse_k_values, default=[1, 3, 5, 10])
    parser.add_argument("--fuzzy-token-threshold", type=float, default=0.6)
    parser.add_argument("--fuzzy-sequence-threshold", type=float, default=0.18)
    parser.add_argument("--max-fuzzy-per-evidence", type=int, default=5)
    parser.add_argument("--no-page-fallback", action="store_true")
    args = parser.parse_args()

    samples = load_jsonl(args.benchmark)
    chunks = load_jsonl(args.chunks)
    max_k = max(args.k_values)

    match_config = MatchConfig(
        fuzzy_token_threshold=args.fuzzy_token_threshold,
        fuzzy_sequence_threshold=args.fuzzy_sequence_threshold,
        max_fuzzy_per_evidence=args.max_fuzzy_per_evidence,
        use_page_fallback=not args.no_page_fallback,
    )
    qrels, qrel_misses = build_qrels(samples, chunks, match_config)
    runs = retrieve(
        samples=samples,
        chunks=chunks,
        index_path=args.index,
        ids_path=args.ids,
        top_k=max_k,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
    )
    chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
    aggregate, per_question = evaluate(samples, qrels, runs, chunks_by_id, args.k_values)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "qrels.jsonl", qrels)
    write_jsonl(args.out_dir / "qrel_misses.jsonl", qrel_misses)
    write_jsonl(args.out_dir / "runs.jsonl", flatten_runs(runs))
    write_jsonl(args.out_dir / "per_question_metrics.jsonl", per_question)
    write_csv(args.out_dir / "aggregate_metrics.csv", aggregate)
    write_report(args.out_dir / "report.md", aggregate, per_question, qrels, qrel_misses, args)

    print(json.dumps({"out_dir": str(args.out_dir), "aggregate": aggregate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
