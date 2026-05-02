from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.bge_embedder import DEFAULT_BGE_MODEL, load_bge_model


def _load_cross_encoder(model_name: str | None):
    if not model_name:
        return None
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder(model_name)
    except Exception as exc:
        print(f"[warn] Không load được CrossEncoder '{model_name}', fallback về heuristic rerank. Lý do: {exc}")
        return None


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------

def _tokenize_bm25(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9._%$-]*", str(text or "").lower())


class BM25Index:
    def __init__(self, docs: list[tuple[str, str]], k1: float = 1.5, b: float = 0.75) -> None:
        # docs: list of (chunk_id, text)
        self.ids = [d[0] for d in docs]
        tokenized = [_tokenize_bm25(d[1]) for d in docs]
        self.doc_lens = np.array([len(t) for t in tokenized], dtype="float32")
        self.avgdl = float(np.mean(self.doc_lens)) if len(self.doc_lens) else 1.0
        self.k1 = k1
        self.b = b
        # xây df
        df: dict[str, int] = defaultdict(int)
        for tokens in tokenized:
            for tok in set(tokens):
                df[tok] += 1
        N = len(docs)
        self.idf: dict[str, float] = {
            tok: math.log(1 + (N - freq + 0.5) / (freq + 0.5))
            for tok, freq in df.items()
        }
        # xây tf
        self.tf: list[dict[str, int]] = [Counter(tokens) for tokens in tokenized]

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        q_tokens = _tokenize_bm25(query)
        scores = np.zeros(len(self.ids), dtype="float32")
        for tok in q_tokens:
            if tok not in self.idf:
                continue
            idf = self.idf[tok]
            for i, tf_doc in enumerate(self.tf):
                freq = tf_doc.get(tok, 0)
                if freq == 0:
                    continue
                dl = self.doc_lens[i]
                num = freq * (self.k1 + 1)
                den = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[i] += idf * num / den
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]


def build_bm25_index(records_by_id: dict[str, dict[str, Any]]) -> BM25Index:
    docs = [(cid, record_text(record)) for cid, record in records_by_id.items()]
    return BM25Index(docs)


def make_bm25_rows(
    *,
    method: str,
    queries: list[dict[str, Any]],
    bm25: BM25Index,
    top_k: int,
    candidate_k: int,
    records_by_id: dict[str, dict[str, Any]],
    tag_mode: str = "none",
    tag_boost_weight: float = 0.03,
    reranker: Any = None,
    rerank_weight: float = 1.0,
    rerank_batch_size: int = 16,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fetch_k = candidate_k if tag_mode != "none" else top_k
    for q_idx, query in enumerate(queries, start=1):
        qid = query_id(query, q_idx)
        qtags = query_tags(query)
        query_has_useful_tags = any(qtags[f] for f in TAG_FIELDS_FOR_AWARE_MODES)
        results = bm25.search(query_text(query), fetch_k)
        candidates = []
        for cid, bm25_score in results:
            record = records_by_id.get(cid, {})
            ctags = record_tags(record)
            overlap = tag_overlap(qtags, ctags)
            overlap_count = sum(len(v) for v in overlap.values())
            if tag_mode == "tag_filter" and query_has_useful_tags and not overlap_count:
                continue
            final_score = bm25_score
            if tag_mode == "tag_boost":
                final_score += tag_boost_weight * overlap_count
            candidates.append((final_score, bm25_score, cid, record, overlap))
        if reranker and candidates:
            pairs = [(query_text(query), record_text(item[3])) for item in candidates]
            rerank_scores = reranker.predict(pairs, batch_size=rerank_batch_size, show_progress_bar=False)
            reranked = []
            for (final_score, bm25_score, cid, record, overlap), rr_score in zip(candidates, rerank_scores):
                rr = float(rr_score)
                reranked.append((final_score + rerank_weight * rr, bm25_score, cid, record, overlap, rr))
            reranked.sort(key=lambda x: x[0], reverse=True)
            candidates_for_output = reranked[:top_k]
        else:
            candidates.sort(key=lambda x: x[0], reverse=True)
            candidates_for_output = [(f, b, c, r, o, None) for (f, b, c, r, o) in candidates[:top_k]]

        for rank, (final_score, bm25_score, cid, record, overlap, rerank_score) in enumerate(candidates_for_output, start=1):
            metadata = record_metadata(record)
            rows.append({
                "method": method,
                "query_id": qid,
                "rank": rank,
                "chunk_id": cid,
                "score": final_score,
                "vector_score": bm25_score,
                "debug_tag_overlap": overlap,
                "debug_rerank_score": rerank_score,
                "source_pdf": metadata.get("source_pdf"),
                "page": metadata.get("page"),
                "modality": metadata.get("modality"),
                "text_preview": clean_preview(record_text(record)),
            })
    return rows


def _rrf_fuse_scores(rank_a: dict[str, int], rank_b: dict[str, int], rrf_k: int) -> dict[str, float]:
    fused: dict[str, float] = {}
    all_ids = set(rank_a) | set(rank_b)
    for cid in all_ids:
        score = 0.0
        if cid in rank_a:
            score += 1.0 / (rrf_k + rank_a[cid])
        if cid in rank_b:
            score += 1.0 / (rrf_k + rank_b[cid])
        fused[cid] = score
    return fused


def make_hybrid_rrf_rows(
    *,
    method: str,
    queries: list[dict[str, Any]],
    bm25: BM25Index,
    dense_index: Any,
    dense_ids: list[str],
    records_by_id: dict[str, dict[str, Any]],
    query_vectors: np.ndarray,
    top_k: int,
    candidate_k: int,
    rrf_k: int,
    tag_mode: str = "none",
    tag_boost_weight: float = 0.03,
    reranker: Any = None,
    rerank_top_n: int = 50,
    rerank_weight: float = 1.0,
    rerank_batch_size: int = 16,
) -> list[dict[str, Any]]:
    search_k = min(max(candidate_k, top_k), len(dense_ids))
    dense_scores_all, dense_indices_all = dense_index.search(query_vectors, search_k)
    rows: list[dict[str, Any]] = []

    for q_idx, (query, dense_scores, dense_indices) in enumerate(zip(queries, dense_scores_all, dense_indices_all), start=1):
        qid = query_id(query, q_idx)
        qtags = query_tags(query)
        query_has_useful_tags = any(qtags[f] for f in TAG_FIELDS_FOR_AWARE_MODES)

        bm25_results = bm25.search(query_text(query), search_k)
        bm25_rank = {cid: rank for rank, (cid, _) in enumerate(bm25_results, start=1)}
        bm25_score_map = {cid: float(score) for cid, score in bm25_results}

        dense_rank: dict[str, int] = {}
        dense_score_map: dict[str, float] = {}
        for rank, (score, raw_idx) in enumerate(zip(dense_scores, dense_indices), start=1):
            if raw_idx < 0:
                continue
            cid = dense_ids[int(raw_idx)]
            dense_rank[cid] = rank
            dense_score_map[cid] = float(score)

        fused = _rrf_fuse_scores(bm25_rank, dense_rank, rrf_k=rrf_k)

        candidates = []
        for cid, fused_score in fused.items():
            record = records_by_id.get(cid)
            if not record:
                continue
            ctags = record_tags(record)
            overlap = tag_overlap(qtags, ctags)
            overlap_count = sum(len(v) for v in overlap.values())
            if tag_mode == "tag_filter" and query_has_useful_tags and overlap_count == 0:
                continue
            final_score = float(fused_score)
            if tag_mode == "tag_boost":
                final_score += tag_boost_weight * overlap_count
            candidates.append(
                (
                    final_score,
                    cid,
                    record,
                    overlap,
                    bm25_score_map.get(cid, 0.0),
                    dense_score_map.get(cid, 0.0),
                )
            )

        candidates.sort(key=lambda x: x[0], reverse=True)

        if rerank_top_n > 0 and candidates:
            rerank_pool = candidates[: max(top_k, min(rerank_top_n, len(candidates)))]
            reranked = []
            if reranker:
                pairs = [(query_text(query), record_text(item[2])) for item in rerank_pool]
                rerank_scores = reranker.predict(pairs, batch_size=rerank_batch_size, show_progress_bar=False)
                for (final_score, cid, record, overlap, bm25_score, dense_score), rr_score in zip(rerank_pool, rerank_scores):
                    rr = float(rr_score)
                    reranked.append((final_score + rerank_weight * rr, cid, record, overlap, bm25_score, dense_score, rr))
            else:
                for final_score, cid, record, overlap, bm25_score, dense_score in rerank_pool:
                    rr = lightweight_rerank_score(query, record)
                    reranked.append((final_score + rerank_weight * rr, cid, record, overlap, bm25_score, dense_score, rr))
            reranked.sort(key=lambda x: x[0], reverse=True)
            selected = reranked[:top_k]
        else:
            selected = [
                (final_score, cid, record, overlap, bm25_score, dense_score, None)
                for (final_score, cid, record, overlap, bm25_score, dense_score) in candidates[:top_k]
            ]

        for rank, (final_score, cid, record, overlap, bm25_score, dense_score, rerank_score) in enumerate(selected, start=1):
            metadata = record_metadata(record)
            rows.append(
                {
                    "method": method,
                    "query_id": qid,
                    "rank": rank,
                    "chunk_id": cid,
                    "score": float(final_score),
                    "vector_score": float(dense_score),
                    "debug_bm25_score": float(bm25_score),
                    "debug_fused_rrf_score": float(final_score if rerank_score is None else final_score - rerank_weight * float(rerank_score)),
                    "debug_tag_overlap": overlap,
                    "debug_rerank_score": rerank_score,
                    "source_pdf": metadata.get("source_pdf"),
                    "page": metadata.get("page"),
                    "modality": metadata.get("modality"),
                    "text_preview": clean_preview(record_text(record)),
                }
            )
    return rows


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
TAG_FIELDS_FOR_AWARE_MODES = ("organizations", "dates", "products", "locations", "named_entities")


TAG_MODE_NOTE = (
    "The paper says semantic tags help choose the right parts of the index, but does not specify "
    "a scoring formula. tag_boost/tag_filter modes here are implementation-specific benchmark variants."
)


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


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("\u00a0", " ")
    text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    text = re.sub(r"[^a-z0-9$%.,()\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(value: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9][a-z0-9$%.,()\-]*", normalize_text(value)))


def token_recall(needle: Any, haystack: Any) -> float:
    needle_tokens = token_set(needle)
    if not needle_tokens:
        return 0.0
    return len(needle_tokens & token_set(haystack)) / len(needle_tokens)


def lightweight_rerank_score(query: dict[str, Any], record: dict[str, Any]) -> float:
    metadata = record_metadata(record)
    score = 0.0
    qtype = query.get("type")
    modality = metadata.get("modality")
    if qtype and modality and qtype == modality:
        score += 0.08
    source_pdf = query.get("source_pdf")
    if source_pdf and metadata.get("source_pdf") == source_pdf:
        score += 0.06
    # lexical tie-break để ưu tiên candidate có phủ token câu hỏi tốt hơn
    score += 0.05 * token_recall(query_text(query), record_text(record))
    return score


def clean_preview(text: Any, limit: int = 350) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def normalize_tags(tags: Any) -> dict[str, list[str]]:
    normalized = {field: [] for field in SCHEMA_FIELDS}
    if not isinstance(tags, dict):
        return normalized
    for field in SCHEMA_FIELDS:
        values = tags.get(field)
        if values is None:
            continue
        if not isinstance(values, list):
            values = [values]
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                continue
            item = re.sub(r"\s+", " ", value).strip()
            if not item:
                continue
            key = normalize_tag_value(item)
            if key in seen:
                continue
            seen.add(key)
            normalized[field].append(item)
    return normalized


def normalize_tag_value(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"\b(incorporated|inc\.?|corporation|corp\.?|company|co\.?|llc|ltd\.?)\b", "", value)
    value = value.replace("&", "and")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def tag_overlap(query_tags: dict[str, list[str]], chunk_tags: dict[str, list[str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for field in SCHEMA_FIELDS:
        query_by_key = {normalize_tag_value(value): value for value in query_tags.get(field, [])}
        chunk_keys = {normalize_tag_value(value) for value in chunk_tags.get(field, [])}
        result[field] = [value for key, value in query_by_key.items() if key and key in chunk_keys]
    return result


def tag_overlap_count(query_tags: dict[str, list[str]], chunk_tags: dict[str, list[str]]) -> int:
    overlap = tag_overlap(query_tags, chunk_tags)
    return sum(len(overlap[field]) for field in TAG_FIELDS_FOR_AWARE_MODES)


def page_overlaps(metadata: dict[str, Any], page: int | None) -> bool:
    if page is None:
        return False
    start = metadata.get("page_start", metadata.get("page"))
    end = metadata.get("page_end", metadata.get("page"))
    try:
        start_i = int(start) if start is not None else None
        end_i = int(end) if end is not None else start_i
    except (TypeError, ValueError):
        return False
    return start_i is not None and end_i is not None and start_i <= page <= end_i


def chunk_id_from_record(record: dict[str, Any]) -> str:
    return str(record.get("chunk_id") or record.get("id") or record.get("metadata", {}).get("id"))


def record_text(record: dict[str, Any]) -> str:
    return str(record.get("text") or record.get("embed_text") or record.get("summary") or "")


def record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    excluded = {"text", "embed_text", "summary", "semantic_tags", "tags"}
    return {key: value for key, value in record.items() if key not in excluded}


def record_tags(record: dict[str, Any]) -> dict[str, list[str]]:
    if isinstance(record.get("semantic_tags"), dict):
        return normalize_tags(record["semantic_tags"])
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return normalize_tags(metadata.get("semantic_tags"))
    return normalize_tags(record.get("tags"))


def query_id(row: dict[str, Any], idx: int) -> str:
    for key in ("question_id", "query_id", "id"):
        if row.get(key):
            return str(row[key])
    return f"q_{idx}"


def query_text(row: dict[str, Any]) -> str:
    for key in ("question", "query", "text"):
        if row.get(key):
            return str(row[key])
    return ""


def query_tags(row: dict[str, Any]) -> dict[str, list[str]]:
    for key in ("query_semantic_tags", "semantic_tags", "tags"):
        if key in row:
            return normalize_tags(row[key])
    return normalize_tags(None)


def load_tagged_index(index_dir: Path) -> tuple[Any, list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    faiss = require_faiss()
    meta_path = index_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    index = faiss.read_index(str(index_dir / "chunks.faiss"))
    ids = json.loads((index_dir / "chunk_ids.json").read_text(encoding="utf-8"))
    records = load_jsonl(index_dir / "records.jsonl")
    records_by_id = {chunk_id_from_record(record): record for record in records}
    return index, ids, records_by_id, meta


def load_baseline_records(chunks_path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    for row in load_jsonl(chunks_path):
        cid = str(row.get("id") or row.get("chunk_id"))
        if not cid:
            continue
        text_parts = [row.get("embed_text"), row.get("text"), row.get("summary")]
        text = next((str(part) for part in text_parts if part), "")
        records[cid] = {
            "chunk_id": cid,
            "text": text,
            "metadata": record_metadata(row),
            "semantic_tags": normalize_tags(row.get("semantic_tags") or row.get("tags")),
        }
    return records


def load_baseline_index(index_path: Path, ids_path: Path, chunks_path: Path) -> tuple[Any, list[str], dict[str, dict[str, Any]]]:
    faiss = require_faiss()
    index = faiss.read_index(str(index_path))
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    records_by_id = load_baseline_records(chunks_path)
    return index, ids, records_by_id


def evidence_score(query: dict[str, Any], record: dict[str, Any]) -> float:
    metadata = record_metadata(record)
    if query.get("source_pdf") and metadata.get("source_pdf") != query.get("source_pdf"):
        return 0.0
    page = query.get("page")
    if page is not None and not page_overlaps(metadata, page):
        return 0.0
    text = record_text(record)
    evidence = query.get("evidence") or ""
    answer = query.get("answer") or ""
    score = 0.0
    ev_norm = normalize_text(evidence)
    text_norm = normalize_text(text)
    if ev_norm and ev_norm in text_norm:
        score += 20.0
    if answer and normalize_text(answer) in text_norm:
        score += 8.0
    score += 10.0 * token_recall(evidence, text)
    qtype = query.get("type")
    modality = metadata.get("modality")
    if qtype == modality:
        score += 2.0
    return score


def build_qrels(
    queries: list[dict[str, Any]],
    records_by_id: dict[str, dict[str, Any]],
    max_qrels: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = list(records_by_id.values())
    qrels: list[dict[str, Any]] = []
    misses: list[dict[str, Any]] = []
    for idx, query in enumerate(queries, start=1):
        qid = query_id(query, idx)
        scored = []
        for record in records:
            score = evidence_score(query, record)
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            misses.append(
                {
                    "query_id": qid,
                    "question": query_text(query),
                    "source_pdf": query.get("source_pdf"),
                    "page": query.get("page"),
                    "evidence": query.get("evidence"),
                    "reason": "no_evidence_chunk_match",
                }
            )
            continue
        for score, record in scored[:max_qrels]:
            metadata = record_metadata(record)
            qrels.append(
                {
                    "query_id": qid,
                    "chunk_id": chunk_id_from_record(record),
                    "relevance": 1,
                    "match_score": float(score),
                    "source_pdf": metadata.get("source_pdf"),
                    "page": metadata.get("page"),
                    "page_start": metadata.get("page_start", metadata.get("page")),
                    "page_end": metadata.get("page_end", metadata.get("page")),
                    "modality": metadata.get("modality"),
                }
            )
    return qrels, misses


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances))


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    rels = [1 if chunk_id in relevant_ids else 0 for chunk_id in retrieved_ids[:k]]
    actual = dcg(rels)
    ideal = dcg([1] * min(len(relevant_ids), k))
    return actual / ideal if ideal else 0.0


def make_retrieval_rows(
    *,
    method: str,
    queries: list[dict[str, Any]],
    index: Any,
    ids: list[str],
    records_by_id: dict[str, dict[str, Any]],
    query_vectors: np.ndarray,
    top_k: int,
    candidate_k: int,
    tag_mode: str,
    tag_boost_weight: float,
    reranker: Any = None,
    rerank_weight: float = 1.0,
    rerank_batch_size: int = 16,
) -> list[dict[str, Any]]:
    search_k = min(max(candidate_k, top_k), len(ids))
    scores, indices = index.search(query_vectors, search_k)
    rows: list[dict[str, Any]] = []
    for q_idx, (query, query_scores, query_indices) in enumerate(zip(queries, scores, indices), start=1):
        qid = query_id(query, q_idx)
        qtags = query_tags(query)
        candidates = []
        for score, raw_idx in zip(query_scores, query_indices):
            if raw_idx < 0:
                continue
            cid = ids[int(raw_idx)]
            record = records_by_id.get(cid)
            if not record:
                continue
            ctags = record_tags(record)
            overlap = tag_overlap(qtags, ctags)
            overlap_count = sum(len(values) for values in overlap.values())
            if tag_mode == "tag_filter_then_vector" and any(qtags[field] for field in TAG_FIELDS_FOR_AWARE_MODES):
                if not overlap_count:
                    continue
            final_score = float(score)
            if tag_mode == "vector_plus_tag_boost":
                final_score += tag_boost_weight * overlap_count
            candidates.append((final_score, float(score), cid, record, overlap))

        if reranker and candidates:
            pairs = [(query_text(query), record_text(item[3])) for item in candidates]
            rerank_scores = reranker.predict(pairs, batch_size=rerank_batch_size, show_progress_bar=False)
            reranked = []
            for (final_score, vector_score, cid, record, overlap), rr_score in zip(candidates, rerank_scores):
                rr = float(rr_score)
                reranked.append((final_score + rerank_weight * rr, vector_score, cid, record, overlap, rr))
            reranked.sort(key=lambda item: item[0], reverse=True)
            candidates_for_output = reranked[:top_k]
        else:
            candidates.sort(key=lambda item: item[0], reverse=True)
            candidates_for_output = [(f, v, c, r, o, None) for (f, v, c, r, o) in candidates[:top_k]]

        for rank, (final_score, vector_score, cid, record, overlap, rerank_score) in enumerate(candidates_for_output, start=1):
            metadata = record_metadata(record)
            rows.append(
                {
                    "method": method,
                    "query_id": qid,
                    "rank": rank,
                    "chunk_id": cid,
                    "score": float(final_score),
                    "vector_score": float(vector_score),
                    "debug_tag_overlap": overlap,
                    "debug_rerank_score": rerank_score,
                    "source_pdf": metadata.get("source_pdf"),
                    "page": metadata.get("page"),
                    "modality": metadata.get("modality"),
                    "text_preview": clean_preview(record_text(record)),
                }
            )
    return rows


def load_rag_sem_output(path: Path, method: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in load_jsonl(path):
        qid = str(line.get("question_id") or line.get("query_id") or line.get("id"))
        for chunk in line.get("retrieved_chunks", []):
            rows.append(
                {
                    "method": method,
                    "query_id": qid,
                    "rank": int(chunk.get("rank", 0)),
                    "chunk_id": str(chunk.get("chunk_id")),
                    "score": float(chunk.get("vector_score", chunk.get("score", 0.0))),
                    "vector_score": float(chunk.get("vector_score", chunk.get("score", 0.0))),
                    "debug_tag_overlap": chunk.get("debug_tag_overlap") or {},
                    "source_pdf": (chunk.get("metadata") or {}).get("source_pdf"),
                    "page": (chunk.get("metadata") or {}).get("page"),
                    "modality": (chunk.get("metadata") or {}).get("modality"),
                    "text_preview": chunk.get("text_preview"),
                }
            )
    return rows


def compute_metrics(
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    k_values: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relevant_by_qid: dict[str, set[str]] = defaultdict(set)
    for row in qrels:
        relevant_by_qid[row["query_id"]].add(row["chunk_id"])

    results_by_method_qid: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in retrieval_rows:
        results_by_method_qid[row["method"]][row["query_id"]].append(row)

    query_ids = [query_id(row, idx) for idx, row in enumerate(queries, start=1)]
    per_question: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for method, by_qid in sorted(results_by_method_qid.items()):
        method_question_rows = []
        for idx, query in enumerate(queries, start=1):
            qid = query_id(query, idx)
            relevant = relevant_by_qid.get(qid, set())
            retrieved = sorted(by_qid.get(qid, []), key=lambda row: row["rank"])
            retrieved_ids = [row["chunk_id"] for row in retrieved]
            row: dict[str, Any] = {
                "method": method,
                "query_id": qid,
                "question": query_text(query),
                "query_type": query.get("type"),
                "num_relevant": len(relevant),
                "retrieved_ids": retrieved_ids,
            }
            for k in k_values:
                top_ids = retrieved_ids[:k]
                hits = [cid for cid in top_ids if cid in relevant]
                row[f"precision@{k}"] = len(hits) / k
                row[f"recall@{k}"] = len(set(hits)) / len(relevant) if relevant else math.nan
                row[f"hit@{k}"] = 1.0 if hits else 0.0
                row[f"mrr@{k}"] = reciprocal_rank(retrieved_ids, relevant, k)
                row[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant, k)
            method_question_rows.append(row)
            per_question.append(row)

        summary_row: dict[str, Any] = {"method": method, "queries": len(query_ids)}
        for k in k_values:
            for metric in ("precision", "recall", "hit", "mrr", "ndcg"):
                key = f"{metric}@{k}"
                values = [row[key] for row in method_question_rows]
                values = [float(value) for value in values if not (isinstance(value, float) and math.isnan(value))]
                summary_row[key] = float(np.mean(values)) if values else None
        summary.append(summary_row)
    return summary, per_question


def debug_cases(
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    per_question: list[dict[str, Any]],
    max_cases: int,
) -> list[dict[str, Any]]:
    qrels_by_qid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retrieval_by_method_qid: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    query_by_id = {query_id(row, idx): row for idx, row in enumerate(queries, start=1)}
    for row in qrels:
        qrels_by_qid[row["query_id"]].append(row)
    for row in retrieval_rows:
        retrieval_by_method_qid[(row["method"], row["query_id"])].append(row)

    candidates = sorted(
        per_question,
        key=lambda row: (row.get("hit@5", 0.0), row.get("mrr@5", 0.0), row.get("recall@5", 0.0)),
    )
    output = []
    seen: set[tuple[str, str]] = set()
    for row in candidates:
        key = (row["method"], row["query_id"])
        if key in seen:
            continue
        seen.add(key)
        query = query_by_id.get(row["query_id"], {})
        retrieved = sorted(retrieval_by_method_qid.get(key, []), key=lambda item: item["rank"])[:5]
        relevant_ids = {rel["chunk_id"] for rel in qrels_by_qid.get(row["query_id"], [])}
        output.append(
            {
                "method": row["method"],
                "query_id": row["query_id"],
                "query": query_text(query),
                "query_type": query.get("type"),
                "ground_truth": qrels_by_qid.get(row["query_id"], [])[:8],
                "retrieved_chunks": [
                    {
                        **item,
                        "is_relevant": item["chunk_id"] in relevant_ids,
                    }
                    for item in retrieved
                ],
                "hit@5": row.get("hit@5"),
                "recall@5": row.get("recall@5"),
                "mrr@5": row.get("mrr@5"),
            }
        )
        if len(output) >= max_cases:
            break
    return output


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
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
    summary: list[dict[str, Any]],
    qrel_misses: list[dict[str, Any]],
    debug_rows: list[dict[str, Any]],
    k_values: list[int],
) -> None:
    cols = ["method", "queries"]
    for k in k_values:
        cols.extend([f"hit@{k}", f"recall@{k}", f"mrr@{k}", f"ndcg@{k}", f"precision@{k}"])
    lines = [
        "# RAG_SEM Retrieval Evaluation",
        "",
        "## Metrics",
        "",
        markdown_table(summary, cols),
        "",
        "## Notes",
        "",
        f"- {TAG_MODE_NOTE}",
        f"- Qrel mapping misses: {len(qrel_misses)}",
        "",
        "## Debug Miss Samples",
        "",
    ]
    for row in debug_rows[:5]:
        lines.append(f"### {row['method']} / {row['query_id']}")
        lines.append("")
        lines.append(f"- query: {row['query']}")
        lines.append(f"- hit@5: {row['hit@5']} recall@5: {row['recall@5']} mrr@5: {row['mrr@5']}")
        lines.append("- retrieved:")
        for item in row["retrieved_chunks"]:
            lines.append(
                f"  - rank {item['rank']} id={item['chunk_id']} relevant={item['is_relevant']} score={item['score']:.4f} preview={clean_preview(item.get('text_preview'), 160)}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Evaluate baseline vs RAG_SEM tagged retrieval without answer generation.")
    parser.add_argument("--queries", type=Path, default=Path("data/qa/test_qa_tagged.jsonl"))
    parser.add_argument("--rag-sem-output", type=Path, default=Path("outputs/rag_sem_retrieval_results.jsonl"))
    parser.add_argument("--tagged-index-dir", type=Path, default=Path("data/index_tagged_chunks"))
    parser.add_argument("--baseline-index", type=Path, default=Path("data/index_bge/all.faiss"))
    parser.add_argument("--baseline-ids", type=Path, default=Path("data/index_bge/all_chunk_ids.json"))
    parser.add_argument("--baseline-chunks", type=Path, default=Path("data/chunks/all_chunks.jsonl"))
    parser.add_argument("--text-index", type=Path, default=Path("data/index_bge/text.faiss"))
    parser.add_argument("--text-ids", type=Path, default=Path("data/index_bge/text_chunk_ids.json"))
    parser.add_argument("--text-chunks", type=Path, default=Path("data/chunks/text_chunks.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/rag_sem_retrieval_eval"))
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--max-qrels", type=int, default=5)
    parser.add_argument("--query-embedding-model", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device")
    parser.add_argument("--tag-boost-weight", type=float, default=0.03)
    parser.add_argument(
        "--include-tag-aware",
        action="store_true",
        help="Thêm các mode tag_boost và tag_filter vào benchmark.",
    )
    parser.add_argument(
        "--include-bm25",
        action="store_true",
        help="Thêm BM25 lexical retrieval vào benchmark.",
    )
    parser.add_argument(
        "--include-text-only",
        action="store_true",
        help="Thêm Text-only BGE (chỉ index text, không table/image) vào benchmark.",
    )
    parser.add_argument(
        "--include-hybrid-rrf",
        action="store_true",
        help="Thêm Hybrid search bằng Reciprocal Rank Fusion (BM25 + Dense BGE).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="Hằng số k trong công thức RRF: 1/(k+rank).",
    )
    parser.add_argument(
        "--hybrid-rerank-model",
        default=None,
        help="Model CrossEncoder để rerank top-N sau khi RRF fusion.",
    )
    parser.add_argument(
        "--hybrid-rerank-top-n",
        type=int,
        default=50,
        help="Số candidate top-N của RRF để đưa vào CrossEncoder rerank.",
    )
    parser.add_argument(
        "--hybrid-rerank-weight",
        type=float,
        default=1.0,
        help="Trọng số cộng điểm rerank cho Hybrid RRF.",
    )
    parser.add_argument(
        "--hybrid-rerank-batch-size",
        type=int,
        default=16,
        help="Batch size cho Hybrid RRF rerank.",
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="Tên model CrossEncoder để rerank (ví dụ: BAAI/bge-reranker-base).",
    )
    parser.add_argument(
        "--rerank-weight",
        type=float,
        default=1.0,
        help="Trọng số cộng vào final score khi dùng rerank model.",
    )
    parser.add_argument(
        "--rerank-batch-size",
        type=int,
        default=16,
        help="Batch size cho CrossEncoder rerank.",
    )
    args = parser.parse_args()

    queries = load_jsonl(args.queries)
    max_k = max(args.k_values)
    top_k = max(max_k, 10)

    tagged_index, tagged_ids, tagged_records, tagged_meta = load_tagged_index(args.tagged_index_dir)
    baseline_index, baseline_ids, baseline_records = load_baseline_index(args.baseline_index, args.baseline_ids, args.baseline_chunks)
    qrels, qrel_misses = build_qrels(queries, tagged_records, max_qrels=args.max_qrels)

    model_name = args.query_embedding_model or tagged_meta.get("embedding_model") or DEFAULT_BGE_MODEL
    embedder = load_bge_model(model_name=model_name, batch_size=args.batch_size, device=args.device)
    query_vectors = embedder.encode_queries([query_text(row) for row in queries])
    reranker = _load_cross_encoder(args.rerank_model)
    hybrid_reranker = _load_cross_encoder(args.hybrid_rerank_model)

    retrieval_rows: list[dict[str, Any]] = []
    bm25: BM25Index | None = None

    # BM25 và các biến thể tag-aware
    if args.include_bm25:
        print("Building BM25 index...")
        bm25 = build_bm25_index(baseline_records)
        # BM25 thuần
        retrieval_rows += make_bm25_rows(
            method="bm25",
            queries=queries,
            bm25=bm25,
            top_k=top_k,
            candidate_k=args.candidate_k,
            records_by_id=baseline_records,
            tag_mode="none",
            reranker=reranker,
            rerank_weight=args.rerank_weight,
            rerank_batch_size=args.rerank_batch_size,
        )
        if args.include_tag_aware:
            # BM25 + Tag Boost
            retrieval_rows += make_bm25_rows(
                method="bm25_tag_boost",
                queries=queries,
                bm25=bm25,
                top_k=top_k,
                candidate_k=args.candidate_k,
                records_by_id=tagged_records,
                tag_mode="tag_boost",
                tag_boost_weight=args.tag_boost_weight,
                reranker=reranker,
                rerank_weight=args.rerank_weight,
                rerank_batch_size=args.rerank_batch_size,
            )
            # BM25 + Tag Filter
            retrieval_rows += make_bm25_rows(
                method="bm25_tag_filter",
                queries=queries,
                bm25=bm25,
                top_k=top_k,
                candidate_k=args.candidate_k,
                records_by_id=tagged_records,
                tag_mode="tag_filter",
                tag_boost_weight=args.tag_boost_weight,
                reranker=reranker,
                rerank_weight=args.rerank_weight,
                rerank_batch_size=args.rerank_batch_size,
            )

    if args.include_hybrid_rrf:
        if bm25 is None:
            print("Building BM25 index for hybrid RRF...")
            bm25 = build_bm25_index(baseline_records)
        retrieval_rows += make_hybrid_rrf_rows(
            method="hybrid_rrf_bm25_dense",
            queries=queries,
            bm25=bm25,
            dense_index=baseline_index,
            dense_ids=baseline_ids,
            records_by_id=baseline_records,
            query_vectors=query_vectors,
            top_k=top_k,
            candidate_k=args.candidate_k,
            rrf_k=args.rrf_k,
            tag_mode="none",
            reranker=hybrid_reranker,
            rerank_top_n=args.hybrid_rerank_top_n,
            rerank_weight=args.hybrid_rerank_weight,
            rerank_batch_size=args.hybrid_rerank_batch_size,
        )
        if args.include_tag_aware:
            retrieval_rows += make_hybrid_rrf_rows(
                method="hybrid_rrf_bm25_dense_tag_boost",
                queries=queries,
                bm25=bm25,
                dense_index=baseline_index,
                dense_ids=baseline_ids,
                records_by_id=tagged_records,
                query_vectors=query_vectors,
                top_k=top_k,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
                tag_mode="tag_boost",
                tag_boost_weight=args.tag_boost_weight,
                reranker=hybrid_reranker,
                rerank_top_n=args.hybrid_rerank_top_n,
                rerank_weight=args.hybrid_rerank_weight,
                rerank_batch_size=args.hybrid_rerank_batch_size,
            )
            retrieval_rows += make_hybrid_rrf_rows(
                method="hybrid_rrf_bm25_dense_tag_filter",
                queries=queries,
                bm25=bm25,
                dense_index=baseline_index,
                dense_ids=baseline_ids,
                records_by_id=tagged_records,
                query_vectors=query_vectors,
                top_k=top_k,
                candidate_k=args.candidate_k,
                rrf_k=args.rrf_k,
                tag_mode="tag_filter",
                tag_boost_weight=args.tag_boost_weight,
                reranker=hybrid_reranker,
                rerank_top_n=args.hybrid_rerank_top_n,
                rerank_weight=args.hybrid_rerank_weight,
                rerank_batch_size=args.hybrid_rerank_batch_size,
            )

    # Text-only BGE (chỉ text chunks, không table/image)
    if args.include_text_only and args.text_index.exists():
        text_index, text_ids, text_records = load_baseline_index(args.text_index, args.text_ids, args.text_chunks)
        retrieval_rows += make_retrieval_rows(
            method="text_only_bge",
            queries=queries,
            index=text_index,
            ids=text_ids,
            records_by_id=text_records,
            query_vectors=query_vectors,
            top_k=top_k,
            candidate_k=args.candidate_k,
            tag_mode="vector_only",
            tag_boost_weight=args.tag_boost_weight,
            reranker=reranker,
            rerank_weight=args.rerank_weight,
            rerank_batch_size=args.rerank_batch_size,
        )

    # Dense BGE (text+table+image, không tag)
    retrieval_rows += make_retrieval_rows(
        method="dense_bge",
        queries=queries,
        index=baseline_index,
        ids=baseline_ids,
        records_by_id=baseline_records,
        query_vectors=query_vectors,
        top_k=top_k,
        candidate_k=args.candidate_k,
        tag_mode="vector_only",
        tag_boost_weight=args.tag_boost_weight,
        reranker=reranker,
        rerank_weight=args.rerank_weight,
        rerank_batch_size=args.rerank_batch_size,
    )

    # RAG_SEM vector only (tagged index, không dùng tag khi search)
    retrieval_rows += make_retrieval_rows(
        method="rag_sem_vector_only",
        queries=queries,
        index=tagged_index,
        ids=tagged_ids,
        records_by_id=tagged_records,
        query_vectors=query_vectors,
        top_k=top_k,
        candidate_k=args.candidate_k,
        tag_mode="vector_only",
        tag_boost_weight=args.tag_boost_weight,
        reranker=reranker,
        rerank_weight=args.rerank_weight,
        rerank_batch_size=args.rerank_batch_size,
    )

    existing_rag_rows = load_rag_sem_output(args.rag_sem_output, "rag_sem_existing_output_topk")
    if existing_rag_rows:
        retrieval_rows += existing_rag_rows

    if args.include_tag_aware:
        # Dense BGE + Tag Boost
        retrieval_rows += make_retrieval_rows(
            method="dense_bge_tag_boost",
            queries=queries,
            index=tagged_index,
            ids=tagged_ids,
            records_by_id=tagged_records,
            query_vectors=query_vectors,
            top_k=top_k,
            candidate_k=args.candidate_k,
            tag_mode="vector_plus_tag_boost",
            tag_boost_weight=args.tag_boost_weight,
            reranker=reranker,
            rerank_weight=args.rerank_weight,
            rerank_batch_size=args.rerank_batch_size,
        )
        # Dense BGE + Tag Filter
        retrieval_rows += make_retrieval_rows(
            method="dense_bge_tag_filter",
            queries=queries,
            index=tagged_index,
            ids=tagged_ids,
            records_by_id=tagged_records,
            query_vectors=query_vectors,
            top_k=top_k,
            candidate_k=args.candidate_k,
            tag_mode="tag_filter_then_vector",
            tag_boost_weight=args.tag_boost_weight,
            reranker=reranker,
            rerank_weight=args.rerank_weight,
            rerank_batch_size=args.rerank_batch_size,
        )

    summary, per_question = compute_metrics(queries, qrels, retrieval_rows, args.k_values)
    debug_rows = debug_cases(queries, qrels, retrieval_rows, per_question, max_cases=20)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "qrels.jsonl", qrels)
    write_jsonl(args.out_dir / "qrel_misses.jsonl", qrel_misses)
    write_jsonl(args.out_dir / "retrieval_results_all_modes.jsonl", retrieval_rows)
    write_jsonl(args.out_dir / "metrics_by_question.jsonl", per_question)
    write_jsonl(args.out_dir / "debug_cases.jsonl", debug_rows)
    write_json(args.out_dir / "metrics_summary.json", {"summary": summary, "tag_mode_note": TAG_MODE_NOTE})
    write_csv(args.out_dir / "metrics_summary.csv", summary)
    write_report(args.out_dir / "retrieval_eval_report.md", summary, qrel_misses, debug_rows, args.k_values)

    print("RAG_SEM retrieval evaluation summary:")
    print(markdown_table(summary, ["method", "hit@5", "recall@5", "mrr@5", "hit@10", "recall@10", "mrr@10", "ndcg@10"]))
    print(f"\nOutputs: {args.out_dir}")
    print(f"Qrel misses: {len(qrel_misses)}")
    print(TAG_MODE_NOTE)


if __name__ == "__main__":
    main()
