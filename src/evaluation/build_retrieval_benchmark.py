from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


DEFAULT_QA_PATH = Path("data/qa/eval_qa.jsonl")
DEFAULT_CHUNKS_PATH = Path("data/chunks/all_chunks.jsonl")
DEFAULT_OUT_DIR = Path("data/benchmark")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize_text(text: Any) -> str:
    value = str(text or "").lower()
    value = value.replace("\u00a0", " ")
    value = re.sub(r"[^a-z0-9$%.\-()/]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def token_set(text: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9$%.\-()/]+", normalize_text(text)))


def token_recall(needle: Any, haystack: Any) -> float:
    needle_tokens = token_set(needle)
    if not needle_tokens:
        return 0.0
    return len(needle_tokens & token_set(haystack)) / len(needle_tokens)


def sequence_ratio(needle: Any, haystack: Any) -> float:
    n = normalize_text(needle)
    h = normalize_text(haystack)
    if not n or not h:
        return 0.0
    if len(h) > 12000:
        h = h[:12000]
    return SequenceMatcher(None, n, h).ratio()


def chunk_content(chunk: dict[str, Any]) -> str:
    parts = [
        chunk.get("text"),
        chunk.get("summary"),
        chunk.get("table_markdown"),
        json.dumps(chunk.get("table_json"), ensure_ascii=False) if chunk.get("table_json") else None,
        chunk.get("image_path"),
    ]
    return "\n".join(str(part) for part in parts if part)


def page_overlaps(chunk: dict[str, Any], page: int | None) -> bool:
    if page is None:
        return False
    start = chunk.get("page_start", chunk.get("page"))
    end = chunk.get("page_end", chunk.get("page"))
    try:
        start_i = int(start) if start is not None else None
        end_i = int(end) if end is not None else start_i
    except (TypeError, ValueError):
        return False
    return start_i is not None and end_i is not None and start_i <= page <= end_i


def source_matches(chunk: dict[str, Any], source_pdf: str | None) -> bool:
    return bool(source_pdf) and chunk.get("source_pdf") == source_pdf


def score_candidate(item: dict[str, Any], chunk: dict[str, Any]) -> float:
    content = chunk_content(chunk)
    evidence = item.get("evidence", "")
    question = item.get("question", "")
    answer = item.get("answer", "")
    score = 0.0
    score += 0.45 * token_recall(evidence, content)
    score += 0.20 * sequence_ratio(evidence, content)
    score += 0.20 * token_recall(question, content)
    score += 0.15 * token_recall(answer, content)
    return score


def desired_modalities(qtype: str) -> set[str]:
    if qtype == "text":
        return {"text"}
    if qtype == "table":
        return {"table"}
    if qtype == "image":
        return {"image"}
    if qtype == "multimodal":
        return {"text", "table", "image"}
    return {"text", "table", "image"}


def map_item_to_chunks(
    item: dict[str, Any],
    chunks_by_source: dict[str, list[dict[str, Any]]],
    min_score: float,
    max_qrels_per_query: int,
) -> tuple[list[dict[str, Any]], str]:
    qtype = item.get("type", "unknown")
    source_pdf = item.get("source_pdf")
    page = item.get("page")
    candidates = [
        chunk
        for chunk in chunks_by_source.get(source_pdf, [])
        if chunk.get("modality") in desired_modalities(qtype)
    ]

    evidence_chunk_id = item.get("evidence_chunk_id")
    if evidence_chunk_id:
        exact = [chunk for chunk in candidates if chunk.get("id") == evidence_chunk_id or chunk.get("chunk_id") == evidence_chunk_id]
        if exact:
            return [{"chunk": exact[0], "score": 1.0, "method": "given_evidence_chunk_id"}], "mapped"

    page_candidates = [chunk for chunk in candidates if page_overlaps(chunk, page)]
    if page_candidates:
        candidates = page_candidates

    scored = []
    for chunk in candidates:
        score = score_candidate(item, chunk)
        if score >= min_score:
            scored.append({"chunk": chunk, "score": score, "method": "content_page_match" if page_candidates else "content_match"})

    scored.sort(key=lambda row: row["score"], reverse=True)
    if scored:
        return scored[:max_qrels_per_query], "mapped"
    return [], "no_chunk_match"


def visual_kind(summary: str) -> str | None:
    text = summary.lower()
    if "bar chart" in text:
        return "bar chart"
    if "line graph" in text or "line chart" in text or "multiple lines" in text:
        return "line chart"
    if "pie chart" in text:
        return "pie chart"
    if "table" in text:
        return "table-like chart"
    if "chart" in text:
        return "chart"
    return None


def company_from_source(source_pdf: str) -> str:
    name = Path(source_pdf).name
    ticker = name.split("_", 1)[0]
    return {
        "AAPL": "Apple",
        "HD": "Home Depot",
        "INTU": "Intuit",
        "MS": "Morgan Stanley",
        "NVDA": "NVIDIA",
    }.get(ticker, ticker)


def build_image_object_queries(
    image_chunks: list[dict[str, Any]],
    existing_query_count: int,
    target_count: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    used_sources = Counter()
    selected = []
    for chunk in image_chunks:
        summary = str(chunk.get("summary") or "")
        kind = visual_kind(summary)
        if not kind:
            continue
        # Prefer image chunks that have enough descriptive text to be retrievable.
        if len(summary.split()) < 18:
            continue
        selected.append((chunk, kind))

    selected.sort(key=lambda pair: (used_sources[pair[0].get("source_pdf")], pair[0].get("source_pdf") or "", pair[0].get("id") or ""))
    for chunk, kind in selected:
        if len(queries) >= target_count:
            break
        source_pdf = chunk.get("source_pdf")
        company = company_from_source(source_pdf or "")
        query_id = f"img{existing_query_count + len(queries) + 1:03d}"
        question = f"Which {company} filing image shows a {kind}?"
        queries.append(
            {
                "query_id": query_id,
                "question": question,
                "type": "image",
                "answer": kind,
                "answer_type": "text",
                "source_pdf": source_pdf,
                "evidence_chunk_ids": [chunk["id"]],
                "gold_image_path": chunk.get("image_path"),
                "gold_page": chunk.get("page"),
                "benchmark_source": "image_object_generated",
            }
        )
        qrels.append(
            {
                "query_id": query_id,
                "chunk_id": chunk["id"],
                "relevance": 1,
                "modality": "image",
                "match_method": "image_object_gold",
                "source_pdf": source_pdf,
                "page": chunk.get("page"),
                "image_path": chunk.get("image_path"),
            }
        )
        used_sources[source_pdf] += 1
    return queries, qrels


def corpus_row(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["id"],
        "modality": chunk.get("modality"),
        "source_pdf": chunk.get("source_pdf"),
        "source_html": chunk.get("source_html"),
        "page": chunk.get("page"),
        "page_start": chunk.get("page_start", chunk.get("page")),
        "page_end": chunk.get("page_end", chunk.get("page")),
        "text": chunk_content(chunk),
        "image_path": chunk.get("image_path"),
    }


def build_benchmark(
    qa_path: Path,
    chunks_path: Path,
    out_dir: Path,
    min_score: float,
    max_qrels_per_query: int,
    image_object_queries: int,
) -> dict[str, Any]:
    qa_items = load_jsonl(qa_path)
    chunks = load_jsonl(chunks_path)
    chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_source[chunk.get("source_pdf")].append(chunk)

    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for item in qa_items:
        # Existing image QA is mostly page/checkbox/marker based. It is not a
        # reliable image-object benchmark unless it already has a gold image id.
        if item.get("type") == "image" and not item.get("evidence_chunk_id") and not item.get("image_path"):
            unmatched.append(
                {
                    "query_id": item.get("id"),
                    "type": item.get("type"),
                    "reason": "image_item_missing_evidence_chunk_id_or_image_path",
                    "question": item.get("question"),
                    "source_pdf": item.get("source_pdf"),
                    "page": item.get("page"),
                }
            )
            continue

        matches, reason = map_item_to_chunks(item, chunks_by_source, min_score, max_qrels_per_query)
        if not matches:
            unmatched.append(
                {
                    "query_id": item.get("id"),
                    "type": item.get("type"),
                    "reason": reason,
                    "question": item.get("question"),
                    "source_pdf": item.get("source_pdf"),
                    "page": item.get("page"),
                }
            )
            continue

        evidence_ids = [match["chunk"]["id"] for match in matches]
        queries.append(
            {
                "query_id": item["id"],
                "question": item["question"],
                "type": item.get("type"),
                "answer": item.get("answer"),
                "answer_type": item.get("answer_type", "text"),
                "source_pdf": item.get("source_pdf"),
                "gold_page": item.get("page"),
                "evidence": item.get("evidence"),
                "evidence_chunk_ids": evidence_ids,
                "benchmark_source": "qa_mapped_to_chunk",
            }
        )
        for match in matches:
            chunk = match["chunk"]
            qrels.append(
                {
                    "query_id": item["id"],
                    "chunk_id": chunk["id"],
                    "relevance": 1,
                    "modality": chunk.get("modality"),
                    "match_method": match["method"],
                    "match_score": round(float(match["score"]), 6),
                    "source_pdf": chunk.get("source_pdf"),
                    "page": chunk.get("page"),
                    "page_start": chunk.get("page_start", chunk.get("page")),
                    "page_end": chunk.get("page_end", chunk.get("page")),
                }
            )

    image_chunks = [chunk for chunk in chunks if chunk.get("modality") == "image"]
    image_queries, image_qrels = build_image_object_queries(
        image_chunks=image_chunks,
        existing_query_count=len(queries),
        target_count=image_object_queries,
    )
    queries.extend(image_queries)
    qrels.extend(image_qrels)

    relevant_chunk_ids = {row["chunk_id"] for row in qrels}
    corpus = [corpus_row(chunk) for chunk in chunks if chunk["id"] in relevant_chunk_ids]
    # Retrieval needs a full corpus. Keep a second compact gold corpus for audit.
    full_corpus = [corpus_row(chunk) for chunk in chunks]

    write_jsonl(out_dir / "queries.jsonl", queries)
    write_jsonl(out_dir / "qrels.jsonl", qrels)
    write_jsonl(out_dir / "corpus.jsonl", full_corpus)
    write_jsonl(out_dir / "gold_corpus.jsonl", corpus)
    write_jsonl(out_dir / "unmatched.jsonl", unmatched)

    qrels_per_query = Counter(row["query_id"] for row in qrels)
    summary = {
        "qa_path": str(qa_path).replace("\\", "/"),
        "chunks_path": str(chunks_path).replace("\\", "/"),
        "queries": len(queries),
        "qrels": len(qrels),
        "full_corpus_chunks": len(full_corpus),
        "gold_corpus_chunks": len(corpus),
        "unmatched": len(unmatched),
        "query_type_counts": dict(Counter(row.get("type") for row in queries)),
        "qrel_modality_counts": dict(Counter(row.get("modality") for row in qrels)),
        "qrels_per_query": {
            "min": min(qrels_per_query.values()) if qrels_per_query else 0,
            "max": max(qrels_per_query.values()) if qrels_per_query else 0,
            "avg": sum(qrels_per_query.values()) / len(qrels_per_query) if qrels_per_query else 0,
            "distribution": dict(Counter(qrels_per_query.values())),
        },
        "settings": {
            "min_score": min_score,
            "max_qrels_per_query": max_qrels_per_query,
            "image_object_queries": image_object_queries,
        },
        "note": (
            "Use corpus.jsonl as the retrieval corpus and qrels.jsonl as gold labels. "
            "gold_corpus.jsonl is only for audit. Existing page-only image QA rows are "
            "excluded unless they provide evidence_chunk_id/image_path; generated image "
            "object queries are added from image chunks."
        ),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build chunk-level retrieval benchmark files.")
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-score", type=float, default=0.18)
    parser.add_argument("--max-qrels-per-query", type=int, default=2)
    parser.add_argument("--image-object-queries", type=int, default=25)
    args = parser.parse_args()

    summary = build_benchmark(
        qa_path=args.qa,
        chunks_path=args.chunks,
        out_dir=args.out_dir,
        min_score=args.min_score,
        max_qrels_per_query=args.max_qrels_per_query,
        image_object_queries=args.image_object_queries,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
