from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.bge_embedder import DEFAULT_BGE_MODEL, load_bge_model


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
QUERY_TAG_FIELDS = ("query_semantic_tags", "semantic_tags", "tags")


def require_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing faiss-cpu. Install it with: python -m pip install faiss-cpu") from exc
    return faiss


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_tags(tags: Any) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {field: [] for field in SCHEMA_FIELDS}
    if not isinstance(tags, dict):
        return normalized
    for field in SCHEMA_FIELDS:
        value = tags.get(field)
        if value is None:
            continue
        values = value if isinstance(value, list) else [value]
        seen: set[str] = set()
        for item in values:
            if not isinstance(item, str):
                continue
            text = " ".join(item.split()).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized[field].append(text)
    return normalized


def tag_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).casefold()


def debug_tag_overlap(question_tags: dict[str, list[str]], chunk_tags: dict[str, list[str]]) -> dict[str, list[str]]:
    overlap: dict[str, list[str]] = {}
    for field in SCHEMA_FIELDS:
        query_values = {tag_key(value): value for value in question_tags.get(field, [])}
        chunk_keys = {tag_key(value) for value in chunk_tags.get(field, [])}
        overlap[field] = [value for key, value in query_values.items() if key in chunk_keys]
    return overlap


def text_preview(text: str, limit: int = 700) -> str:
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def query_id(row: dict[str, Any], fallback_index: int) -> str:
    for field in ("question_id", "query_id", "id"):
        if row.get(field) is not None and str(row[field]).strip():
            return str(row[field])
    return f"query_{fallback_index}"


def query_text(row: dict[str, Any]) -> str:
    for field in ("question", "query", "text"):
        if row.get(field):
            return str(row[field])
    return ""


def query_tags(row: dict[str, Any]) -> tuple[str | None, dict[str, list[str]]]:
    for field in QUERY_TAG_FIELDS:
        if field in row:
            return field, normalize_tags(row.get(field))
    return None, normalize_tags(None)


def chunk_id(record: dict[str, Any]) -> str:
    value = record.get("chunk_id") or record.get("id") or record.get("metadata", {}).get("id")
    return str(value)


def chunk_tags(record: dict[str, Any]) -> dict[str, list[str]]:
    if isinstance(record.get("semantic_tags"), dict):
        return normalize_tags(record["semantic_tags"])
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        return normalize_tags(metadata.get("semantic_tags"))
    return normalize_tags(None)


def discover_index_candidates(root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for meta_path in root.rglob("meta.json"):
        index_dir = meta_path.parent
        index_path = index_dir / "chunks.faiss"
        ids_path = index_dir / "chunk_ids.json"
        records_path = index_dir / "records.jsonl"
        if not (index_path.exists() and ids_path.exists() and records_path.exists()):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        score = 10
        reasons = ["has chunks.faiss, chunk_ids.json, records.jsonl, meta.json"]
        if set(meta.get("tag_schema_fields") or []) == SCHEMA_FIELD_SET:
            score += 20
            reasons.append("meta tag_schema_fields match the 11-field semantic tag schema")
        if meta.get("backend") == "faiss":
            score += 5
            reasons.append("backend is faiss")
        if meta.get("chunks_indexed"):
            score += 5
            reasons.append(f"chunks_indexed={meta.get('chunks_indexed')}")
        candidates.append({"path": str(index_dir), "score": score, "reasons": reasons})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def discover_query_candidates(root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in root.rglob("*.jsonl"):
        lowered = str(path).lower()
        if any(skip in lowered for skip in ("retrieval", "records.jsonl", "index_errors", "audit_schema")):
            continue
        try:
            rows = load_jsonl(path)[:50]
        except (OSError, json.JSONDecodeError):
            continue
        if not rows:
            continue
        tag_fields = Counter()
        question_like = 0
        explicit_question_like = 0
        exact_schema = 0
        for row in rows:
            if row.get("question") or row.get("query"):
                explicit_question_like += 1
            if query_text(row):
                question_like += 1
            field, tags = query_tags(row)
            if field:
                tag_fields[field] += 1
                if set(tags) == SCHEMA_FIELD_SET:
                    exact_schema += 1
        if question_like and tag_fields:
            score = question_like * 3 + explicit_question_like * 20 + sum(tag_fields.values()) * 5 + exact_schema * 5
            if tag_fields.get("query_semantic_tags"):
                score += tag_fields["query_semantic_tags"] * 15
            path_lower = str(path).lower().replace("\\", "/")
            if "/qa/" in path_lower or "queries" in path_lower or "questions" in path_lower:
                score += 100
            if "/data/chunks/" in path_lower:
                score -= 100
            reasons = [
                f"{question_like}/{len(rows)} sampled records have question/query/text",
                f"{explicit_question_like}/{len(rows)} sampled records have explicit question/query",
                f"tag containers found: {dict(tag_fields)}",
                f"{exact_schema}/{len(rows)} sampled records normalize to the 11-field schema",
            ]
            candidates.append({"path": str(path), "score": score, "reasons": reasons})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def choose_path(explicit: Path | None, candidates: list[dict[str, Any]], label: str) -> Path:
    if explicit is not None:
        return explicit
    if not candidates:
        raise FileNotFoundError(f"No {label} candidate found. Pass --{label} explicitly.")
    return Path(candidates[0]["path"])


def print_candidates(title: str, candidates: list[dict[str, Any]], selected: Path) -> None:
    print(f"{title}:")
    for item in candidates[:8]:
        marker = " SELECTED" if Path(item["path"]).resolve() == selected.resolve() else ""
        print(f"- {item['path']}{marker}")
        print(f"  score={item['score']}")
        for reason in item["reasons"][:4]:
            print(f"  reason: {reason}")


def load_index_bundle(index_dir: Path) -> tuple[Any, list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    faiss = require_faiss()
    meta_path = index_dir / "meta.json"
    index_path = index_dir / "chunks.faiss"
    ids_path = index_dir / "chunk_ids.json"
    records_path = index_dir / "records.jsonl"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    index = faiss.read_index(str(index_path))
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    records = load_jsonl(records_path)
    records_by_id = {chunk_id(record): record for record in records}
    if len(ids) != index.ntotal:
        raise RuntimeError(f"Index vector count {index.ntotal} does not match chunk_ids count {len(ids)}.")
    missing = [cid for cid in ids if cid not in records_by_id]
    if missing:
        raise RuntimeError(f"records.jsonl is missing {len(missing)} chunk ids; first missing id={missing[0]}")
    return index, ids, records_by_id, meta


def retrieve(
    *,
    index_dir: Path,
    queries_path: Path,
    output_path: Path,
    top_k: int,
    query_embedding_model: str | None,
    limit: int | None,
    batch_size: int,
    device: str | None,
) -> dict[str, Any]:
    index, ids, records_by_id, meta = load_index_bundle(index_dir)
    model_name = query_embedding_model or meta.get("embedding_model") or meta.get("model_name") or DEFAULT_BGE_MODEL

    queries = load_jsonl(queries_path)
    if limit is not None:
        queries = queries[:limit]

    prepared: list[dict[str, Any]] = []
    for idx, row in enumerate(queries, start=1):
        question = query_text(row)
        if not question.strip():
            continue
        tag_field, tags = query_tags(row)
        prepared.append(
            {
                "question_id": query_id(row, idx),
                "question": question,
                "question_tags": tags,
                "tag_field": tag_field,
            }
        )

    if not prepared:
        raise RuntimeError("No valid tagged queries/questions to retrieve.")

    embedder = load_bge_model(model_name=model_name, batch_size=batch_size, device=device)
    query_vectors = embedder.encode_queries([row["question"] for row in prepared])
    search_k = min(top_k, len(ids))
    scores, indices = index.search(query_vectors, search_k)

    output_rows: list[dict[str, Any]] = []
    success_count = 0
    for query_row, query_scores, query_indices in zip(prepared, scores, indices):
        retrieved_chunks: list[dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(query_scores, query_indices), start=1):
            if idx < 0:
                continue
            cid = ids[int(idx)]
            record = records_by_id[cid]
            tags = chunk_tags(record)
            retrieved_chunks.append(
                {
                    "rank": rank,
                    "chunk_id": cid,
                    "vector_score": float(score),
                    "chunk_tags": tags,
                    "debug_tag_overlap": debug_tag_overlap(query_row["question_tags"], tags),
                    "metadata": record.get("metadata") or {},
                    "text_preview": text_preview(record.get("text") or ""),
                }
            )
        if retrieved_chunks:
            success_count += 1
        output_rows.append(
            {
                "question_id": query_row["question_id"],
                "question": query_row["question"],
                "question_tags": query_row["question_tags"],
                "retrieved_chunks": retrieved_chunks,
            }
        )

    write_jsonl(output_path, output_rows)
    sample = output_rows[0] if output_rows else None
    if sample:
        sample = dict(sample)
        sample["retrieved_chunks"] = sample["retrieved_chunks"][:3]
    return {
        "index_dir": str(index_dir).replace("\\", "/"),
        "queries_file": str(queries_path).replace("\\", "/"),
        "queries_loaded": len(prepared),
        "queries_retrieved_successfully": success_count,
        "top_k": top_k,
        "output_path": str(output_path).replace("\\", "/"),
        "query_embedding_model": model_name,
        "embedding_backend": meta.get("backend") or "faiss",
        "sample_top3": sample,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Run RAG_SEM retrieval over tagged FAISS chunks. This does not generate answers."
    )
    parser.add_argument("--index-dir", type=Path, help="Tagged vector index dir. Auto-detected when omitted.")
    parser.add_argument("--queries", type=Path, help="Tagged queries/questions JSONL. Auto-detected when omitted.")
    parser.add_argument("--output", type=Path, default=Path("outputs/rag_sem_retrieval_results.jsonl"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query-embedding-model", help="Defaults to embedding_model from index meta.json.")
    parser.add_argument("--limit", type=int, help="Optional number of queries for quick debug runs.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="Example: cuda, cpu. Leave empty for sentence-transformers auto device.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    index_candidates = discover_index_candidates(project_root)
    query_candidates = discover_query_candidates(project_root)

    index_dir = choose_path(args.index_dir, index_candidates, "index-dir")
    queries_path = choose_path(args.queries, query_candidates, "queries")
    if not index_dir.is_absolute():
        index_dir = (project_root / index_dir).resolve()
    if not queries_path.is_absolute():
        queries_path = (project_root / queries_path).resolve()
    output_path = args.output if args.output.is_absolute() else (project_root / args.output).resolve()

    print_candidates("Tagged vector index candidates", index_candidates, index_dir)
    print()
    print_candidates("Tagged query candidates", query_candidates, queries_path)

    summary = retrieve(
        index_dir=index_dir,
        queries_path=queries_path,
        output_path=output_path,
        top_k=args.top_k,
        query_embedding_model=args.query_embedding_model,
        limit=args.limit,
        batch_size=args.batch_size,
        device=args.device,
    )

    print("\nRAG_SEM retrieval summary:")
    print(f"- index_dir: {summary['index_dir']}")
    print(f"- queries_file: {summary['queries_file']}")
    print(f"- queries_loaded: {summary['queries_loaded']}")
    print(f"- queries_retrieved_successfully: {summary['queries_retrieved_successfully']}")
    print(f"- top_k: {summary['top_k']}")
    print(f"- query_embedding_model: {summary['query_embedding_model']}")
    print(f"- embedding_backend: {summary['embedding_backend']}")
    print(f"- output_path: {summary['output_path']}")
    print("- sample_query_top3:")
    print(json.dumps(summary["sample_top3"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
