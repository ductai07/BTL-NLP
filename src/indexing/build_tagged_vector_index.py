from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

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
TAG_CONTAINER_CANDIDATES = ("semantic_tags", "tags")
OLD_FORBIDDEN_FIELDS = {
    "chunk_role",
    "evidence_type",
    "section_tags",
    "financial_metrics",
    "business_topics",
    "risk_topics",
    "retrieval_keywords",
}


def require_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing faiss-cpu. Install it with: python -m pip install faiss-cpu") from exc
    return faiss


def load_jsonl_safe(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append({"line": line_no, "error": f"json_parse_error: {exc}"})
                continue
            if not isinstance(value, dict):
                errors.append({"line": line_no, "error": "record_is_not_object"})
                continue
            rows.append((line_no, value))
    return rows, errors


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_id(row: dict[str, Any]) -> str | None:
    value = row.get("chunk_id") or row.get("id")
    return str(value) if value is not None and str(value).strip() else None


def chunk_text(row: dict[str, Any]) -> str:
    if row.get("embed_text"):
        return str(row["embed_text"])
    if row.get("modality") in {"table", "image"}:
        return str(row.get("summary") or row.get("text") or "")
    return str(row.get("text") or row.get("summary") or "")


def find_tag_container(row: dict[str, Any]) -> tuple[str | None, Any]:
    for field in TAG_CONTAINER_CANDIDATES:
        if field in row:
            return field, row.get(field)
    return None, None


def normalize_tags(tags: Any) -> tuple[dict[str, list[str]], list[str]]:
    """Return exactly the 11 paper-inspired tag fields without inferring new tags."""
    normalized: dict[str, list[str]] = {field: [] for field in SCHEMA_FIELDS}
    warnings: list[str] = []
    if not isinstance(tags, dict):
        warnings.append(f"tag_container_not_object:{type(tags).__name__}")
        return normalized, warnings

    extra_fields = sorted(set(tags) - SCHEMA_FIELD_SET)
    if extra_fields:
        warnings.append(f"extra_tag_fields:{','.join(extra_fields)}")
    forbidden = sorted(set(tags) & OLD_FORBIDDEN_FIELDS)
    if forbidden:
        warnings.append(f"forbidden_old_tag_fields:{','.join(forbidden)}")

    for field in SCHEMA_FIELDS:
        value = tags.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            values = value
        elif isinstance(value, str):
            warnings.append(f"field_not_list:{field}:str")
            values = [value]
        else:
            warnings.append(f"field_not_list:{field}:{type(value).__name__}")
            continue

        cleaned: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(values):
            if not isinstance(item, str):
                warnings.append(f"non_string_item:{field}:{index}:{type(item).__name__}")
                continue
            text = " ".join(item.split()).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                warnings.append(f"duplicate_tag:{field}:{text}")
                continue
            seen.add(key)
            cleaned.append(text)
        normalized[field] = cleaned
    return normalized, warnings


def build_metadata(row: dict[str, Any], tags: dict[str, list[str]]) -> dict[str, Any]:
    excluded = {"text", "embed_text", "summary", "semantic_tags", "tags"}
    metadata = {key: value for key, value in row.items() if key not in excluded}
    metadata["semantic_tags"] = tags
    return metadata


def candidate_score(path: Path, sample_limit: int = 50) -> dict[str, Any]:
    score = 0
    reasons: list[str] = []
    rows, parse_errors = load_jsonl_safe(path)
    sample = rows[:sample_limit]
    tag_fields = Counter()
    exact_schema = 0
    old_schema = 0
    text_like = 0
    ids = 0
    for _, row in sample:
        cid = chunk_id(row)
        if cid:
            ids += 1
            score += 2
        if chunk_text(row):
            text_like += 1
            score += 3
        tag_field, tags = find_tag_container(row)
        if tag_field:
            tag_fields[tag_field] += 1
            score += 8
        if isinstance(tags, dict):
            keys = set(tags)
            if keys == SCHEMA_FIELD_SET:
                exact_schema += 1
                score += 12
            if keys & OLD_FORBIDDEN_FIELDS:
                old_schema += 1
                score -= 20
    if tag_fields:
        reasons.append(f"tag containers found: {dict(tag_fields)}")
    if exact_schema:
        reasons.append(f"{exact_schema}/{len(sample)} sampled records match the 11-field tag schema exactly")
    if old_schema:
        reasons.append(f"{old_schema}/{len(sample)} sampled records include old/forbidden tag fields")
    if text_like:
        reasons.append(f"{text_like}/{len(sample)} sampled records have chunk text/embed text")
    if ids:
        reasons.append(f"{ids}/{len(sample)} sampled records have id/chunk_id")
    if parse_errors:
        reasons.append(f"{len(parse_errors)} parse errors while scanning")
        score -= len(parse_errors)
    return {
        "path": str(path),
        "score": score,
        "sampled_records": len(sample),
        "tag_fields": dict(tag_fields),
        "exact_schema_sample_count": exact_schema,
        "old_schema_sample_count": old_schema,
        "text_like_sample_count": text_like,
        "id_sample_count": ids,
        "reasons": reasons,
    }


def discover_tagged_chunk_candidates(root: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for path in root.rglob("*.jsonl"):
        lowered = str(path).lower()
        if any(skip in lowered for skip in ("audit_schema_issues", "suspicious_organizations", "sample_tags")):
            continue
        score = candidate_score(path)
        if (
            score["score"] > 0
            and score["text_like_sample_count"] > 0
            and score["id_sample_count"] > 0
            and any(score["tag_fields"].get(field, 0) for field in TAG_CONTAINER_CANDIDATES)
        ):
            candidates.append(score)
    candidates.sort(key=lambda item: (item["exact_schema_sample_count"], item["score"]), reverse=True)
    return candidates


def choose_chunks_path(explicit: Path | None, candidates: list[dict[str, Any]]) -> Path:
    if explicit is not None:
        return explicit
    if not candidates:
        raise FileNotFoundError("No tagged chunk JSONL candidate found. Pass --chunks explicitly.")
    return Path(candidates[0]["path"])


def print_candidates(candidates: list[dict[str, Any]], selected: Path) -> None:
    print("Tagged chunk candidates:")
    for item in candidates[:10]:
        is_selected = Path(item["path"]).resolve() == selected.resolve()
        suffix = " SELECTED" if is_selected else ""
        print(f"- {item['path']}{suffix}")
        print(f"  score={item['score']} sampled={item['sampled_records']}")
        for reason in item["reasons"][:5]:
            print(f"  reason: {reason}")


def prepare_records(path: Path, limit: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows, load_errors = load_jsonl_safe(path)
    if limit is not None:
        rows = rows[:limit]

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = list(load_errors)
    warning_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for line_no, row in rows:
        cid = chunk_id(row)
        if not cid:
            errors.append({"line": line_no, "error": "missing_chunk_id"})
            continue
        if cid in seen_ids:
            errors.append({"line": line_no, "chunk_id": cid, "error": "duplicate_chunk_id"})
            continue
        seen_ids.add(cid)

        text = chunk_text(row)
        if not text.strip():
            errors.append({"line": line_no, "chunk_id": cid, "error": "empty_chunk_text"})
            continue

        tag_field, raw_tags = find_tag_container(row)
        if tag_field is None:
            errors.append({"line": line_no, "chunk_id": cid, "error": "missing_semantic_tags_or_tags"})
            continue
        tags, warnings = normalize_tags(raw_tags)
        for warning in warnings:
            warning_counts[warning.split(":", 1)[0]] += 1

        records.append(
            {
                "chunk_id": cid,
                "text": text,
                "metadata": build_metadata(row, tags),
                "semantic_tags": tags,
                "_source": {"line": line_no, "tag_field": tag_field},
            }
        )

    stats = {
        "loaded_records": len(rows) + len(load_errors),
        "parsed_records": len(rows),
        "prepared_records": len(records),
        "error_count": len(errors),
        "normalization_warning_counts": dict(warning_counts),
    }
    return records, errors, stats


def build_index(
    *,
    chunks_path: Path,
    index_out: Path,
    embedding_model: str,
    batch_size: int,
    device: str | None,
    rebuild: bool,
    limit: int | None,
    source_index_dir: Path | None,
    force_reembed: bool,
) -> dict[str, Any]:
    faiss = require_faiss()
    if index_out.exists():
        if not rebuild:
            raise FileExistsError(f"{index_out} already exists. Use --rebuild to overwrite it.")
        shutil.rmtree(index_out)
    index_out.mkdir(parents=True, exist_ok=True)

    records, errors, prep_stats = prepare_records(chunks_path, limit)
    texts = [record["text"] for record in records]
    ids = [record["chunk_id"] for record in records]

    if not records:
        raise RuntimeError("No valid tagged chunks to index.")

    index_path = index_out / "chunks.faiss"
    ids_path = index_out / "chunk_ids.json"
    records_path = index_out / "records.jsonl"
    embeddings_path = index_out / "embeddings.npy"
    errors_path = index_out / "index_errors.jsonl"
    meta_path = index_out / "meta.json"

    embedding_source = "newly_encoded"
    source_index_meta: dict[str, Any] = {}
    if source_index_dir is not None and not force_reembed:
        source_index_path = source_index_dir / "all.faiss"
        source_ids_path = source_index_dir / "all_chunk_ids.json"
        source_meta_path = source_index_dir / "all_meta.json"
        if source_index_path.exists() and source_ids_path.exists():
            source_ids = json.loads(source_ids_path.read_text(encoding="utf-8"))
            if source_ids[: len(ids)] == ids and (limit is not None or len(source_ids) == len(ids)):
                source_index = faiss.read_index(str(source_index_path))
                if source_index.ntotal < len(ids):
                    raise RuntimeError(
                        f"Existing source index has {source_index.ntotal} vectors but {len(ids)} records are needed."
                    )
                if limit is None and source_index.ntotal == len(ids):
                    faiss.write_index(source_index, str(index_path))
                    embeddings = source_index.reconstruct_n(0, source_index.ntotal)
                else:
                    embeddings = source_index.reconstruct_n(0, len(ids)).astype("float32")
                    index = faiss.IndexFlatIP(embeddings.shape[1])
                    index.add(embeddings)
                    faiss.write_index(index, str(index_path))
                np.save(embeddings_path, embeddings)
                embedding_source = "existing_faiss_index"
                if source_meta_path.exists():
                    source_index_meta = json.loads(source_meta_path.read_text(encoding="utf-8"))
            else:
                print("Existing FAISS index IDs do not match tagged chunk order; encoding records instead.")

    if embedding_source == "newly_encoded":
        embedder = load_bge_model(model_name=embedding_model, batch_size=batch_size, device=device)
        embeddings = embedder.encode_documents(texts)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))
        np.save(embeddings_path, embeddings)

    ids_path.write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(records_path, records)
    write_jsonl(errors_path, errors)

    sample_record = records[0]
    meta = {
        "backend": "faiss",
        "index_type": "IndexFlatIP",
        "embedding_model": embedding_model,
        "embedding_source": embedding_source,
        "source_index_dir": str(source_index_dir).replace("\\", "/") if source_index_dir else None,
        "source_index_meta": source_index_meta,
        "tag_schema_fields": SCHEMA_FIELDS,
        "source_chunks_file": str(chunks_path).replace("\\", "/"),
        "limit": limit,
        "chunks_loaded": prep_stats["loaded_records"],
        "chunks_parsed": prep_stats["parsed_records"],
        "chunks_indexed": len(records),
        "chunks_error_count": len(errors),
        "normalization_warning_counts": prep_stats["normalization_warning_counts"],
        "dimension": int(embeddings.shape[1]),
        "normalized_embeddings": True,
        "metric": "inner_product_cosine",
        "index_path": str(index_path).replace("\\", "/"),
        "chunk_ids_path": str(ids_path).replace("\\", "/"),
        "records_path": str(records_path).replace("\\", "/"),
        "embeddings_path": str(embeddings_path).replace("\\", "/"),
        "errors_path": str(errors_path).replace("\\", "/"),
        "sample_record_metadata": sample_record["metadata"],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Build a FAISS vector index from tagged chunks, preserving 11-field semantic_tags metadata."
    )
    parser.add_argument("--chunks", type=Path, help="Tagged chunks JSONL. Auto-detected when omitted.")
    parser.add_argument("--index-out", type=Path, default=Path("data/index_tagged_chunks"))
    parser.add_argument("--embedding-model", default=DEFAULT_BGE_MODEL)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--limit", type=int, help="Optional record limit for quick tests.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="Example: cuda, cpu. Leave empty for sentence-transformers auto device.")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--source-index-dir",
        type=Path,
        default=Path("data/index_bge"),
        help="Existing FAISS index dir to reuse when chunk IDs match. Use --force-reembed to ignore it.",
    )
    parser.add_argument("--force-reembed", action="store_true", help="Encode chunks again instead of reusing FAISS vectors.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    candidates = discover_tagged_chunk_candidates(project_root)
    chunks_path = choose_chunks_path(args.chunks, candidates)
    if not chunks_path.is_absolute():
        chunks_path = (project_root / chunks_path).resolve()

    print_candidates(candidates, chunks_path)
    print(f"\nSelected tagged chunks file: {chunks_path}")

    index_out = args.index_out
    if not index_out.is_absolute():
        index_out = (project_root / index_out).resolve()
    source_index_dir = args.source_index_dir
    if source_index_dir is not None and not source_index_dir.is_absolute():
        source_index_dir = (project_root / source_index_dir).resolve()

    meta = build_index(
        chunks_path=chunks_path,
        index_out=index_out,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device,
        rebuild=args.rebuild,
        limit=args.limit,
        source_index_dir=source_index_dir,
        force_reembed=args.force_reembed,
    )

    print("\nTagged vector index summary:")
    print(f"- tagged_chunks_file: {meta['source_chunks_file']}")
    print(f"- chunks_loaded: {meta['chunks_loaded']}")
    print(f"- chunks_indexed: {meta['chunks_indexed']}")
    print(f"- chunks_error_count: {meta['chunks_error_count']}")
    print(f"- embedding_backend: {meta['backend']}")
    print(f"- embedding_model: {meta['embedding_model']}")
    print(f"- embedding_source: {meta['embedding_source']}")
    print(f"- vector_index_output_path: {str(index_out).replace(chr(92), '/')}")
    print("- sample_record_metadata:")
    print(json.dumps(meta["sample_record_metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
