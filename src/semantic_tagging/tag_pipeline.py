from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Iterable

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from semantic_tagging.semantic_tagger import SemanticTagger


DEFAULT_INPUT = Path("data/chunks.jsonl")
DEFAULT_OUTPUT = Path("data/chunks_tagged_paper_schema.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_key(chunk: dict[str, Any]) -> str:
    key = chunk.get("chunk_id") or chunk.get("id")
    if not key:
        raise ValueError("chunk is missing chunk_id/id")
    return str(key)


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        existing[chunk_key(row)] = row
    return existing


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tag_file(
    input_path: Path,
    output_path: Path,
    *,
    model: str | None,
    resume: bool,
    dry_run: bool,
    overwrite: bool,
    max_chunks: int | None,
) -> None:
    logger = logging.getLogger(__name__)
    rows = load_jsonl(input_path)
    if max_chunks is not None:
        rows = rows[:max_chunks]

    if output_path.exists() and not resume and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to run from scratch or --resume to continue.")

    existing = load_existing(output_path) if resume else {}
    input_keys = [chunk_key(row) for row in rows]
    completed = len(set(input_keys) & set(existing))
    missing = len(input_keys) - completed
    print(f"total={len(input_keys)} completed={completed} missing={missing} estimated_llm_calls={missing}")

    tagger = SemanticTagger(model=model, dry_run=dry_run, logger=logger)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written: set[str] = set()
    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="semantic tagging"):
            key = chunk_key(row)
            if key in written:
                continue
            written.add(key)

            if resume and key in existing and existing[key].get("semantic_tags"):
                out = existing[key]
            else:
                out = dict(row)
                out["semantic_tags"] = tagger.tag_chunk(row)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic tag financial chunks using the paper schema only.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the LLM; write empty paper-schema tags.")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-file", type=Path, default=Path("tagging_errors.log"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file, encoding="utf-8")],
    )
    tag_file(
        args.input,
        args.output,
        model=args.model,
        resume=args.resume,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()
