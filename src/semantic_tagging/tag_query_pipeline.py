from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from semantic_tagging.query_tagger import QueryTagger


DEFAULT_INPUT = Path("data/qa/test_qa.jsonl")
DEFAULT_OUTPUT = Path("data/qa/test_qa_tagged.jsonl")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def row_key(row: dict[str, Any]) -> str:
    key = row.get("id") or row.get("query_id")
    if not key:
        raise ValueError("QA row is missing id/query_id")
    return str(key)


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        existing[row_key(row)] = row
    return existing


def tag_queries(
    input_path: Path,
    output_path: Path,
    *,
    model: str | None,
    resume: bool,
    overwrite: bool,
    dry_run: bool,
    max_chunks: int | None,
) -> None:
    logger = logging.getLogger(__name__)
    rows = load_jsonl(input_path)
    if max_chunks is not None:
        rows = rows[:max_chunks]

    if output_path.exists() and not resume and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite or --resume.")

    existing = load_existing(output_path) if resume else {}
    input_keys = [row_key(row) for row in rows]
    completed = len(set(input_keys) & set(existing))
    missing = len(input_keys) - completed
    print(f"total={len(input_keys)} completed={completed} missing={missing} estimated_llm_calls={missing}")

    tagger = QueryTagger(model=model, dry_run=dry_run, logger=logger)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written: set[str] = set()
    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="query semantic tagging"):
            key = row_key(row)
            if key in written:
                continue
            written.add(key)

            if resume and key in existing and existing[key].get("query_semantic_tags"):
                out = existing[key]
            else:
                out = dict(row)
                question = str(row.get("question") or row.get("query") or "")
                out["query_semantic_tags"] = tagger.tag_query(question)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tag QA queries using paper semantic schema.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--log-file", type=Path, default=Path("query_tagging_errors.log"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file, encoding="utf-8")],
    )
    tag_queries(
        args.input,
        args.output,
        model=args.model,
        resume=args.resume,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()
