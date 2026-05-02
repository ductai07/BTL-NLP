from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Iterable

from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from graph_extraction.llm_openie_model import LLMOPENIEModel


DEFAULT_INPUT = Path("data/chunks/all_chunks.jsonl")
DEFAULT_OUTPUT = Path("data/graph/chunk_graph.jsonl")
DEFAULT_GRAPH_SUMMARY = Path("data/graph/graph_summary.json")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_key(chunk: dict[str, Any]) -> str:
    key = chunk.get("chunk_id") or chunk.get("id")
    if key is None or not str(key).strip():
        raise ValueError("chunk is missing chunk_id/id")
    return str(key)


def chunk_text(chunk: dict[str, Any]) -> str:
    for field in ("text", "summary", "embed_text"):
        value = chunk.get(field)
        if value and str(value).strip():
            return str(value)
    return ""


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    existing: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        try:
            existing[chunk_key(row)] = row
        except ValueError:
            continue
    return existing


def extract_graph_for_chunks(
    *,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    model: str | None,
    resume: bool,
    overwrite: bool,
    max_chunks: int | None,
    enable_delta_heuristic: bool,
) -> dict[str, Any]:
    logger = logging.getLogger(__name__)
    rows = load_jsonl(input_path)
    if max_chunks is not None:
        rows = rows[:max_chunks]

    if output_path.exists() and not resume and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to run from scratch or --resume to continue.")

    existing = load_existing(output_path) if resume else {}
    model_runner = LLMOPENIEModel(model=model, enable_delta_heuristic=enable_delta_heuristic)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    written: set[str] = set()
    all_entities: set[str] = set()
    all_triples: set[tuple[str, str, str]] = set()
    success = 0
    failed = 0

    with output_path.open("w", encoding="utf-8") as f:
        for row in tqdm(rows, desc="graph extraction"):
            try:
                key = chunk_key(row)
            except ValueError:
                failed += 1
                continue
            if key in written:
                continue
            written.add(key)

            if resume and key in existing and isinstance(existing[key].get("graph"), dict):
                out = existing[key]
            else:
                text = chunk_text(row)
                if not text.strip():
                    failed += 1
                    continue
                graph = model_runner(text)
                out = dict(row)
                out["graph"] = {
                    "entities": graph.get("extracted_entities", []),
                    "triples": graph.get("extracted_triples", []),
                    "clean_triples": graph.get("clean_triples", []),
                    "noisy_triples": graph.get("noisy_triples", []),
                }
                success += 1

            graph_out = out.get("graph") or {}
            for entity in graph_out.get("entities") or []:
                all_entities.add(str(entity))
            for triple in graph_out.get("triples") or []:
                if isinstance(triple, list) and len(triple) == 3:
                    all_triples.add((str(triple[0]), str(triple[1]), str(triple[2])))

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()

    summary = {
        "input_path": str(input_path).replace("\\", "/"),
        "output_path": str(output_path).replace("\\", "/"),
        "model": model or "OPENAI_MODEL from env or default in client",
        "chunks_total": len(rows),
        "chunks_processed": len(written),
        "chunks_success": success,
        "chunks_failed": failed,
        "unique_entities": len(all_entities),
        "unique_triples": len(all_triples),
        "resume": resume,
        "enable_delta_heuristic": enable_delta_heuristic,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Graph extraction done: %s", json.dumps(summary, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract chunk-level knowledge graph using LLM from env API settings.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_GRAPH_SUMMARY)
    parser.add_argument("--model", default=None, help="Override model name. If empty, use OPENAI_MODEL from env.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument(
        "--enable-delta-heuristic",
        action="store_true",
        help="Bat fallback heuristic cho cau co tu khoa tang/giam khi output LLM qua it.",
    )
    parser.add_argument("--log-file", type=Path, default=Path("graph_extraction_errors.log"))
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log_file, encoding="utf-8")],
    )

    summary = extract_graph_for_chunks(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary_out,
        model=args.model,
        resume=args.resume,
        overwrite=args.overwrite,
        max_chunks=args.max_chunks,
        enable_delta_heuristic=args.enable_delta_heuristic,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
