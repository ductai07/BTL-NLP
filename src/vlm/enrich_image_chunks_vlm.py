from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from vlm.nvidia_vlm_client import IMAGE_PROMPT, call_gemma_vision


CHUNK_DIR = Path("data/chunks")


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_jsonish(text: str) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.removeprefix("json").strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return {"summary": stripped, "key_values": [], "evidence": "Raw VLM response was not valid JSON."}


def summarize_with_retries(image_path: Path, max_image_side: int, retries: int, timeout: int) -> str:
    last_error: Exception | None = None
    sides = []
    current_side = max_image_side
    while current_side >= 384:
        sides.append(current_side)
        current_side = int(current_side * 0.75)

    for side in sides:
        for attempt in range(1, retries + 1):
            try:
                return call_gemma_vision(
                    IMAGE_PROMPT,
                    image_path,
                    stream=False,
                    max_tokens=800,
                    max_image_side=side,
                    timeout=timeout,
                )
            except (requests.RequestException, RuntimeError) as exc:
                last_error = exc
                message = str(exc)
                if "maximum context length" in message:
                    print(f"  image side {side}px too large for context; trying smaller image")
                    break
                wait = min(30, 2**attempt)
                print(f"  attempt {attempt}/{retries} at {side}px failed: {exc}. retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"VLM failed after {retries} attempts for {image_path}: {last_error}")


def fallback_output(row: dict, error: Exception) -> dict:
    image_name = Path(row.get("image_path") or row.get("crop_path") or "").name
    return {
        "visual_type": "unreadable",
        "title": None,
        "summary": (
            "VLM summarization failed for this image. "
            f"The image asset is {image_name or 'unknown'} from an SEC filing."
        ),
        "x_axis": None,
        "y_axis": None,
        "legend": [],
        "metrics": [],
        "periods": [],
        "key_values": [],
        "trend": "unreadable",
        "evidence": "Fallback record because the VLM request failed.",
        "error": str(error),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize cropped image chunks with NVIDIA Gemma Vision.")
    parser.add_argument("--input", type=Path, default=CHUNK_DIR / "image_chunks_pdfminer.jsonl")
    parser.add_argument("--output", type=Path, default=CHUNK_DIR / "image_chunks_vlm.jsonl")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-width", type=int, default=200)
    parser.add_argument("--min-height", type=int, default=120)
    parser.add_argument("--min-area", type=int, default=100_000)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--max-image-side", type=int, default=1024)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--replace-current", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("NVIDIA_API_KEY"):
        raise RuntimeError("NVIDIA_API_KEY is not set. Set it before running VLM summarization.")

    rows = load_jsonl(args.input)
    rows = [
        row
        for row in rows
        if int(row.get("image_width") or 0) >= args.min_width
        and int(row.get("image_height") or 0) >= args.min_height
        and int(row.get("image_area") or 0) >= args.min_area
    ]
    if args.limit is not None:
        rows = rows[: args.limit]

    existing = []
    done_ids = set()
    cached_by_path = {}
    if args.output.exists() and not args.no_resume:
        existing = load_jsonl(args.output)
        done_ids = {row.get("id") for row in existing}
        cached_by_path = {
            row.get("image_path") or row.get("crop_path"): row
            for row in existing
            if row.get("image_path") or row.get("crop_path")
        }
        rows = [row for row in rows if row.get("id") not in done_ids]
        print(f"Resume enabled: {len(done_ids)} existing rows, {len(rows)} remaining rows")
    elif args.output.exists():
        args.output.unlink()

    enriched = []
    for idx, row in enumerate(rows, start=1):
        image_path = Path(row.get("crop_path") or row.get("image_path") or "")
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image asset for chunk {row.get('id')}: {image_path}")

        print(f"[{idx}/{len(rows)}] VLM image summary: {image_path}")
        cache_key = str(image_path).replace("\\", "/")
        cached = cached_by_path.get(cache_key) or cached_by_path.get(str(image_path))
        if cached:
            parsed = cached.get("vlm_output") or {}
            summary = cached.get("summary") or parsed.get("summary") or ""
        else:
            raw = ""
            try:
                raw = summarize_with_retries(
                    image_path=image_path,
                    max_image_side=args.max_image_side,
                    retries=args.retries,
                    timeout=args.timeout,
                )
                parsed = parse_jsonish(raw)
            except Exception as exc:
                print(f"  VLM failed permanently, using image fallback: {exc}")
                parsed = fallback_output(row, exc)
            summary = parsed.get("summary") or raw.strip()
            cached_by_path[cache_key] = {"summary": summary, "vlm_output": parsed}
        row = dict(row)
        row["summary"] = summary
        row["text"] = summary
        row["vlm_model"] = "google/gemma-3-27b-it"
        row["vlm_output"] = parsed
        row["chunking_method"] = "sec_html_image_asset_gemma3_vision_summary"
        enriched.append(row)
        append_jsonl(args.output, row)
        time.sleep(args.sleep)

    if args.replace_current:
        final_rows = load_jsonl(args.output)
        write_jsonl(CHUNK_DIR / "image_chunks.jsonl", final_rows)
        text_chunks = load_jsonl(CHUNK_DIR / "text_chunks.jsonl")
        table_chunks = load_jsonl(CHUNK_DIR / "table_chunks.jsonl")
        write_jsonl(CHUNK_DIR / "all_chunks.jsonl", text_chunks + table_chunks + final_rows)

    print(
        json.dumps(
            {
                "input": str(args.input),
                "output": str(args.output),
                "new_rows": len(enriched),
                "total_rows": len(load_jsonl(args.output)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
