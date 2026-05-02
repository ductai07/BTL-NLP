from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CHUNK_DIR = Path("data/chunks")
OUT_DIR = Path("data/benchmark_visual_metadata")

COMPANY_BY_TICKER = {
    "AAPL": "Apple",
    "HD": "Home Depot",
    "INTU": "Intuit",
    "MS": "Morgan Stanley",
    "NVDA": "NVIDIA",
}

VISUAL_LABELS = {
    "bar_chart": "bar chart",
    "line_chart": "line chart",
    "pie_chart": "pie chart",
    "table_like_chart": "table-like visual",
    "diagram": "diagram",
    "logo": "logo",
    "photo": "photo",
    "decorative": "decorative image",
    "unreadable": "unreadable image",
    "other": "visual",
    "other_chart": "chart",
}

GENERIC_EVIDENCE_TERMS = {
    "image is a table-like chart with rows and columns of data",
    "the image is a complex table-like structure with many rows and columns of text",
    "chart context suggests a time series",
    "raw vlm response was not valid json",
    "fallback record because the vlm request failed",
    "line chart with multiple fluctuating lines",
    "chart is a line chart with multiple lines",
    "the image shows a table-like structure with rows and columns of data",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def infer_ticker(source_pdf: str | None) -> str:
    if not source_pdf:
        return ""
    return Path(source_pdf).name.split("_", 1)[0]


def company_name(source_pdf: str | None) -> str:
    ticker = infer_ticker(source_pdf)
    return COMPANY_BY_TICKER.get(ticker, ticker or "the company")


def filing_label(source_pdf: str | None) -> str:
    if not source_pdf:
        return "filing"
    parts = Path(source_pdf).name.split("_")
    if len(parts) >= 2 and parts[1] == "DEF":
        return "proxy statement"
    return parts[1] if len(parts) >= 2 else "filing"


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    out: list[str] = []
    seen: set[str] = set()
    for item in values:
        text = clean_text(item)
        if not text or text.lower() in {"null", "none", "not readable", "unreadable", "not_applicable"}:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def looks_noisy_label(value: str) -> bool:
    text = clean_text(value)
    if not text:
        return True
    if any(ch in text for ch in {"+", "/", "\\", "{", "}", "[", "]", "|"}):
        return True
    if re.fullmatch(r"line\s*\d+", text, flags=re.IGNORECASE):
        return True
    if re.search(r"[A-Za-z]{2,}\d+[A-Za-z0-9]*", text) and not re.search(r"\b(20\d{2}|19\d{2}|Q[1-4])\b", text):
        return True
    if len(text) >= 8 and re.search(r"[A-Z]", text) and re.search(r"[a-z]", text) and re.search(r"\d", text):
        return True
    alpha = re.sub(r"[^A-Za-z]", "", text)
    if len(alpha) >= 8:
        vowels = sum(1 for ch in alpha.lower() if ch in "aeiou")
        if vowels / len(alpha) < 0.18:
            return True
    return False


def meaningful_list(value: Any) -> list[str]:
    return [item for item in as_list(value) if not looks_noisy_label(item)]


def visual_meta(chunk: dict[str, Any]) -> dict[str, Any]:
    meta = chunk.get("visual_metadata")
    if isinstance(meta, dict):
        return meta
    vlm = chunk.get("vlm_output")
    return vlm if isinstance(vlm, dict) else {}


def visual_label(kind: str) -> str:
    return VISUAL_LABELS.get(kind, kind.replace("_", " ") if kind else "visual")


def informative_evidence(evidence: str) -> str:
    evidence = clean_text(evidence)
    lower = evidence.lower().strip(". ")
    if not evidence or lower in GENERIC_EVIDENCE_TERMS:
        return ""
    if len(evidence.split()) < 4:
        return ""
    return evidence


def candidate_score(chunk: dict[str, Any]) -> int:
    meta = visual_meta(chunk)
    kind = clean_text(meta.get("visual_type"))
    if kind in {"", "decorative", "unreadable", "logo", "photo"}:
        return -10

    score = 0
    title = clean_text(meta.get("title"))
    if title:
        score += 5
    for field, weight in [
        ("metrics", 4),
        ("periods", 3),
        ("legend", 3),
        ("key_values", 2),
    ]:
        score += min(len(meaningful_list(meta.get(field))), 4) * weight
    if informative_evidence(clean_text(meta.get("evidence"))):
        score += 3
    if kind in {"line_chart", "bar_chart", "pie_chart"}:
        score += 3
    if kind == "table_like_chart":
        score -= 5
        if not clean_text(meta.get("title")) and not meaningful_list(meta.get("metrics")):
            score -= 8
    return score


def short_list(values: list[str], limit: int = 3) -> str:
    return ", ".join(values[:limit])


def build_question(chunk: dict[str, Any]) -> tuple[str, str] | None:
    meta = visual_meta(chunk)
    kind = clean_text(meta.get("visual_type"))
    if candidate_score(chunk) < 6:
        return None

    company = company_name(chunk.get("source_pdf"))
    filing = filing_label(chunk.get("source_pdf"))
    label = visual_label(kind)
    title = clean_text(meta.get("title"))
    metrics = meaningful_list(meta.get("metrics"))
    periods = meaningful_list(meta.get("periods"))
    legend = meaningful_list(meta.get("legend"))
    values = meaningful_list(meta.get("key_values"))
    trend = clean_text(meta.get("trend"))
    evidence = informative_evidence(clean_text(meta.get("evidence")))

    if title and metrics:
        return (
            f"Which {company} {filing} image shows the {label} titled '{title}' about {short_list(metrics)}?",
            title,
        )
    if title:
        return (
            f"Which {company} {filing} image shows the {label} titled '{title}'?",
            title,
        )
    if metrics and periods:
        return (
            f"Which {company} {filing} image shows a {label} for {short_list(metrics)} over {short_list(periods)}?",
            short_list(metrics),
        )
    if metrics:
        return (
            f"Which {company} {filing} image shows a {label} about {short_list(metrics)}?",
            short_list(metrics),
        )
    if legend and periods:
        return (
            f"Which {company} {filing} image shows a {label} with legend entries {short_list(legend)} across {short_list(periods)}?",
            short_list(legend),
        )
    if kind == "table_like_chart" and not (title or metrics):
        return None
    if values and trend and trend not in {"not_applicable", "unreadable"}:
        return (
            f"Which {company} {filing} image shows a {label} with a {trend} trend and visible labels such as {short_list(values)}?",
            trend,
        )
    if values:
        return (
            f"Which {company} {filing} image shows a {label} with visible labels such as {short_list(values)}?",
            short_list(values),
        )
    return None


def chunk_content(chunk: dict[str, Any]) -> str:
    parts = [
        chunk.get("embed_text"),
        chunk.get("summary"),
        chunk.get("text"),
        json.dumps(chunk.get("visual_metadata"), ensure_ascii=False) if chunk.get("visual_metadata") else None,
        json.dumps(chunk.get("vlm_output"), ensure_ascii=False) if chunk.get("vlm_output") else None,
    ]
    return "\n".join(clean_text(part) for part in parts if clean_text(part))


def corpus_row(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["id"],
        "modality": chunk.get("modality"),
        "source_pdf": chunk.get("source_pdf"),
        "source_html": chunk.get("source_html"),
        "page": chunk.get("page"),
        "text": chunk_content(chunk),
        "image_path": chunk.get("image_path"),
    }


def build_visual_benchmark(chunk_dir: Path, out_dir: Path, target: int, max_per_source: int) -> dict[str, Any]:
    chunks = load_jsonl(chunk_dir / "all_chunks.jsonl")
    image_chunks = [chunk for chunk in chunks if chunk.get("modality") == "image"]

    candidates: list[tuple[int, dict[str, Any], str, str]] = []
    for chunk in image_chunks:
        question_answer = build_question(chunk)
        if not question_answer:
            continue
        question, answer = question_answer
        candidates.append((candidate_score(chunk), chunk, question, answer))

    candidates.sort(key=lambda item: (-item[0], item[1].get("source_pdf") or "", item[1]["id"]))
    by_source: dict[str, int] = defaultdict(int)
    used_questions: set[str] = set()
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []

    for score, chunk, question, answer in candidates:
        if len(queries) >= target:
            break
        source_pdf = chunk.get("source_pdf") or ""
        if by_source[source_pdf] >= max_per_source:
            continue
        qkey = question.casefold()
        if qkey in used_questions:
            continue
        used_questions.add(qkey)
        by_source[source_pdf] += 1
        qid = f"v_image_{len(queries) + 1:03d}"
        meta = visual_meta(chunk)
        queries.append(
            {
                "query_id": qid,
                "question": question,
                "type": "image",
                "answer": answer,
                "answer_type": "text",
                "source_pdf": chunk.get("source_pdf"),
                "evidence_chunk_ids": [chunk["id"]],
                "benchmark_source": "visual_metadata_structured",
                "visual_type": clean_text(meta.get("visual_type")),
            }
        )
        qrels.append(
            {
                "query_id": qid,
                "chunk_id": chunk["id"],
                "relevance": 1,
                "modality": "image",
                "source_pdf": chunk.get("source_pdf"),
                "page": chunk.get("page"),
            }
        )

    full_corpus = [corpus_row(chunk) for chunk in chunks]
    gold_ids = {row["chunk_id"] for row in qrels}
    gold_corpus = [corpus_row(chunk) for chunk in chunks if chunk["id"] in gold_ids]

    write_jsonl(out_dir / "queries.jsonl", queries)
    write_jsonl(out_dir / "qrels.jsonl", qrels)
    write_jsonl(out_dir / "corpus.jsonl", full_corpus)
    write_jsonl(out_dir / "gold_corpus.jsonl", gold_corpus)

    summary = {
        "queries": len(queries),
        "qrels": len(qrels),
        "full_corpus_chunks": len(full_corpus),
        "gold_corpus_chunks": len(gold_corpus),
        "candidate_images": len(candidates),
        "all_images": len(image_chunks),
        "query_type_counts": dict(Counter(row["type"] for row in queries)),
        "qrel_modality_counts": dict(Counter(row["modality"] for row in qrels)),
        "visual_type_counts": dict(Counter(row.get("visual_type") for row in queries)),
        "note": "Image-only retrieval benchmark generated from structured visual_metadata fields.",
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an image retrieval benchmark from visual_metadata.")
    parser.add_argument("--chunk-dir", type=Path, default=CHUNK_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--target", type=int, default=60)
    parser.add_argument("--max-per-source", type=int, default=8)
    args = parser.parse_args()

    summary = build_visual_benchmark(args.chunk_dir, args.out_dir, args.target, args.max_per_source)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
