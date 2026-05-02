from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CHUNKS_PATH = Path("data/chunks/all_chunks.jsonl")
OUT_DIR = Path("data/benchmark_hard")

COMPANY_BY_TICKER = {
    "AAPL": "Apple",
    "HD": "Home Depot",
    "INTU": "Intuit",
    "MS": "Morgan Stanley",
    "NVDA": "NVIDIA",
}

STOPWORDS = {
    "about",
    "above",
    "after",
    "also",
    "among",
    "and",
    "annual",
    "are",
    "because",
    "been",
    "before",
    "between",
    "business",
    "can",
    "company",
    "could",
    "during",
    "each",
    "ended",
    "financial",
    "filing",
    "first",
    "for",
    "from",
    "has",
    "have",
    "inc",
    "including",
    "into",
    "its",
    "may",
    "million",
    "more",
    "not",
    "other",
    "our",
    "page",
    "period",
    "quarter",
    "report",
    "reported",
    "results",
    "sec",
    "section",
    "such",
    "table",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "under",
    "was",
    "were",
    "which",
    "with",
    "year",
}


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


def clean_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def chunk_text(chunk: dict[str, Any]) -> str:
    parts = [
        chunk.get("text"),
        chunk.get("summary"),
        chunk.get("table_markdown"),
        json.dumps(chunk.get("table_json"), ensure_ascii=False) if chunk.get("table_json") else None,
        chunk.get("image_path"),
    ]
    return "\n".join(clean_text(part) for part in parts if part)


def infer_ticker(source_pdf: str | None) -> str:
    if not source_pdf:
        return ""
    return Path(source_pdf).name.split("_", 1)[0]


def company_name(chunk: dict[str, Any]) -> str:
    ticker = infer_ticker(chunk.get("source_pdf"))
    return COMPANY_BY_TICKER.get(ticker, ticker or "the company")


def form_name(chunk: dict[str, Any]) -> str:
    source = Path(str(chunk.get("source_pdf") or "")).name
    parts = source.split("_")
    if len(parts) >= 2 and parts[1] == "DEF":
        return "proxy statement"
    if len(parts) >= 2:
        return parts[1]
    return "filing"


def page_number(chunk: dict[str, Any]) -> int | None:
    value = chunk.get("page_start", chunk.get("page"))
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def salient_terms(text: str, limit: int = 5) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z&'-]{3,}", text)
    counts: Counter[str] = Counter()
    original: dict[str, str] = {}
    for word in words:
        key = word.lower().strip("-'")
        if key in STOPWORDS or len(key) < 4:
            continue
        counts[key] += 1
        original.setdefault(key, word.strip("-'"))
    terms = []
    for key, _ in counts.most_common(20):
        term = original[key]
        if term.lower() not in {t.lower() for t in terms}:
            terms.append(term)
        if len(terms) >= limit:
            break
    return terms


def topic_from_text(text: str, fallback: str = "the relevant disclosure") -> str:
    terms = salient_terms(text, limit=4)
    if len(terms) >= 2:
        return ", ".join(terms[:4])
    return fallback


def table_topic(summary: str) -> str:
    summary = clean_text(summary)
    headers = re.search(r"Headers:\s*([^\.]+)", summary)
    if headers:
        terms = [part.strip() for part in headers.group(1).split("|") if part.strip()]
        terms = [term for term in terms if not re.fullmatch(r"[$()%,.\d\s-]+", term)]
        if terms:
            return ", ".join(terms[:4])
    return topic_from_text(summary, "the table values")


def visual_kind(summary: str) -> str | None:
    text = summary.lower()
    if "bar chart" in text:
        return "bar chart"
    if "line chart" in text or "line graph" in text or "multiple lines" in text:
        return "line chart"
    if "pie chart" in text:
        return "pie chart"
    if "table" in text:
        return "table-like chart"
    if "chart" in text:
        return "chart"
    return None


def is_boilerplate_text(chunk: dict[str, Any]) -> bool:
    text = chunk_text(chunk).lower()
    page = page_number(chunk)
    if page is not None and page <= 3:
        return True
    bad_markers = [
        "united states securities and exchange commission",
        "table of contents",
        "commission file number",
        "exact name of registrant",
        "signature pursuant to the requirements",
        "common stock",
    ]
    return any(marker in text for marker in bad_markers)


def useful_text_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for chunk in chunks:
        if chunk.get("modality") != "text" or is_boilerplate_text(chunk):
            continue
        text = chunk_text(chunk)
        if len(text.split()) < 80:
            continue
        if len(salient_terms(text, limit=3)) < 3:
            continue
        rows.append(chunk)
    return rows


def useful_table_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for chunk in chunks:
        if chunk.get("modality") != "table":
            continue
        summary = clean_text(chunk.get("summary"))
        if len(summary.split()) < 18:
            continue
        page = page_number(chunk)
        if page is not None and page <= 2:
            continue
        if any(term in summary.lower() for term in ("commission file", "telephone number", "state or other jurisdiction")):
            continue
        rows.append(chunk)
    return rows


def useful_image_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for chunk in chunks:
        if chunk.get("modality") != "image":
            continue
        summary = clean_text(chunk.get("summary"))
        if len(summary.split()) < 18 or not visual_kind(summary):
            continue
        rows.append(chunk)
    return rows


def round_robin_by_source(chunks: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order = []
    for chunk in chunks:
        source = chunk.get("source_pdf") or ""
        if source not in grouped:
            order.append(source)
        grouped[source].append(chunk)

    selected = []
    while len(selected) < limit:
        added = False
        for source in order:
            if grouped[source]:
                selected.append(grouped[source].pop(0))
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    return selected


def corpus_row(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": chunk["id"],
        "modality": chunk.get("modality"),
        "source_pdf": chunk.get("source_pdf"),
        "source_html": chunk.get("source_html"),
        "page": chunk.get("page"),
        "page_start": chunk.get("page_start", chunk.get("page")),
        "page_end": chunk.get("page_end", chunk.get("page")),
        "text": chunk_text(chunk),
        "image_path": chunk.get("image_path"),
    }


def add_query(
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    query_id: str,
    qtype: str,
    question: str,
    chunks: list[dict[str, Any]],
    answer: str = "",
) -> None:
    queries.append(
        {
            "query_id": query_id,
            "question": clean_text(question),
            "type": qtype,
            "answer": answer,
            "answer_type": "text",
            "source_pdf": chunks[0].get("source_pdf") if chunks else None,
            "evidence_chunk_ids": [chunk["id"] for chunk in chunks],
            "benchmark_source": "hard_chunk_level",
        }
    )
    for chunk in chunks:
        qrels.append(
            {
                "query_id": query_id,
                "chunk_id": chunk["id"],
                "relevance": 1,
                "modality": chunk.get("modality"),
                "source_pdf": chunk.get("source_pdf"),
                "page": chunk.get("page"),
                "page_start": chunk.get("page_start", chunk.get("page")),
                "page_end": chunk.get("page_end", chunk.get("page")),
            }
        )


def build_hard_benchmark(chunks_path: Path, out_dir: Path, text_n: int, table_n: int, image_n: int, multimodal_n: int) -> dict[str, Any]:
    chunks = load_jsonl(chunks_path)
    text_chunks = round_robin_by_source(useful_text_chunks(chunks), text_n)
    table_chunks = round_robin_by_source(useful_table_chunks(chunks), table_n)
    image_chunks = round_robin_by_source(useful_image_chunks(chunks), image_n)

    by_source_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_table: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in useful_text_chunks(chunks):
        by_source_text[chunk.get("source_pdf")].append(chunk)
    for chunk in useful_table_chunks(chunks):
        by_source_table[chunk.get("source_pdf")].append(chunk)
    for chunk in useful_image_chunks(chunks):
        by_source_image[chunk.get("source_pdf")].append(chunk)

    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []

    for i, chunk in enumerate(text_chunks, start=1):
        topic = topic_from_text(chunk_text(chunk))
        question = (
            f"In {company_name(chunk)}'s {form_name(chunk)} filing, retrieve the passage that explains "
            f"the disclosure involving {topic}."
        )
        add_query(queries, qrels, f"hard_text_{i:03d}", "text", question, [chunk])

    for i, chunk in enumerate(table_chunks, start=1):
        topic = table_topic(clean_text(chunk.get("summary")))
        question = (
            f"Find the financial table in {company_name(chunk)}'s {form_name(chunk)} filing that should be used "
            f"to verify figures related to {topic}."
        )
        add_query(queries, qrels, f"hard_table_{i:03d}", "table", question, [chunk])

    for i, chunk in enumerate(image_chunks, start=1):
        kind = visual_kind(clean_text(chunk.get("summary"))) or "chart"
        topic = topic_from_text(clean_text(chunk.get("summary")), kind)
        question = (
            f"Find the {kind} image in {company_name(chunk)}'s filing that visually compares or trends "
            f"{topic}."
        )
        add_query(queries, qrels, f"hard_image_{i:03d}", "image", question, [chunk], answer=kind)

    multimodal_pairs: list[list[dict[str, Any]]] = []
    trimodal_pairs: list[list[dict[str, Any]]] = []
    for source, tables in by_source_table.items():
        texts = by_source_text.get(source, [])
        images = by_source_image.get(source, [])
        if not texts or not images:
            continue
        for table, image in zip(tables[:3], images[:3]):
            table_page = page_number(table)
            ranked_texts = sorted(
                texts,
                key=lambda t: abs((page_number(t) or 10**6) - (table_page or 10**6)),
            )
            if ranked_texts:
                trimodal_pairs.append([ranked_texts[0], table, image])

    for source, tables in by_source_table.items():
        texts = by_source_text.get(source, [])
        if not texts:
            continue
        for table in tables[:3]:
            table_page = page_number(table)
            ranked_texts = sorted(
                texts,
                key=lambda t: abs((page_number(t) or 10**6) - (table_page or 10**6)),
            )
            if ranked_texts:
                multimodal_pairs.append([ranked_texts[0], table])
    for source, images in by_source_image.items():
        texts = by_source_text.get(source, [])
        if not texts:
            continue
        for image in images[:2]:
            multimodal_pairs.append([texts[0], image])

    selected_multimodal = round_robin_pairs(trimodal_pairs, multimodal_n)
    if len(selected_multimodal) < multimodal_n:
        selected_ids = {tuple(chunk["id"] for chunk in pair) for pair in selected_multimodal}
        for pair in round_robin_pairs(multimodal_pairs, multimodal_n * 2):
            key = tuple(chunk["id"] for chunk in pair)
            if key in selected_ids:
                continue
            selected_multimodal.append(pair)
            if len(selected_multimodal) >= multimodal_n:
                break

    for i, pair in enumerate(selected_multimodal, start=1):
        text_chunk = next((chunk for chunk in pair if chunk.get("modality") == "text"), pair[0])
        table_chunk = next((chunk for chunk in pair if chunk.get("modality") == "table"), None)
        image_chunk = next((chunk for chunk in pair if chunk.get("modality") == "image"), None)
        text_topic = topic_from_text(chunk_text(text_chunk))
        if table_chunk and image_chunk:
            table_topic_text = table_topic(clean_text(table_chunk.get("summary")))
            kind = visual_kind(clean_text(image_chunk.get("summary"))) or "chart"
            question = (
                f"Retrieve all evidence needed from {company_name(text_chunk)}'s {form_name(text_chunk)} filing "
                f"to connect the narrative disclosure about {text_topic}, the financial table covering "
                f"{table_topic_text}, and the related {kind} image."
            )
        elif table_chunk:
            table_topic_text = table_topic(clean_text(table_chunk.get("summary")))
            question = (
                f"Retrieve all evidence needed from {company_name(text_chunk)}'s {form_name(text_chunk)} filing "
                f"to connect the narrative disclosure about {text_topic} with the table covering {table_topic_text}."
            )
        else:
            kind = visual_kind(clean_text(image_chunk.get("summary") if image_chunk else "")) or "chart"
            question = (
                f"Retrieve all evidence needed from {company_name(text_chunk)}'s filing to connect the narrative "
                f"disclosure about {text_topic} with the related {kind} image."
            )
        add_query(queries, qrels, f"hard_multi_{i:03d}", "multimodal", question, pair)

    full_corpus = [corpus_row(chunk) for chunk in chunks]
    gold_ids = {row["chunk_id"] for row in qrels}
    gold_corpus = [corpus_row(chunk) for chunk in chunks if chunk["id"] in gold_ids]

    write_jsonl(out_dir / "queries.jsonl", queries)
    write_jsonl(out_dir / "qrels.jsonl", qrels)
    write_jsonl(out_dir / "corpus.jsonl", full_corpus)
    write_jsonl(out_dir / "gold_corpus.jsonl", gold_corpus)

    qrels_per_query = Counter(row["query_id"] for row in qrels)
    summary = {
        "chunks_path": str(chunks_path).replace("\\", "/"),
        "queries": len(queries),
        "qrels": len(qrels),
        "full_corpus_chunks": len(full_corpus),
        "gold_corpus_chunks": len(gold_corpus),
        "query_type_counts": dict(Counter(row["type"] for row in queries)),
        "qrel_modality_counts": dict(Counter(row["modality"] for row in qrels)),
        "qrels_per_query": {
            "min": min(qrels_per_query.values()) if qrels_per_query else 0,
            "max": max(qrels_per_query.values()) if qrels_per_query else 0,
            "avg": sum(qrels_per_query.values()) / len(qrels_per_query) if qrels_per_query else 0,
            "distribution": dict(Counter(qrels_per_query.values())),
        },
        "note": (
            "Hard benchmark excludes cover-page metadata questions, uses chunk-id qrels, "
            "includes image object qrels, and includes multimodal queries with two required evidence chunks."
        ),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def round_robin_pairs(pairs: list[list[dict[str, Any]]], limit: int) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    order = []
    for pair in pairs:
        source = pair[0].get("source_pdf") or ""
        if source not in grouped:
            order.append(source)
        grouped[source].append(pair)
    selected = []
    while len(selected) < limit:
        added = False
        for source in order:
            if grouped[source]:
                selected.append(grouped[source].pop(0))
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a harder chunk-level retrieval benchmark.")
    parser.add_argument("--chunks", type=Path, default=CHUNKS_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--text", type=int, default=60)
    parser.add_argument("--table", type=int, default=40)
    parser.add_argument("--image", type=int, default=25)
    parser.add_argument("--multimodal", type=int, default=25)
    args = parser.parse_args()

    summary = build_hard_benchmark(
        chunks_path=args.chunks,
        out_dir=args.out_dir,
        text_n=args.text,
        table_n=args.table,
        image_n=args.image,
        multimodal_n=args.multimodal,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
