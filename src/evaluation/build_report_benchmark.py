from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CHUNK_DIR = Path("data/chunks")
QA_PATH = Path("data/qa/eval_qa.jsonl")
OUT_DIR = Path("data/benchmark_report")

COMPANY_BY_TICKER = {
    "AAPL": "Apple",
    "HD": "Home Depot",
    "INTU": "Intuit",
    "MS": "Morgan Stanley",
    "NVDA": "NVIDIA",
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
    if len(parts) >= 2:
        return parts[1]
    return "filing"


def chunk_content(chunk: dict[str, Any]) -> str:
    parts = [
        chunk.get("embed_text"),
        chunk.get("text"),
        chunk.get("summary"),
        chunk.get("table_markdown"),
        json.dumps(chunk.get("table_json"), ensure_ascii=False) if chunk.get("table_json") else None,
    ]
    return "\n".join(clean_text(part) for part in parts if clean_text(part))


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


def is_bad_table(chunk: dict[str, Any]) -> bool:
    summary = clean_text(chunk.get("summary")).lower()
    page = chunk.get("page")
    if page in {1, 2, 3} and any(term in summary for term in ("table of contents", "commission file", "registrant")):
        return True
    bad_terms = (
        "table of contents",
        "commission file",
        "exact name of registrant",
        "large accelerated filer",
        "securities registered pursuant",
        "state or other jurisdiction",
        "telephone number",
    )
    return any(term in summary for term in bad_terms)


def is_good_financial_table(chunk: dict[str, Any]) -> bool:
    if chunk.get("modality") != "table" or is_bad_table(chunk):
        return False
    summary = clean_text(chunk.get("summary")).lower()
    signals = (
        "net sales",
        "revenue",
        "gross margin",
        "operating",
        "income",
        "cash",
        "assets",
        "liabilities",
        "share",
        "tax",
        "expense",
        "balance sheet",
        "statements of operations",
        "cash flow",
    )
    return any(signal in summary for signal in signals)


def normalize_question_from_qa(item: dict[str, Any]) -> str:
    question = clean_text(item.get("question"))
    source_pdf = item.get("source_pdf")
    company = company_name(source_pdf)
    evidence = clean_text(item.get("evidence"))

    context_match = re.search(r"Table context:\s*(.*?)\.\s*HTML table row:", evidence, flags=re.IGNORECASE)
    context = clean_text(context_match.group(1) if context_match else "")

    match = re.search(
        r"what value does (.*?) report for (.*?) under (.*?)\?$",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        row = clean_text(match.group(2))
        col = clean_text(match.group(3))
        metric = natural_metric(row, context, item.get("answer", ""))
        return f"What was {company}'s {metric} for {col}?"

    match = re.search(
        r"For (.*?)'s filing period ended (.*?), what value is reported for (.*?) under (.*?)\?$",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        period = clean_text(match.group(2))
        row = clean_text(match.group(3))
        col = clean_text(match.group(4))
        metric = natural_metric(row, context, item.get("answer", ""))
        return f"For the period ended {period}, what was {company}'s {metric} for {col}?"

    return question


def natural_metric(row: str, context: str, answer: str = "") -> str:
    row_clean = clean_text(row)
    context_l = context.lower()
    row_l = row_clean.lower()
    answer_is_percent = "%" in str(answer)
    if answer_is_percent and "gross margin percentage" in context_l and "percentage" not in row_l:
        return f"{row_clean} gross margin percentage"
    if "gross margin" in context_l and "gross margin" not in row_l:
        return f"{row_clean} gross margin"
    if "net sales" in context_l and "net sales" not in row_l:
        return f"{row_clean} net sales"
    if "revenue" in context_l and "revenue" not in row_l:
        return f"{row_clean} revenue"
    if "cash flow" in context_l and "cash flow" not in row_l:
        return f"{row_clean} cash flow"
    if "balance sheet" in context_l and row_l not in {"assets", "liabilities"}:
        return row_clean
    if "fees billed" in context_l and "fees" not in row_l:
        return f"{row_clean} fees"
    return row_clean


def score_candidate(item: dict[str, Any], chunk: dict[str, Any]) -> float:
    evidence = clean_text(item.get("evidence"))
    answer = clean_text(item.get("answer"))
    content = chunk_content(chunk).lower()
    score = 0.0
    for token in re.findall(r"[a-zA-Z][a-zA-Z]+|\d[\d,%.()$-]*", evidence):
        token_l = token.lower()
        if token_l in content:
            score += 1.0
    if answer and answer.lower() in content:
        score += 8.0
    return score


def map_qa_to_chunks(
    item: dict[str, Any],
    chunks_by_source: dict[str, list[dict[str, Any]]],
    modality: str,
    max_qrels: int,
) -> list[dict[str, Any]]:
    source_pdf = item.get("source_pdf")
    page = item.get("page")
    candidates = [
        chunk
        for chunk in chunks_by_source.get(source_pdf, [])
        if chunk.get("modality") == modality and page_overlaps(chunk, page)
    ]
    if modality == "table":
        candidates = [chunk for chunk in candidates if is_good_financial_table(chunk)]
    scored = [
        {"chunk": chunk, "score": score_candidate(item, chunk)}
        for chunk in candidates
    ]
    scored = [row for row in scored if row["score"] > 0]
    scored.sort(key=lambda row: row["score"], reverse=True)
    return scored[:max_qrels]


def add_query(
    queries: list[dict[str, Any]],
    qrels: list[dict[str, Any]],
    query_id: str,
    qtype: str,
    question: str,
    answer: str,
    answer_type: str,
    chunks: list[dict[str, Any]],
    benchmark_source: str,
) -> None:
    queries.append(
        {
            "query_id": query_id,
            "question": clean_text(question),
            "type": qtype,
            "answer": clean_text(answer),
            "answer_type": answer_type or "text",
            "source_pdf": chunks[0].get("source_pdf") if chunks else None,
            "evidence_chunk_ids": [chunk["id"] for chunk in chunks],
            "benchmark_source": benchmark_source,
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


def text_topic(chunk: dict[str, Any]) -> str | None:
    text = clean_text(chunk.get("text"))
    lower = text.lower()
    company = company_name(chunk.get("source_pdf"))
    if "iphone is the company" in lower or "iphone lineup" in lower:
        return f"What products are included in {company}'s iPhone lineup?"
    if "net sales" in lower and ("increased" in lower or "decreased" in lower):
        return f"What does {company} say drove the change in net sales?"
    if "revenue" in lower and ("increased" in lower or "decreased" in lower):
        return f"What does {company} say about revenue performance in this filing?"
    if "gross margin" in lower:
        return f"What does {company} explain about gross margin in this filing?"
    if re.search(r"\b(liquidity|cash|cash equivalents)\b", lower):
        return f"What does {company} disclose about cash or liquidity?"
    if "risk" in lower and len(text.split()) > 100:
        return f"What risk factor does {company} discuss in this section?"
    if "segment" in lower and ("sales" in lower or "operating" in lower):
        return f"What does {company} disclose about segment performance?"
    return None


def build_text_queries(chunks: list[dict[str, Any]], target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    selected_sources = Counter()
    candidates = []
    for chunk in chunks:
        if chunk.get("modality") != "text":
            continue
        page = chunk.get("page_start", chunk.get("page"))
        if page is not None and int(page) <= 3:
            continue
        question = text_topic(chunk)
        if not question:
            continue
        if len(clean_text(chunk.get("text")).split()) < 90:
            continue
        candidates.append((chunk, question))

    candidates.sort(key=lambda pair: (selected_sources[pair[0].get("source_pdf")], pair[0].get("source_pdf") or ""))
    used_questions = set()
    for chunk, question in candidates:
        if len(queries) >= target:
            break
        key = (chunk.get("source_pdf"), question)
        if key in used_questions:
            continue
        used_questions.add(key)
        selected_sources[chunk.get("source_pdf")] += 1
        qid = f"r_text_{len(queries) + 1:03d}"
        add_query(queries, qrels, qid, "text", question, "", "text", [chunk], "natural_text_from_chunk")
    return queries, qrels


def build_table_queries(
    qa_items: list[dict[str, Any]],
    chunks_by_source: dict[str, list[dict[str, Any]]],
    target: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    used_questions = set()
    for item in qa_items:
        if item.get("type") != "table":
            continue
        if any(bad in clean_text(item.get("evidence")).lower() for bad in ("table of contents", "commission file", "registrant")):
            continue
        raw_question = clean_text(item.get("question"))
        # Filter mechanically parsed proxy/ratio fragments that are not natural
        # financial questions.
        if re.search(r"\$\d|\bbillion\b|compared to|fiscal 20\d{2}", raw_question, flags=re.IGNORECASE):
            continue
        matches = map_qa_to_chunks(item, chunks_by_source, "table", max_qrels=1)
        if not matches:
            audit.append({"id": item.get("id"), "reason": "no_good_table_chunk", "question": item.get("question")})
            continue
        question = normalize_question_from_qa(item)
        if re.search(r"\$\d|\bbillion\b|compared to", question, flags=re.IGNORECASE):
            continue
        key = question.lower()
        if key in used_questions:
            continue
        used_questions.add(key)
        qid = f"r_table_{len(queries) + 1:03d}"
        add_query(
            queries,
            qrels,
            qid,
            "table",
            question,
            item.get("answer", ""),
            item.get("answer_type", "text"),
            [matches[0]["chunk"]],
            "natural_table_from_qa",
        )
        if len(queries) >= target:
            break
    return queries, qrels, audit


def visual_kind(chunk_or_summary: dict[str, Any] | str) -> str | None:
    if isinstance(chunk_or_summary, dict):
        metadata = chunk_or_summary.get("visual_metadata") or {}
        visual_type = clean_text(metadata.get("visual_type")).lower()
        summary = clean_text(chunk_or_summary.get("summary"))
    else:
        visual_type = ""
        summary = clean_text(chunk_or_summary)

    if visual_type:
        visual_type = visual_type.replace("_", " ")
        if visual_type in {"line chart", "line graph"}:
            return "line chart"
        if visual_type == "bar chart":
            return "bar chart"
        if visual_type == "pie chart":
            return "pie chart"
        if visual_type == "chart":
            return "chart"

    lower = summary.lower()
    if "bar chart" in lower:
        return "bar chart"
    if "line chart" in lower or "line graph" in lower or "multiple lines" in lower:
        return "line chart"
    if "pie chart" in lower:
        return "pie chart"
    if "chart" in lower:
        return "chart"
    return None


def image_qa(chunk: dict[str, Any]) -> tuple[str, str, str] | None:
    kind = visual_kind(chunk)
    if kind not in {"line chart", "bar chart", "pie chart"}:
        return None

    summary = clean_text(chunk.get("summary"))
    if len(summary.split()) < 20:
        return None

    company = company_name(chunk.get("source_pdf"))
    metadata = chunk.get("visual_metadata") or {}
    trend = clean_text(metadata.get("trend")).lower()
    metrics = [clean_text(value) for value in metadata.get("metrics") or [] if clean_text(value)]
    periods = [clean_text(value) for value in metadata.get("periods") or [] if clean_text(value)]
    key_values = [clean_text(value) for value in metadata.get("key_values") or [] if clean_text(value)]

    descriptor_parts = []
    if metrics:
        descriptor_parts.append(f"about {', '.join(metrics[:2])}")
    if periods:
        descriptor_parts.append(f"covering {', '.join(periods[:2])}")
    if key_values:
        descriptor_parts.append(f"with visible labels such as {', '.join(key_values[:3])}")
    descriptor = " " + " ".join(descriptor_parts) if descriptor_parts else ""

    if trend in {"upward", "downward", "mixed", "stable"}:
        return (
            f"What overall trend is shown in the {company} filing {kind}{descriptor}?",
            trend,
            "text",
        )

    return (
        f"What type of visual is shown in the {company} filing image{descriptor}?",
        kind,
        "text",
    )


def image_question(chunk: dict[str, Any]) -> str | None:
    qa = image_qa(chunk)
    return qa[0] if qa else None


def build_image_queries(chunks: list[dict[str, Any]], target: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    candidates = []
    for chunk in chunks:
        if chunk.get("modality") != "image":
            continue
        qa = image_qa(chunk)
        if not qa:
            continue
        candidates.append((chunk, qa))

    used = set()
    for chunk, qa in candidates:
        if len(queries) >= target:
            break
        question, answer, answer_type = qa
        key = (chunk.get("source_pdf"), question)
        if key in used:
            continue
        used.add(key)
        qid = f"r_image_{len(queries) + 1:03d}"
        add_query(
            queries,
            qrels,
            qid,
            "image",
            question,
            answer,
            answer_type,
            [chunk],
            "visual_metadata_answerable_image",
        )
    return queries, qrels


def build_multimodal_queries(
    table_queries: list[dict[str, Any]],
    table_qrels: list[dict[str, Any]],
    text_chunks_by_source: dict[str, list[dict[str, Any]]],
    image_chunks_by_source: dict[str, list[dict[str, Any]]],
    chunks_by_id: dict[str, dict[str, Any]],
    target: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    queries: list[dict[str, Any]] = []
    qrels: list[dict[str, Any]] = []
    table_rel_by_q = {row["query_id"]: row for row in table_qrels}
    for table_query in table_queries:
        if len(queries) >= target:
            break
        table_rel = table_rel_by_q.get(table_query["query_id"])
        if not table_rel:
            continue
        table_chunk = chunks_by_id[table_rel["chunk_id"]]
        source = table_chunk.get("source_pdf")
        text_candidates = text_chunks_by_source.get(source, [])
        image_candidates = image_chunks_by_source.get(source, [])
        if not text_candidates or not image_candidates:
            continue
        text_chunk = text_candidates[0]
        image_chunk = image_candidates[0]
        company = company_name(source)
        base_q = table_query["question"].rstrip("?")
        kind = visual_kind(clean_text(image_chunk.get("summary"))) or "chart"
        question = (
            f"For {company}, use the narrative discussion, the relevant financial table, "
            f"and the {kind} image to support this question: {base_q}?"
        )
        qid = f"r_multi_{len(queries) + 1:03d}"
        add_query(
            queries,
            qrels,
            qid,
            "multimodal",
            question,
            table_query.get("answer", ""),
            table_query.get("answer_type", "text"),
            [text_chunk, table_chunk, image_chunk],
            "natural_multimodal_text_table_image",
        )
    return queries, qrels


def build_report_benchmark(qa_path: Path, chunk_dir: Path, out_dir: Path) -> dict[str, Any]:
    qa_items = load_jsonl(qa_path)
    chunks = load_jsonl(chunk_dir / "all_chunks.jsonl")
    chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
    chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    text_chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    image_chunks_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        chunks_by_source[chunk.get("source_pdf")].append(chunk)
        if chunk.get("modality") == "text":
            page = chunk.get("page_start", chunk.get("page"))
            if page is None or int(page) > 3:
                text_chunks_by_source[chunk.get("source_pdf")].append(chunk)
        elif chunk.get("modality") == "image" and image_question(chunk):
            image_chunks_by_source[chunk.get("source_pdf")].append(chunk)

    text_queries, text_qrels = build_text_queries(chunks, target=40)
    table_queries, table_qrels, audit = build_table_queries(qa_items, chunks_by_source, target=50)
    image_queries, image_qrels = build_image_queries(chunks, target=25)
    multi_queries, multi_qrels = build_multimodal_queries(
        table_queries,
        table_qrels,
        text_chunks_by_source,
        image_chunks_by_source,
        chunks_by_id,
        target=25,
    )

    queries = text_queries + table_queries + image_queries + multi_queries
    qrels = text_qrels + table_qrels + image_qrels + multi_qrels
    full_corpus = [corpus_row(chunk) for chunk in chunks]
    gold_ids = {row["chunk_id"] for row in qrels}
    gold_corpus = [corpus_row(chunk) for chunk in chunks if chunk["id"] in gold_ids]

    write_jsonl(out_dir / "queries.jsonl", queries)
    write_jsonl(out_dir / "qrels.jsonl", qrels)
    write_jsonl(out_dir / "corpus.jsonl", full_corpus)
    write_jsonl(out_dir / "gold_corpus.jsonl", gold_corpus)
    write_jsonl(out_dir / "audit_rejected.jsonl", audit)

    qrels_per_query = Counter(row["query_id"] for row in qrels)
    summary = {
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
            "Report benchmark with natural questions. It excludes cover-page metadata and table-of-contents qrels. "
            "Multimodal queries require text, table, and image evidence chunks."
        ),
    }
    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a report-quality retrieval benchmark with natural queries.")
    parser.add_argument("--qa", type=Path, default=QA_PATH)
    parser.add_argument("--chunk-dir", type=Path, default=CHUNK_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    summary = build_report_benchmark(args.qa, args.chunk_dir, args.out_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
