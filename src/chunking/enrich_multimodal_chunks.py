from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


CHUNK_DIR = Path("data/chunks")

COMPANY_BY_TICKER = {
    "AAPL": "Apple",
    "HD": "Home Depot",
    "INTU": "Intuit",
    "MS": "Morgan Stanley",
    "NVDA": "NVIDIA",
}

FINANCIAL_TERMS = [
    "revenue",
    "net sales",
    "net income",
    "gross profit",
    "operating income",
    "operating expenses",
    "cash",
    "cash equivalents",
    "assets",
    "liabilities",
    "equity",
    "earnings",
    "eps",
    "diluted earnings per share",
    "dividends",
    "share repurchases",
    "free cash flow",
    "capital expenditures",
    "debt",
    "interest expense",
    "tax",
    "margin",
    "var",
    "risk",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def infer_ticker(source_pdf: str | None) -> str:
    if not source_pdf:
        return ""
    return Path(source_pdf).name.split("_", 1)[0]


def infer_form(source_pdf: str | None) -> str:
    if not source_pdf:
        return ""
    parts = Path(source_pdf).name.split("_")
    if len(parts) >= 2 and parts[1] == "DEF":
        return "DEF 14A proxy statement"
    return parts[1] if len(parts) >= 2 else ""


def company_name(source_pdf: str | None) -> str:
    ticker = infer_ticker(source_pdf)
    return COMPANY_BY_TICKER.get(ticker, ticker)


def base_context(chunk: dict[str, Any]) -> list[str]:
    source_pdf = chunk.get("source_pdf")
    page = chunk.get("page_start", chunk.get("page"))
    parts = [
        f"Company: {company_name(source_pdf)}" if company_name(source_pdf) else "",
        f"Ticker: {infer_ticker(source_pdf)}" if infer_ticker(source_pdf) else "",
        f"Filing type: {infer_form(source_pdf)}" if infer_form(source_pdf) else "",
        f"Source filing: {Path(source_pdf).name}" if source_pdf else "",
        f"Page: {page}" if page is not None else "",
        f"Modality: {chunk.get('modality')}",
    ]
    return [part for part in parts if clean_text(part)]


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def unique_clean(values: list[Any], limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = clean_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if limit is not None and len(out) >= limit:
            break
    return out


def looks_numeric(value: str) -> bool:
    text = value.strip()
    return bool(re.search(r"[$(]?\s*-?\d[\d,]*(?:\.\d+)?%?\)?", text))


def looks_empty_or_noise(value: str) -> bool:
    text = clean_text(value)
    if not text or text in {"-", "--", "—", "n/a", "N/A"}:
        return True
    if len(text) > 220:
        return True
    return False


def extract_periods(text: str) -> list[str]:
    patterns = [
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{4}\b",
        r"\bQ[1-4]\s+\d{4}\b",
        r"\b(?:first|second|third|fourth)\s+quarter\b",
    ]
    values: list[str] = []
    for pattern in patterns:
        values.extend(re.findall(pattern, text, flags=re.IGNORECASE))
    return unique_clean(values, limit=12)


def infer_unit(text: str) -> str:
    lower = text.lower()
    if "in millions" in lower or "millions" in lower:
        return "million"
    if "in billions" in lower or "billions" in lower:
        return "billion"
    if "%" in text or "percent" in lower:
        return "percent"
    if "$" in text:
        return "dollar"
    return ""


def financial_terms_in(text: str, limit: int = 12) -> list[str]:
    lower = text.lower()
    found: list[str] = []
    for term in FINANCIAL_TERMS:
        pattern = r"(?<![a-z0-9])" + re.escape(term.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, lower):
            found.append(term)
        if len(found) >= limit:
            break
    return found


def row_label_from(row: dict[str, Any], headers: list[str]) -> str:
    for header in headers:
        value = clean_text(row.get(header))
        if value and not looks_numeric(value) and len(value) <= 120:
            return value
    for value in row.values():
        text = clean_text(value)
        if text and not looks_numeric(text) and len(text) <= 120:
            return text
    return ""


def build_table_facts(chunk: dict[str, Any], max_facts: int) -> tuple[dict[str, Any], list[dict[str, str]]]:
    table_json = chunk.get("table_json") if isinstance(chunk.get("table_json"), dict) else {}
    headers = [clean_text(h) for h in as_list(table_json.get("headers")) if clean_text(h)]
    rows = table_json.get("rows") if isinstance(table_json.get("rows"), list) else []
    summary = clean_text(chunk.get("summary") or chunk.get("rule_summary") or chunk.get("text"))
    unit = infer_unit(" ".join([summary, " ".join(headers)]))
    periods = extract_periods(" ".join([summary, " ".join(headers)]))
    metrics = financial_terms_in(" ".join([summary, " ".join(headers)]))

    facts: list[dict[str, str]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        row_label = row_label_from(row, headers)
        row_text = " ".join(clean_text(v) for v in row.values())
        row_metric = financial_terms_in(" ".join([row_label, row_text]), limit=4)
        for column, raw_value in row.items():
            column_text = clean_text(column)
            value = clean_text(raw_value)
            if looks_empty_or_noise(value):
                continue
            if row_label and value == row_label:
                continue
            row_fin_terms = financial_terms_in(" ".join([row_label, row_text, column_text]), limit=4)
            if not looks_numeric(value) and not row_fin_terms:
                continue
            if value.startswith("(") and value.endswith(")") and not looks_numeric(value):
                continue
            if not looks_numeric(value) and len(value.split()) > 10:
                continue
            period = ", ".join(extract_periods(column_text) or extract_periods(value))
            metric = row_label or ", ".join(row_fin_terms or row_metric) or column_text
            facts.append(
                {
                    "row_index": str(row_index),
                    "row": row_label,
                    "column": column_text,
                    "value": value,
                    "metric": metric,
                    "period": period,
                    "unit": infer_unit(" ".join([value, column_text, summary])) or unit,
                }
            )
            if len(facts) >= max_facts:
                break
        if len(facts) >= max_facts:
            break

    row_labels = unique_clean([fact["row"] for fact in facts if fact.get("row")], limit=30)
    column_labels = unique_clean(headers or [fact["column"] for fact in facts], limit=30)
    metric_candidates = metrics + [term for fact in facts for term in financial_terms_in(fact.get("metric", ""))]
    metadata = {
        "title": summary[:240],
        "unit": unit,
        "periods": periods,
        "metrics": unique_clean(metric_candidates, limit=30),
        "row_labels": row_labels,
        "column_labels": column_labels,
        "num_rows": len(rows),
        "num_facts": len(facts),
    }
    return metadata, facts


def fact_text(facts: list[dict[str, str]], max_items: int = 40) -> str:
    items = []
    for fact in facts[:max_items]:
        left = fact.get("metric") or fact.get("row") or fact.get("column")
        column = fact.get("column")
        value = fact.get("value")
        period = fact.get("period")
        unit = fact.get("unit")
        parts = [left]
        if column and column != left:
            parts.append(column)
        if period:
            parts.append(period)
        if value:
            parts.append(value)
        if unit:
            parts.append(unit)
        items.append(" | ".join(clean_text(p) for p in parts if clean_text(p)))
    return "; ".join(items)


def enrich_table(chunk: dict[str, Any], max_facts: int) -> dict[str, Any]:
    row = dict(chunk)
    metadata, facts = build_table_facts(row, max_facts=max_facts)
    row["table_metadata"] = metadata
    row["table_facts"] = facts

    parts = base_context(row)
    parts.extend(
        [
            f"Table title/summary: {clean_text(row.get('summary') or row.get('text'))}",
            f"Table metrics: {', '.join(metadata['metrics'])}" if metadata["metrics"] else "",
            f"Table periods: {', '.join(metadata['periods'])}" if metadata["periods"] else "",
            f"Table unit: {metadata['unit']}" if metadata["unit"] else "",
            f"Row labels: {', '.join(metadata['row_labels'])}" if metadata["row_labels"] else "",
            f"Column labels: {', '.join(metadata['column_labels'])}" if metadata["column_labels"] else "",
            f"Key table facts: {fact_text(facts)}" if facts else "",
        ]
    )
    row["embed_text"] = "\n".join(part for part in parts if clean_text(part))
    return row


def infer_visual_type(text: str, current: str = "") -> str:
    lower = " ".join([current, text]).lower()
    if "bar chart" in lower or "bar graph" in lower:
        return "bar_chart"
    if "line chart" in lower or "line graph" in lower or "multiple lines" in lower:
        return "line_chart"
    if "pie chart" in lower:
        return "pie_chart"
    if "table" in lower or "grid" in lower:
        return "table_like_chart"
    if "diagram" in lower:
        return "diagram"
    if "logo" in lower:
        return "logo"
    if "chart" in lower:
        return "other_chart"
    if "unreadable" in lower or "not readable" in lower:
        return "unreadable"
    return "image"


def infer_trend(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ["upward", "increased", "rising", "grew", "higher"]):
        return "upward"
    if any(term in lower for term in ["downward", "decreased", "declining", "fell", "lower"]):
        return "downward"
    if any(term in lower for term in ["mixed", "fluctuate", "variation", "overlap"]):
        return "mixed"
    if any(term in lower for term in ["stable", "flat"]):
        return "stable"
    return "not_applicable"


def enrich_image(chunk: dict[str, Any]) -> dict[str, Any]:
    row = dict(chunk)
    vlm = row.get("vlm_output") if isinstance(row.get("vlm_output"), dict) else {}
    summary = clean_text(vlm.get("summary") or row.get("summary") or row.get("text"))
    key_values = unique_clean(as_list(vlm.get("key_values")), limit=40)
    legacy_summary = clean_text(row.get("legacy_summary"))
    legacy_key_values = unique_clean(as_list(row.get("legacy_key_values")), limit=40)
    legacy_embed_text = clean_text(row.get("legacy_embed_text"))
    combined = " ".join(
        [
            summary,
            legacy_summary,
            legacy_embed_text,
            " ".join(key_values),
            " ".join(legacy_key_values),
            clean_text(vlm.get("evidence")),
        ]
    )
    visual_type = clean_text(vlm.get("visual_type")) or infer_visual_type(combined)
    metadata = {
        "visual_type": visual_type,
        "title": clean_text(vlm.get("title")),
        "x_axis": clean_text(vlm.get("x_axis")),
        "y_axis": clean_text(vlm.get("y_axis")),
        "legend": unique_clean(as_list(vlm.get("legend")), limit=20),
        "metrics": unique_clean(as_list(vlm.get("metrics")) + financial_terms_in(combined), limit=20),
        "periods": unique_clean(as_list(vlm.get("periods")) + extract_periods(combined), limit=20),
        "key_values": unique_clean(key_values + legacy_key_values, limit=60),
        "trend": clean_text(vlm.get("trend")) or infer_trend(combined),
        "evidence": clean_text(vlm.get("evidence")),
    }
    row["visual_metadata"] = metadata

    image_path = row.get("image_path") or row.get("crop_path")
    image_name = Path(image_path).name if image_path else ""
    parts = base_context(row)
    parts.extend(
        [
            f"Visual type: {metadata['visual_type']}",
            f"Image title: {metadata['title']}" if metadata["title"] else "",
            f"Image file: {image_name}" if image_name else "",
            f"Image path: {image_path}" if image_path else "",
            f"X axis: {metadata['x_axis']}" if metadata["x_axis"] else "",
            f"Y axis: {metadata['y_axis']}" if metadata["y_axis"] else "",
            f"Legend: {', '.join(metadata['legend'])}" if metadata["legend"] else "",
            f"Visual metrics: {', '.join(metadata['metrics'])}" if metadata["metrics"] else "",
            f"Visual periods: {', '.join(metadata['periods'])}" if metadata["periods"] else "",
            f"Visible labels and values: {', '.join(metadata['key_values'])}" if metadata["key_values"] else "",
            f"Trend: {metadata['trend']}" if metadata["trend"] else "",
            f"Visual evidence: {metadata['evidence']}" if metadata["evidence"] else "",
            f"Legacy image summary: {legacy_summary}" if legacy_summary else "",
            f"Legacy retrieval text: {legacy_embed_text}" if legacy_embed_text else "",
            f"Image summary: {summary}",
        ]
    )
    row["embed_text"] = "\n".join(part for part in parts if clean_text(part))
    row["summary"] = summary or row.get("summary") or ""
    row["text"] = row["summary"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Improve table/image chunks with structured retrieval metadata.")
    parser.add_argument("--chunk-dir", type=Path, default=CHUNK_DIR)
    parser.add_argument("--max-table-facts", type=int, default=120)
    args = parser.parse_args()

    text_rows = load_jsonl(args.chunk_dir / "text_chunks.jsonl")
    table_rows = [enrich_table(row, args.max_table_facts) for row in load_jsonl(args.chunk_dir / "table_chunks.jsonl")]
    image_rows = [enrich_image(row) for row in load_jsonl(args.chunk_dir / "image_chunks.jsonl")]

    write_jsonl(args.chunk_dir / "table_chunks.jsonl", table_rows)
    write_jsonl(args.chunk_dir / "image_chunks.jsonl", image_rows)
    write_jsonl(args.chunk_dir / "all_chunks.jsonl", text_rows + table_rows + image_rows)

    summary = {
        "text_unchanged": len(text_rows),
        "table_enriched": len(table_rows),
        "image_enriched": len(image_rows),
        "all_chunks": len(text_rows) + len(table_rows) + len(image_rows),
        "table_facts": sum(len(row.get("table_facts") or []) for row in table_rows),
        "images_with_visual_metadata": sum(1 for row in image_rows if row.get("visual_metadata")),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
