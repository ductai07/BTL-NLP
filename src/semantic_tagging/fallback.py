from __future__ import annotations

import re
from typing import Any


CHUNK_TAG_SCHEMA: dict[str, Any] = {
    "chunk_role": "unknown",
    "is_navigation_only": False,
    "is_front_matter": False,
    "companies": [],
    "years": [],
    "dates": [],
    "organizations": [],
    "products": [],
    "industries": [],
    "sectors": [],
    "financial_metrics": [],
    "business_topics": [],
    "risk_topics": [],
    "section_tags": [],
    "evidence_type": "unknown",
    "retrieval_keywords": [],
}

QUERY_TAG_SCHEMA: dict[str, Any] = {
    "companies": [],
    "years": [],
    "dates": [],
    "organizations": [],
    "products": [],
    "financial_metrics": [],
    "business_topics": [],
    "risk_topics": [],
    "expected_sections": [],
    "evidence_type_needed": "unknown",
    "intent": "unknown",
    "retrieval_keywords": [],
}

VALID_EVIDENCE_TYPES = {"text", "table", "mixed", "figure", "metadata", "unknown"}
VALID_CHUNK_ROLES = {
    "cover_page",
    "table_of_contents",
    "section_content",
    "financial_statement",
    "footnote",
    "risk_factor",
    "mdna",
    "signature",
    "exhibit_list",
    "metadata",
    "unknown",
}
VALID_INTENTS = {"extract_value", "calculate", "compare", "explain", "summarize", "unknown"}

METRIC_PATTERNS = {
    "revenue": r"\brevenue\b",
    "net sales": r"\bnet sales\b",
    "operating income": r"\boperating income\b",
    "net income": r"\bnet income\b",
    "gross profit": r"\bgross profit\b",
    "gross margin": r"\bgross margin\b",
    "capital expenditure": r"\bcap(?:ital)? ex(?:penditure)?\b|\bcapital expenditures?\b|\bpurchases of property, plant and equipment\b",
    "property plant and equipment": r"\bproperty,?\s+plant\s+and\s+equipment\b|\bpp&e\b|\bppe\b",
    "total assets": r"\btotal assets\b",
    "liabilities": r"\bliabilities\b",
    "cash flow": r"\bcash flows?\b|\bcash flow statement\b",
    "free cash flow": r"\bfree cash flow\b",
    "dividend": r"\bdividends?\b",
    "eps": r"\beps\b|\bearnings per share\b",
}

SECTION_PATTERNS = {
    "income statement": r"\bincome statement\b|\bstatement of operations\b",
    "balance sheet": r"\bbalance sheets?\b|\bfinancial position\b",
    "cash flow statement": r"\bcash flows?\b|\bcash flow statement\b",
    "legal proceedings": r"\blegal proceedings?\b",
    "md&a": r"\bmanagement'?s discussion and analysis\b|\bmd&a\b",
    "market risk": r"\bmarket risk\b",
    "financial statements": r"\bfinancial statements?\b",
    "corporate governance": r"\bcorporate governance\b",
    "forward-looking statements": r"\bforward-looking statements?\b",
    "risk factors": r"\brisk factors?\b",
    "part ii": r"\bpart ii\b",
    "part iii": r"\bpart iii\b",
    "part iv": r"\bpart iv\b",
    "segments": r"\bsegments?\b|\bsegment information\b",
}

BUSINESS_TOPIC_PATTERNS = {
    "capital investment": r"\bcapital expenditures?\b|\bpurchases of property, plant and equipment\b|\bcapital investment\b",
    "liquidity": r"\bliquidity\b|\bcash flows?\b|\bworking capital\b",
    "profitability": r"\boperating income\b|\bnet income\b|\bgross margin\b|\bgross profit\b",
    "sales performance": r"\brevenue\b|\bnet sales\b",
    "shareholder returns": r"\bdividends?\b|\bshare repurchases?\b|\bbuybacks?\b",
}

RISK_TOPIC_PATTERNS = {
    "market risk": r"\bmarket risk\b",
    "credit risk": r"\bcredit risk\b",
    "liquidity risk": r"\bliquidity risk\b",
    "cybersecurity risk": r"\bcybersecurity\b|\bcyber security\b",
    "legal risk": r"\blegal proceedings?\b|\blitigation\b",
}


def empty_chunk_tags() -> dict[str, Any]:
    return dict(CHUNK_TAG_SCHEMA)


def empty_query_tags() -> dict[str, Any]:
    return dict(QUERY_TAG_SCHEMA)


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = normalize_text(value)
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            result.append(clean)
    return result


def detect_years(text: str) -> list[str]:
    return unique(re.findall(r"\b20\d{2}\b", text))


def metadata_year(chunk: dict[str, Any]) -> list[str]:
    value = normalize_text(chunk.get("doc_period") or chunk.get("fiscal_year"))
    if re.fullmatch(r"20\d{2}", value):
        return [value]
    return []


def find_patterns(text: str, patterns: dict[str, str]) -> list[str]:
    return [label for label, pattern in patterns.items() if re.search(pattern, text, flags=re.IGNORECASE)]


def detect_evidence_type(text: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    modality = normalize_text(metadata.get("modality")).lower()
    section = normalize_text(metadata.get("section")).lower()
    combined = f"{section} {text}".lower()
    number_count = len(re.findall(r"[-(]?\$?\d[\d,]*(?:\.\d+)?%?\)?", text))
    line_count = len([line for line in text.splitlines() if line.strip()])

    if modality in {"table", "image"}:
        return "table" if modality == "table" else "figure"
    if re.search(r"\b(consolidated statement|balance sheet|cash flow statement|statement of cash flows)\b", combined):
        return "table"
    table_cues = re.search(
        r"\b(years ended|three months ended|six months ended|in millions|in thousands|assets|liabilities|net sales|total revenue)\b",
        combined,
    )
    if table_cues and number_count >= 8 and line_count >= 3:
        return "table"
    if re.search(r"\b(figure|chart|graph)\b", combined):
        return "figure"
    return "text" if text.strip() else "unknown"


def detect_chunk_role(text: str, chunk: dict[str, Any]) -> str:
    lowered = text.lower()
    page = chunk.get("page") or chunk.get("page_start")
    try:
        page_num = int(page)
    except (TypeError, ValueError):
        page_num = None

    item_hits = len(re.findall(r"\bitem\s+\d+[a-z]?\b", lowered))
    part_hits = len(re.findall(r"\bpart\s+(?:i|ii|iii|iv)\b", lowered))
    exhibit_hits = len(re.findall(r"\bexhibit\b", lowered))

    if "signatures" in lowered:
        return "signature"
    if "exhibit index" in lowered or exhibit_hits >= 4:
        return "exhibit_list"
    if "table of contents" in lowered or (item_hits >= 5 and part_hits >= 2):
        return "table_of_contents"
    if page_num == 1 and all(token in lowered for token in ("form 10-k", "commission file number")):
        return "cover_page"
    if page_num == 1 and "securities and exchange commission" in lowered and ("trading symbol" in lowered or "nasdaq" in lowered):
        return "cover_page"
    if re.search(r"\b(consolidated statement|balance sheet|statement of cash flows|statement of operations)\b", lowered):
        return "financial_statement"
    if re.search(r"\brisk factors?\b", lowered) and item_hits < 3:
        return "risk_factor"
    if re.search(r"\bmanagement'?s discussion and analysis\b|\bmd&a\b", lowered):
        return "mdna"
    if re.search(r"\bnotes to consolidated financial statements\b|\bnote \d+\b", lowered):
        return "footnote"
    return "section_content" if text.strip() else "unknown"


def is_navigation_role(role: str) -> bool:
    return role in {"table_of_contents", "exhibit_list"}


def is_front_matter_role(role: str) -> bool:
    return role in {"cover_page", "table_of_contents", "signature", "metadata"}


def build_retrieval_keywords(text: str, metrics: list[str], sections: list[str], limit: int = 12) -> list[str]:
    phrases = metrics + sections
    if re.search(r"\bpurchases of property, plant and equipment\b", text, flags=re.IGNORECASE):
        phrases.append("purchases of property plant and equipment")
        phrases.append("pp&e")
    for match in re.findall(r"\b(?:net sales|operating income|net income|gross profit|cash flow|total assets|earnings per share)\b", text, flags=re.IGNORECASE):
        phrases.append(match.lower())
    return unique(phrases)[:limit]


def fallback_chunk_tags(chunk: dict[str, Any]) -> dict[str, Any]:
    text = normalize_text(chunk.get("text") or chunk.get("embed_text") or chunk.get("summary"))
    metadata_text = " ".join(normalize_text(chunk.get(key)) for key in ("company", "ticker", "doc_name", "doc_type", "doc_period", "section"))
    combined = f"{metadata_text} {text}"
    metrics = find_patterns(combined, METRIC_PATTERNS)
    section_tags = find_patterns(combined, SECTION_PATTERNS)
    role = detect_chunk_role(text, chunk)
    is_navigation = is_navigation_role(role)
    is_front_matter = is_front_matter_role(role)
    is_metadata = role in {"cover_page", "table_of_contents", "signature", "exhibit_list", "metadata"}
    business_topics = [] if is_metadata or is_navigation else find_patterns(combined, BUSINESS_TOPIC_PATTERNS)
    risk_topics = [] if is_metadata or is_navigation else find_patterns(combined, RISK_TOPIC_PATTERNS)
    evidence_type = "metadata" if is_metadata else detect_evidence_type(text, chunk)
    years = metadata_year(chunk) or ([] if role == "cover_page" else detect_years(combined))
    retrieval_keywords = [] if is_metadata else build_retrieval_keywords(combined, metrics, section_tags)

    tags = empty_chunk_tags()
    company = normalize_text(chunk.get("company") or chunk.get("ticker"))
    tags.update(
        {
            "chunk_role": role,
            "is_navigation_only": is_navigation,
            "is_front_matter": is_front_matter,
            "companies": [company] if company else [],
            "years": years,
            "financial_metrics": [] if is_metadata else metrics,
            "business_topics": business_topics,
            "risk_topics": risk_topics,
            "section_tags": section_tags,
            "evidence_type": evidence_type,
            "retrieval_keywords": retrieval_keywords,
        }
    )
    return tags


def fallback_query_tags(question: str) -> dict[str, Any]:
    text = normalize_text(question)
    metrics = find_patterns(text, METRIC_PATTERNS)
    topics = find_patterns(text, BUSINESS_TOPIC_PATTERNS)
    risks = find_patterns(text, RISK_TOPIC_PATTERNS)
    sections = find_patterns(text, SECTION_PATTERNS)
    lowered = text.lower()

    intent = "unknown"
    if re.search(r"\b(calculate|compute|how much|difference|change|growth|margin)\b", lowered):
        intent = "calculate"
    elif re.search(r"\b(compare|versus|vs\.?|between)\b", lowered):
        intent = "compare"
    elif re.search(r"\b(explain|why|describe)\b", lowered):
        intent = "explain"
    elif re.search(r"\b(summarize|summary)\b", lowered):
        intent = "summarize"
    elif metrics or re.search(r"\b(what|which|show|find)\b", lowered):
        intent = "extract_value"

    evidence = "unknown"
    if re.search(r"\b(table|statement|balance sheet|cash flow)\b", lowered) or intent in {"calculate", "extract_value"} and metrics:
        evidence = "table"
    elif re.search(r"\b(chart|figure|graph|image)\b", lowered):
        evidence = "figure"
    elif intent in {"explain", "summarize"}:
        evidence = "text"

    tags = empty_query_tags()
    companies = unique(re.findall(r"\b(?:3M|[A-Z]{2,5})\b", question))
    tags.update(
        {
            "companies": companies,
            "years": detect_years(text),
            "financial_metrics": metrics,
            "business_topics": topics,
            "risk_topics": risks,
            "expected_sections": sections,
            "evidence_type_needed": evidence,
            "intent": intent,
            "retrieval_keywords": build_retrieval_keywords(text, metrics, sections),
        }
    )
    return tags
