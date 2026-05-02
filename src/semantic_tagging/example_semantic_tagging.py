from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from semantic_tagging.query_tagger import QueryTagger
from semantic_tagging.retrieval_with_tags import tag_overlap_score
from semantic_tagging.fallback import fallback_chunk_tags


EXAMPLE_CHUNK = {
    "chunk_id": "3M_2022_10K_p72_c1",
    "doc_name": "3M_2022_10K",
    "company": "3M",
    "doc_type": "10-K",
    "doc_period": 2022,
    "page": 72,
    "section": "Consolidated Statement of Cash Flows",
    "text": "3M Company and Subsidiaries Consolidated Statement of Cash Flows Years ended December 31 Millions 2022 2021 2020 Purchases of property, plant and equipment (1,749) (1,593) (1,497)",
}


def main() -> None:
    chunk_tags = fallback_chunk_tags(EXAMPLE_CHUNK)
    chunk_tags["_semantic_tag_source"] = "fallback_example_only"
    query_tags = QueryTagger(dry_run=True).tag_query("What were 3M capital expenditures in 2022?")
    print(json.dumps({"semantic_tags": chunk_tags, "query_tags": query_tags}, indent=2, ensure_ascii=False))
    print("tag_overlap_score:", round(tag_overlap_score(query_tags, chunk_tags), 4))

    assert "3M" in chunk_tags["companies"]
    assert "2022" in chunk_tags["years"]
    assert "capital expenditure" in chunk_tags["financial_metrics"]
    assert "cash flow statement" in chunk_tags["section_tags"]
    assert chunk_tags["evidence_type"] == "table"


if __name__ == "__main__":
    main()
