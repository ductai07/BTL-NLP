from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("data/graph/chunk_graph.jsonl")
DEFAULT_NODES_OUT = Path("data/graph/nodes.jsonl")
DEFAULT_EDGES_OUT = Path("data/graph/edges.jsonl")
DEFAULT_RELATIONS_OUT = Path("data/graph/relations.jsonl")
DEFAULT_NODES_CSV_OUT = Path("data/graph/nodes.csv")
DEFAULT_EDGES_CSV_OUT = Path("data/graph/edges.csv")
DEFAULT_RELATIONS_CSV_OUT = Path("data/graph/relations.csv")
DEFAULT_SUMMARY_OUT = Path("data/graph/graph_nodes_edges_summary.json")

DEFAULT_RELATION_WHITELIST = {
    "is_mentioned_in",
    "equivalent",
    "has_ticker",
    "listed_on",
    "files_report",
    "located_in",
    "has_address",
    "has_phone",
    "acquired",
    "partnered_with",
    "offers",
    "owns",
    "is",
    "increased",
    "decreased",
    "improved",
    "worsened",
    "reported",
    "generated",
    "incurred",
    "guidance_for",
    "expects",
    "projects",
    "merged_with",
    "invested_in",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunk_id(row: dict[str, Any]) -> str:
    value = row.get("chunk_id") or row.get("id")
    if value is None or not str(value).strip():
        raise ValueError("missing chunk id")
    return str(value)


def chunk_text(row: dict[str, Any]) -> str:
    for field in ("embed_text", "text", "summary"):
        value = row.get(field)
        if value and str(value).strip():
            return str(value)
    return ""


def normalize_entity_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip().lower()
    text = re.sub(r"[^\w\s&.-]", "", text)
    return text


def entity_node_id(entity_name: str) -> str:
    key = normalize_entity_name(entity_name)
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return f"entity:{digest}"


def chunk_node_id(cid: str) -> str:
    return f"chunk:{cid}"


def edge_id(src: str, rel: str, dst: str) -> str:
    raw = f"{src}|{rel}|{dst}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    return f"edge:{digest}"


def build_graph(
    *,
    input_path: Path,
    nodes_out: Path,
    edges_out: Path,
    relations_out: Path,
    nodes_csv_out: Path,
    edges_csv_out: Path,
    relations_csv_out: Path,
    summary_out: Path,
    add_equivalent_edges: bool,
    relation_whitelist: set[str] | None,
    drop_low_value_edges: bool,
) -> dict[str, Any]:
    rows = load_jsonl(input_path)

    node_map: dict[str, dict[str, Any]] = {}
    edge_map: dict[str, dict[str, Any]] = {}
    relation_map: dict[str, dict[str, Any]] = {}
    entities_by_norm: dict[str, set[str]] = {}

    for row in rows:
        try:
            cid = chunk_id(row)
        except ValueError:
            continue

        c_node_id = chunk_node_id(cid)
        node_map[c_node_id] = {
            "node_id": c_node_id,
            "node_type": "chunk",
            "chunk_id": cid,
            "source_pdf": row.get("source_pdf"),
            "ticker": row.get("ticker"),
            "modality": row.get("modality"),
            "text_preview": chunk_text(row)[:500],
        }

        graph = row.get("graph") if isinstance(row.get("graph"), dict) else {}
        entities = graph.get("entities") if isinstance(graph.get("entities"), list) else []
        triples = graph.get("clean_triples") if isinstance(graph.get("clean_triples"), list) else []
        if not triples:
            triples = graph.get("triples") if isinstance(graph.get("triples"), list) else []

        relation_map.setdefault(
            "is_mentioned_in",
            {"relation": "is_mentioned_in", "attributes": {"source": "system"}},
        )

        for entity in entities:
            entity_name = str(entity).strip()
            if not entity_name:
                continue
            e_node_id = entity_node_id(entity_name)
            node_map[e_node_id] = {
                "node_id": e_node_id,
                "node_type": "entity",
                "name": entity_name,
                "name_normalized": normalize_entity_name(entity_name),
            }
            entities_by_norm.setdefault(normalize_entity_name(entity_name), set()).add(e_node_id)

            rel = "is_mentioned_in"
            e_id = edge_id(e_node_id, rel, c_node_id)
            edge_map[e_id] = {
                "edge_id": e_id,
                "source_node": e_node_id,
                "relation": rel,
                "target_node": c_node_id,
                "provenance": {"chunk_id": cid, "source": "system", "confidence": 1.0},
            }

        for triple in triples:
            if not isinstance(triple, list) or len(triple) != 3:
                continue
            s_text = str(triple[0]).strip()
            p_text = str(triple[1]).strip()
            o_text = str(triple[2]).strip()
            if not s_text or not p_text or not o_text:
                continue
            if relation_whitelist is not None and p_text not in relation_whitelist:
                continue
            if drop_low_value_edges and is_low_value_triple(s_text, p_text, o_text):
                continue

            relation_map.setdefault(
                p_text,
                {"relation": p_text, "attributes": {"source": "llm_openie"}},
            )
            s_id = entity_node_id(s_text)
            o_id = entity_node_id(o_text)
            node_map[s_id] = {
                "node_id": s_id,
                "node_type": "entity",
                "name": s_text,
                "name_normalized": normalize_entity_name(s_text),
            }
            node_map[o_id] = {
                "node_id": o_id,
                "node_type": "entity",
                "name": o_text,
                "name_normalized": normalize_entity_name(o_text),
            }
            entities_by_norm.setdefault(normalize_entity_name(s_text), set()).add(s_id)
            entities_by_norm.setdefault(normalize_entity_name(o_text), set()).add(o_id)

            triple_edge_id = edge_id(s_id, p_text, o_id)
            edge_map[triple_edge_id] = {
                "edge_id": triple_edge_id,
                "source_node": s_id,
                "relation": p_text,
                "target_node": o_id,
                "provenance": {"chunk_id": cid, "source": "llm_triple", "confidence": 0.7},
            }

    if add_equivalent_edges:
        relation_map.setdefault(
            "equivalent",
            {"relation": "equivalent", "attributes": {"source": "system_normalized_name"}},
        )
        for _, node_ids in entities_by_norm.items():
            if len(node_ids) < 2:
                continue
            ids = sorted(node_ids)
            for i in range(len(ids) - 1):
                left = ids[i]
                right = ids[i + 1]
                forward_id = edge_id(left, "equivalent", right)
                backward_id = edge_id(right, "equivalent", left)
                edge_map[forward_id] = {
                    "edge_id": forward_id,
                    "source_node": left,
                    "relation": "equivalent",
                    "target_node": right,
                    "provenance": {"source": "system_normalized_name"},
                }
                edge_map[backward_id] = {
                    "edge_id": backward_id,
                    "source_node": right,
                    "relation": "equivalent",
                    "target_node": left,
                    "provenance": {"source": "system_normalized_name"},
                }

    nodes = sorted(node_map.values(), key=lambda x: x["node_id"])
    edges = sorted(edge_map.values(), key=lambda x: x["edge_id"])
    relations = sorted(relation_map.values(), key=lambda x: x["relation"])

    write_jsonl(nodes_out, nodes)
    write_jsonl(edges_out, edges)
    write_jsonl(relations_out, relations)
    write_csv_files(nodes, edges, relations, nodes_csv_out, edges_csv_out, relations_csv_out)

    summary = {
        "input_path": str(input_path).replace("\\", "/"),
        "nodes_out": str(nodes_out).replace("\\", "/"),
        "edges_out": str(edges_out).replace("\\", "/"),
        "relations_out": str(relations_out).replace("\\", "/"),
        "nodes_csv_out": str(nodes_csv_out).replace("\\", "/"),
        "edges_csv_out": str(edges_csv_out).replace("\\", "/"),
        "relations_csv_out": str(relations_csv_out).replace("\\", "/"),
        "total_chunks": sum(1 for node in nodes if node.get("node_type") == "chunk"),
        "total_entities": sum(1 for node in nodes if node.get("node_type") == "entity"),
        "total_nodes": len(nodes),
        "total_relations": len(relations),
        "total_edges": len(edges),
        "triple_edges": sum(1 for edge in edges if edge.get("provenance", {}).get("source") == "llm_triple"),
        "mention_edges": sum(1 for edge in edges if edge.get("relation") == "is_mentioned_in"),
        "equivalent_edges": sum(1 for edge in edges if edge.get("relation") == "equivalent"),
        "add_equivalent_edges": add_equivalent_edges,
        "drop_low_value_edges": drop_low_value_edges,
        "relation_whitelist": sorted(relation_whitelist) if relation_whitelist is not None else None,
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_csv_files(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    nodes_csv_out: Path,
    edges_csv_out: Path,
    relations_csv_out: Path,
) -> None:
    nodes_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with nodes_csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["uid", "name", "type", "attributes"])
        writer.writeheader()
        for node in nodes:
            node_name = node.get("name") or node.get("chunk_id") or node.get("node_id")
            writer.writerow(
                {
                    "uid": node["node_id"],
                    "name": node_name,
                    "type": node.get("node_type"),
                    "attributes": json.dumps(
                        {
                            key: value
                            for key, value in node.items()
                            if key not in {"node_id", "node_type", "name"}
                        },
                        ensure_ascii=False,
                    ),
                }
            )

    with relations_csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "attributes"])
        writer.writeheader()
        for relation in relations:
            writer.writerow(
                {
                    "name": relation["relation"],
                    "attributes": json.dumps(relation.get("attributes", {}), ensure_ascii=False),
                }
            )

    with edges_csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "relation", "target", "attributes"])
        writer.writeheader()
        for edge in edges:
            writer.writerow(
                {
                    "source": edge["source_node"],
                    "relation": edge["relation"],
                    "target": edge["target_node"],
                    "attributes": json.dumps(edge.get("provenance", {}), ensure_ascii=False),
                }
            )


def normalize_text_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def looks_mostly_numeric(value: str) -> bool:
    cleaned = re.sub(r"[\s,.$()%:-]", "", value)
    if not cleaned:
        return False
    digits = sum(ch.isdigit() for ch in cleaned)
    return digits / len(cleaned) >= 0.6


def is_low_value_triple(subject: str, relation: str, obj: str) -> bool:
    subj = normalize_text_key(subject)
    rel = normalize_text_key(relation)
    obj_norm = normalize_text_key(obj)

    # Chỉ lọc rất nhẹ để tránh làm mất thông tin hữu ích.
    if subj == obj_norm:
        return True
    # Các fact trạng thái yes/no thường nhiễu cho retrieval.
    if obj_norm in {"yes", "no", "true", "false"}:
        return True
    # Một số relation trạng thái hệ thống quá đặc thù.
    if rel in {"filer_status", "shell_company_status"}:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Graph RAG nodes/edges from chunk_graph.jsonl.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--nodes-out", type=Path, default=DEFAULT_NODES_OUT)
    parser.add_argument("--edges-out", type=Path, default=DEFAULT_EDGES_OUT)
    parser.add_argument("--relations-out", type=Path, default=DEFAULT_RELATIONS_OUT)
    parser.add_argument("--nodes-csv-out", type=Path, default=DEFAULT_NODES_CSV_OUT)
    parser.add_argument("--edges-csv-out", type=Path, default=DEFAULT_EDGES_CSV_OUT)
    parser.add_argument("--relations-csv-out", type=Path, default=DEFAULT_RELATIONS_CSV_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument(
        "--add-equivalent-edges",
        action="store_true",
        help="Add entity-equivalent edges using normalized entity names.",
    )
    parser.add_argument(
        "--drop-low-value-edges",
        action="store_true",
        help="Bo edge it gia tri: self-loop, object yes/no, relation filer_status/shell_company_status.",
    )
    parser.add_argument(
        "--relation-whitelist",
        help="Comma-separated relations to keep. Example: has_ticker,listed_on,files_report,is_mentioned_in,equivalent",
    )
    parser.add_argument(
        "--use-default-relation-whitelist",
        action="store_true",
        help="Use default retrieval-focused relation whitelist.",
    )
    args = parser.parse_args()

    relation_whitelist: set[str] | None = None
    if args.use_default_relation_whitelist:
        relation_whitelist = set(DEFAULT_RELATION_WHITELIST)
    if args.relation_whitelist:
        user_relations = {
            rel.strip()
            for rel in args.relation_whitelist.split(",")
            if rel.strip()
        }
        relation_whitelist = user_relations if relation_whitelist is None else (relation_whitelist | user_relations)

    summary = build_graph(
        input_path=args.input,
        nodes_out=args.nodes_out,
        edges_out=args.edges_out,
        relations_out=args.relations_out,
        nodes_csv_out=args.nodes_csv_out,
        edges_csv_out=args.edges_csv_out,
        relations_csv_out=args.relations_csv_out,
        summary_out=args.summary_out,
        add_equivalent_edges=args.add_equivalent_edges,
        relation_whitelist=relation_whitelist,
        drop_low_value_edges=args.drop_low_value_edges,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
