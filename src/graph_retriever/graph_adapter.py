"""
Adapter: graph của project (graph_tensor/graph.pt + JSON) sang bundle chuẩn cho KGC.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

# Tên relation hệ thống entity → chunk trong pipeline graph_extraction.
DEFAULT_MENTION_RELATION_KEY = "is_mentioned_in"


@dataclass
class GraphMappings:
    """Ánh xạ entity/relation và liên kết entity → document (chunk)."""

    entity2id: dict[str, int]
    id2entity: dict[str, str]
    relation2id: dict[str, int]
    id2relation: dict[str, str]
    mention_relation_id_base: int | None
    entity_to_documents: dict[str, list[str]]


@dataclass
class GraphBundle:
    data: Data
    mappings: GraphMappings


def _invert_str_int_mapping(d: dict[str, Any]) -> dict[str, str]:
    inverted: dict[str, str] = {}
    for k, v in d.items():
        inverted[str(k)] = str(v) if isinstance(v, (str, int)) else str(v)
    return inverted


def build_entity_to_documents(
    data: Data,
    relation2id: dict[str, int],
    id2node_json: dict[str, Any],
    mention_key: str = DEFAULT_MENTION_RELATION_KEY,
) -> dict[str, list[str]]:
    """
    entity_to_documents: các cạnh entity -> chunk theo relation (mặc định is_mentioned_in).
    Ánh xạ id số -> uid nhờ id2node.json.
    """

    def node_uid(node_id: int) -> str:
        return str(id2node_json.get(str(node_id), str(node_id)))

    if mention_key not in relation2id:
        logger.warning(
            "Không tìm relation %r trong relation2id — bỏ qua entity→document.",
            mention_key,
        )
        return {}

    rid = int(relation2id[mention_key])
    ei = data.target_edge_index.cpu()
    rt = data.target_edge_type.cpu()
    rows = ei[:, rt == rid]
    mentions: dict[str, list[str]] = {}

    for i in range(rows.size(1)):
        h, tail = int(rows[0, i]), int(rows[1, i])
        e_uid = node_uid(h)
        c_uid = node_uid(tail)
        mentions.setdefault(e_uid, []).append(c_uid)

    return {e: sorted(set(docs)) for e, docs in mentions.items()}


def load_graph_bundle(
    tensor_dir: Path,
    *,
    build_relation_graph: bool = False,
    mention_relation_key: str = DEFAULT_MENTION_RELATION_KEY,
    device: torch.device | None = None,
) -> GraphBundle:
    """
    tensor_dir: chứa graph.pt, node2id.json, id2node.json, rel2id.json.
    Inverse edges đã nằm trong graph.pt (sinh bởi build_graph_tensor).
    """

    pt_path = tensor_dir / "graph.pt"
    if not pt_path.is_file():
        raise FileNotFoundError(f"Thiếu graph.pt tại {pt_path}")

    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(data, Data):
        raise TypeError(f"graph.pt phải chứa torch_geometric.data.Data, nhận {type(data)}")

    entity2id = json.loads((tensor_dir / "node2id.json").read_text(encoding="utf-8"))
    id2node = json.loads((tensor_dir / "id2node.json").read_text(encoding="utf-8"))
    relation2id = json.loads((tensor_dir / "rel2id.json").read_text(encoding="utf-8"))

    id2entity = _invert_str_int_mapping(id2node)
    id2relation = {str(v): k for k, v in relation2id.items()}
    relation2id = {str(k): int(v) for k, v in relation2id.items()}
    entity2id = {str(k): int(v) for k, v in entity2id.items()}
    rid_base = relation2id.get(mention_relation_key)

    et_docs = build_entity_to_documents(
        data, relation2id, id2node, mention_key=mention_relation_key
    )

    if build_relation_graph:
        from .tasks import build_relation_graph as brg

        data = brg(data)

    logger.info(
        "Đã load graph: num_nodes=%s num_relations=%s target_edges=%s",
        int(data.num_nodes),
        int(data.num_relations),
        int(data.target_edge_index.size(1)),
    )

    if device is not None:
        data = data.to(device)

    mappings = GraphMappings(
        entity2id=entity2id,
        id2entity=id2entity,
        relation2id=relation2id,
        id2relation=id2relation,
        mention_relation_id_base=rid_base,
        entity_to_documents=et_docs,
    )

    return GraphBundle(data=data, mappings=mappings)


def build_target_to_other_types(
    graph,
    relation2id: dict[str, int],
    mention_key: str = DEFAULT_MENTION_RELATION_KEY,
) -> dict[str, "torch.Tensor"]:
    """
    Build sparse mapping entity→chunk cho GNNRetriever.map_entities_to_docs.

    Trả về dict {"chunk": sparse_coo_tensor(N, N)} trong đó:
    - N = graph.num_nodes (tổng tất cả node types)
    - Rows = global index của entity nodes (head của is_mentioned_in)
    - Cols = global index của chunk nodes (tail của is_mentioned_in)
    - Values = 1.0 per edge

    SimpleRanker: sparse.mm(node_pred [B, N], ent2chunk [N, N]) → (B, N)
    map_entities_to_docs lấy [:, chunk_indices] để điền chunk scores.
    """
    if mention_key not in relation2id:
        logger.warning(
            "Không tìm relation '%s' trong relation2id — target_to_other_types rỗng.",
            mention_key,
        )
        return {}

    rid = int(relation2id[mention_key])
    ei = graph.target_edge_index.cpu()
    rt = graph.target_edge_type.cpu()
    mask = rt == rid
    edges = ei[:, mask]  # shape (2, n_mention_edges); row0=entity, row1=chunk

    if edges.size(1) == 0:
        logger.warning("Không có cạnh '%s' trong graph.", mention_key)
        return {}

    n = int(graph.num_nodes)
    ent_to_chunk = torch.sparse_coo_tensor(
        edges,
        torch.ones(edges.size(1), dtype=torch.float32),
        size=(n, n),
    ).coalesce()

    logger.info(
        "build_target_to_other_types: %d entity→chunk edges, sparse shape=(%d, %d)",
        edges.size(1),
        n,
        n,
    )
    return {"chunk": ent_to_chunk}


def resolve_entity_uid(mappings: GraphMappings, raw: str) -> str | None:
    """Chấp nhận uid đầy đủ hoặc tên gần đúng nếu trùng khoá trong entity2id."""
    if raw in mappings.entity2id:
        return raw
    prefixed = raw if ":" in raw else f"entity:{raw}"
    if prefixed in mappings.entity2id:
        return prefixed
    return None


def resolve_relation_key(mappings: GraphMappings, raw: str) -> str | None:
    """Chọn relation trong relation2id; hỗ trợ có/không prefix inverse_."""
    if raw in mappings.relation2id:
        return raw
    cand = raw.replace("inverse_", "")
    if cand in mappings.relation2id:
        return cand
    return None
