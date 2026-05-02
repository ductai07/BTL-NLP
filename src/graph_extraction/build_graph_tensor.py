from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch_geometric.data import Data

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_GRAPH_DIR = Path("data/graph")
DEFAULT_OUTPUT_DIR = Path("data/graph_tensor")


def parse_attributes(value: Any) -> dict[str, Any]:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return {}
    if isinstance(value, dict):
        return value
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def attributes_to_text(attributes: dict[str, Any] | None = None, **kwargs: Any) -> str:
    attributes = attributes or {}
    lines: list[str] = []
    for key, value in kwargs.items():
        if value is not None:
            lines.append(f"{key}: {value}")
    for key, value in attributes.items():
        if value is not None:
            lines.append(f"{key}: {value}")
    return "\n".join(lines).strip()


def read_indexed_csv(path: Path, index_candidates: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required graph CSV: {path}")

    df = pd.read_csv(path, keep_default_na=False)
    df["id"] = df.index
    for column in index_candidates:
        if column in df.columns:
            unique_count = df[column].nunique()
            if unique_count != len(df):
                raise ValueError(
                    f"{path}: column '{column}' must be unique, found "
                    f"{unique_count} unique values for {len(df)} rows"
                )
            df = df.set_index(column)
            break
    else:
        raise ValueError(
            f"{path}: expected one of these identifier columns: {index_candidates}"
        )

    if "attributes" not in df.columns:
        df["attributes"] = [{} for _ in range(len(df))]
    else:
        df["attributes"] = df["attributes"].apply(parse_attributes)
    return df


def encode_texts(
    texts: list[str],
    *,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> torch.Tensor:
    from common.bge_embedder import load_bge_model

    embedder = load_bge_model(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    embeddings = embedder.encode_documents(texts)
    return torch.from_numpy(embeddings).float().cpu()


def build_tensor_graph(
    *,
    graph_dir: Path,
    output_dir: Path,
    embed_features: bool,
    embedding_model: str,
    embedding_device: str | None,
    embedding_batch_size: int,
    inverse_relation_feat: str,
    strict_edges: bool,
) -> dict[str, Any]:
    nodes_file = graph_dir / "nodes.csv"
    relations_file = graph_dir / "relations.csv"
    edges_file = graph_dir / "edges.csv"

    nodes_df = read_indexed_csv(nodes_file, ("uid", "name"))
    relations_df = read_indexed_csv(relations_file, ("name", "relation", "uid"))
    edges_df = pd.read_csv(edges_file, keep_default_na=False)
    required_edge_cols = {"source", "relation", "target"}
    missing_cols = sorted(required_edge_cols - set(edges_df.columns))
    if missing_cols:
        raise ValueError(f"{edges_file}: missing required columns: {missing_cols}")
    if "attributes" not in edges_df.columns:
        edges_df["attributes"] = [{} for _ in range(len(edges_df))]
    else:
        edges_df["attributes"] = edges_df["attributes"].apply(parse_attributes)

    if "type" not in nodes_df.columns:
        nodes_df["type"] = "entity"

    node2id = {str(key): int(value) for key, value in nodes_df["id"].to_dict().items()}
    rel2id = {
        str(key): int(value) for key, value in relations_df["id"].to_dict().items()
    }

    node_type_ids, node_type_names_index = pd.factorize(nodes_df["type"])
    node_type_names = [str(value) for value in node_type_names_index.tolist()]
    node_type = torch.LongTensor(node_type_ids)
    nodes_df["type_id"] = node_type_ids
    nodes_by_type = {
        str(node_type_name): torch.LongTensor(node_ids)
        for node_type_name, node_ids in nodes_df.groupby("type")["id"]
        .apply(list)
        .to_dict()
        .items()
    }

    edges_df["u"] = edges_df["source"].map(node2id)
    edges_df["v"] = edges_df["target"].map(node2id)
    edges_df["r"] = edges_df["relation"].map(rel2id)
    invalid_edges = edges_df[edges_df[["u", "v", "r"]].isnull().any(axis=1)]
    if strict_edges and len(invalid_edges) > 0:
        sample = invalid_edges[["source", "relation", "target"]].head(5).to_dict(
            orient="records"
        )
        raise ValueError(f"Found {len(invalid_edges)} invalid edges, sample={sample}")

    valid_edges_df = edges_df.dropna(subset=["u", "v", "r"]).copy()
    target_edge_index = torch.tensor(
        [
            valid_edges_df["u"].astype(int).to_list(),
            valid_edges_df["v"].astype(int).to_list(),
        ],
        dtype=torch.long,
    )
    target_edge_type = torch.tensor(
        valid_edges_df["r"].astype(int).to_list(),
        dtype=torch.long,
    )

    num_nodes = len(node2id)
    num_base_relations = len(rel2id)
    edge_index = torch.cat([target_edge_index, target_edge_index.flip(0)], dim=1)
    edge_type = torch.cat([target_edge_type, target_edge_type + num_base_relations])

    id2rel = {value: key for key, value in rel2id.items()}
    full_rel2id = dict(rel2id)
    for rel_id, rel_name in id2rel.items():
        full_rel2id[f"inverse_{rel_name}"] = rel_id + num_base_relations

    node_emb = None
    rel_emb = None
    edge_emb = None
    feat_dim = 0
    if embed_features:
        node_texts = nodes_df.apply(
            lambda row: attributes_to_text(
                row["attributes"],
                name=row.name,
                type=row["type"],
            ),
            axis=1,
        ).to_list()
        node_emb = encode_texts(
            node_texts,
            model_name=embedding_model,
            device=embedding_device,
            batch_size=embedding_batch_size,
        )

        rel_texts = relations_df.apply(
            lambda row: attributes_to_text(row["attributes"], name=row.name),
            axis=1,
        ).to_list()
        rel_base_emb = encode_texts(
            rel_texts,
            model_name=embedding_model,
            device=embedding_device,
            batch_size=embedding_batch_size,
        )
        if inverse_relation_feat == "inverse":
            rel_emb = torch.cat([rel_base_emb, -rel_base_emb], dim=0)
        else:
            inverse_rel_texts = relations_df.apply(
                lambda row: attributes_to_text(
                    row["attributes"],
                    name=f"inverse_{row.name}",
                ),
                axis=1,
            ).to_list()
            inverse_rel_emb = encode_texts(
                inverse_rel_texts,
                model_name=embedding_model,
                device=embedding_device,
                batch_size=embedding_batch_size,
            )
            rel_emb = torch.cat([rel_base_emb, inverse_rel_emb], dim=0)
        feat_dim = int(node_emb.size(1))

    graph = Data(
        node_type=node_type,
        node_type_names=node_type_names,
        nodes_by_type=nodes_by_type,
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_nodes,
        target_edge_index=target_edge_index,
        target_edge_type=target_edge_type,
        num_relations=num_base_relations * 2,
        x=node_emb,
        rel_attr=rel_emb,
        edge_attr=edge_emb,
        feat_dim=feat_dim,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output_dir / "graph.pt")
    (output_dir / "node2id.json").write_text(
        json.dumps(node2id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "id2node.json").write_text(
        json.dumps({value: key for key, value in node2id.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "rel2id.json").write_text(
        json.dumps(full_rel2id, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if len(invalid_edges) > 0:
        invalid_edges[["source", "relation", "target", "attributes"]].to_json(
            output_dir / "skipped_edges.jsonl",
            orient="records",
            lines=True,
            force_ascii=False,
        )

    summary = {
        "graph_dir": str(graph_dir).replace("\\", "/"),
        "output_dir": str(output_dir).replace("\\", "/"),
        "num_nodes": num_nodes,
        "node_types": node_type_names,
        "nodes_by_type": {key: int(value.numel()) for key, value in nodes_by_type.items()},
        "num_base_relations": num_base_relations,
        "num_relations_with_inverse": num_base_relations * 2,
        "num_target_edges": int(target_edge_index.size(1)),
        "num_edges_with_inverse": int(edge_index.size(1)),
        "skipped_edges": int(len(invalid_edges)),
        "embed_features": embed_features,
        "feat_dim": feat_dim,
        "files": {
            "graph": "graph.pt",
            "node2id": "node2id.json",
            "id2node": "id2node.json",
            "rel2id": "rel2id.json",
        },
    }
    (output_dir / "tensor_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert graph CSV files into a PyTorch Geometric tensor graph."
    )
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--embed-features",
        action="store_true",
        help="Encode node/relation text attributes into BGE features.",
    )
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-base-en-v1.5",
        help="SentenceTransformer/BGE model used when --embed-features is set.",
    )
    parser.add_argument("--embedding-device", default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument(
        "--inverse-relation-feat",
        choices=("text", "inverse"),
        default="text",
        help="How to create inverse relation features when --embed-features is set.",
    )
    parser.add_argument(
        "--allow-skipped-edges",
        action="store_true",
        help="Skip edges whose source/target/relation is missing instead of failing.",
    )
    args = parser.parse_args()

    summary = build_tensor_graph(
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        embed_features=args.embed_features,
        embedding_model=args.embedding_model,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        inverse_relation_feat=args.inverse_relation_feat,
        strict_edges=not args.allow_skipped_edges,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
