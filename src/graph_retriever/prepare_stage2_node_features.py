"""
Attach node text embeddings to graph.x for Stage 2 SFT distillation.

Run from project root:
  python src/graph_retriever/prepare_stage2_node_features.py \
      --config configs/graph_retriever/stage2_sft.yaml --force
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.common.bge_embedder import load_bge_model

logger = logging.getLogger(__name__)


def _project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (_REPO_ROOT / path).resolve()


def _parse_attributes(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    text = str(value).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {"raw": text}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _first_attr_text(attrs: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = attrs.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _first_row_or_attr_text(row: pd.Series, attrs: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in row and row.get(key) is not None and str(row.get(key)).strip():
            return str(row.get(key)).strip()
    return _first_attr_text(attrs, keys)


def _node_text(row: pd.Series) -> str:
    attrs = _parse_attributes(row.get("attributes"))
    node_type = str(row.get("type") or "").strip()
    name = str(row.get("name") or "").strip()

    if node_type == "chunk":
        text = _first_row_or_attr_text(row, attrs, ("text", "summary", "text_preview"))
        return text or name or str(row.get("uid") or "")

    if node_type == "entity":
        description = _first_row_or_attr_text(row, attrs, ("description",))
        if name and description:
            return f"{name}\n{description}"
        if name:
            return name
        normalized = _first_row_or_attr_text(row, attrs, ("name_normalized",))
        return normalized or str(row.get("uid") or "")

    text = _first_row_or_attr_text(row, attrs, ("text", "summary", "description", "text_preview", "name_normalized"))
    return text or name or str(row.get("uid") or "")


def prepare_node_features(cfg: dict, *, force: bool = False) -> Path:
    graph_cfg = cfg["graph"]
    data_cfg = cfg["data"]

    graph_dir = _project_path(str(graph_cfg.get("graph_dir", "data/graph")))
    tensor_dir = _project_path(str(graph_cfg["tensor_dir"]))
    nodes_csv = graph_dir / "nodes.csv"
    graph_path = tensor_dir / "graph.pt"
    node2id_path = tensor_dir / "node2id.json"
    summary_path = tensor_dir / "stage2_node_features_summary.json"

    if not nodes_csv.is_file():
        raise FileNotFoundError(f"Missing nodes.csv: {nodes_csv}")
    if not graph_path.is_file():
        raise FileNotFoundError(f"Missing graph.pt: {graph_path}")
    if not node2id_path.is_file():
        raise FileNotFoundError(f"Missing node2id.json: {node2id_path}")

    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    if not isinstance(graph, Data):
        raise TypeError(f"graph.pt must contain torch_geometric.data.Data, got {type(graph)}")

    existing_x = getattr(graph, "x", None)
    if existing_x is not None and not force:
        logger.info("graph.x already exists with shape=%s; use --force to rebuild", tuple(existing_x.shape))
        return summary_path

    node2id = {str(k): int(v) for k, v in json.loads(node2id_path.read_text(encoding="utf-8")).items()}
    nodes_df = pd.read_csv(nodes_csv, keep_default_na=False)
    if "uid" not in nodes_df.columns:
        raise ValueError(f"{nodes_csv}: missing required column 'uid'")

    rows_by_uid = {str(row["uid"]): row for _, row in nodes_df.iterrows()}
    missing = sorted(uid for uid in node2id if uid not in rows_by_uid)
    if missing:
        raise ValueError(f"{nodes_csv}: {len(missing)} graph nodes missing from CSV, sample={missing[:5]}")

    ordered_texts = [""] * len(node2id)
    ordered_types = [""] * len(node2id)
    for uid, node_id in node2id.items():
        row = rows_by_uid[uid]
        ordered_texts[node_id] = _node_text(row)
        ordered_types[node_id] = str(row.get("type") or "")

    if len(ordered_texts) != int(graph.num_nodes):
        raise ValueError(
            f"node2id has {len(ordered_texts)} nodes but graph.num_nodes={int(graph.num_nodes)}"
        )

    model_name = str(data_cfg.get("text_emb_model", "BAAI/bge-base-en-v1.5"))
    device = data_cfg.get("emb_device")
    batch_size = int(data_cfg.get("embedding_batch_size", graph_cfg.get("relation_embedding_batch_size", 32)))
    logger.info("Encoding %d node texts with %s", len(ordered_texts), model_name)
    embedder = load_bge_model(model_name=model_name, device=device, batch_size=batch_size)
    graph.x = torch.from_numpy(embedder.encode_documents(ordered_texts)).float().cpu()
    graph.feat_dim = int(graph.x.size(1))

    if int(graph.x.size(0)) != int(graph.num_nodes):
        raise RuntimeError(f"Built graph.x shape={tuple(graph.x.shape)} for graph.num_nodes={graph.num_nodes}")

    torch.save(graph, graph_path)

    node_type_counts: dict[str, int] = {}
    for node_type in ordered_types:
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1

    summary = {
        "graph_path": str(graph_path.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "nodes_csv": str(nodes_csv.relative_to(_REPO_ROOT)).replace("\\", "/"),
        "model_name": model_name,
        "shape": list(graph.x.shape),
        "feat_dim": int(graph.x.size(1)),
        "node_type_counts": node_type_counts,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved graph.x shape=%s to %s", tuple(graph.x.shape), graph_path)
    logger.info("Wrote summary: %s", summary_path)
    return summary_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare graph.x node features for Stage 2 distillation.")
    parser.add_argument("--config", type=Path, default=Path("configs/graph_retriever/stage2_sft.yaml"))
    parser.add_argument("--force", action="store_true", help="Rebuild graph.x even when it already exists.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ns = parse_args(argv)
    config_path = _project_path(ns.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw = OmegaConf.load(config_path)
    cfg_all = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(cfg_all, dict)
    prepare_node_features(cfg_all, force=ns.force)


if __name__ == "__main__":
    main()
