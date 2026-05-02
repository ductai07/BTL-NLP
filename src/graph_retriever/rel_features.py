from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


def _load_rel2id(rel2id_path: Path) -> dict[str, int]:
    raw = json.loads(rel2id_path.read_text(encoding="utf-8"))
    return {str(k): int(v) for k, v in raw.items()}


def build_relation_texts(rel2id: dict[str, int]) -> list[str]:
    size = max(rel2id.values()) + 1 if rel2id else 0
    texts = [""] * size
    for name, rid in rel2id.items():
        if 0 <= rid < size:
            texts[rid] = str(name)
    # fallback an toàn nếu có id trống
    for i, t in enumerate(texts):
        if not t:
            texts[i] = f"rel_{i}"
    return texts


def encode_texts_bge(
    texts: list[str],
    *,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> torch.Tensor:
    """
    Dùng embedder sẵn có của project (`src/common/bge_embedder.py`) để tạo embedding.
    Trả tensor CPU float32 shape [N, dim].
    """
    from src.common.bge_embedder import load_bge_model

    embedder = load_bge_model(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    emb = embedder.encode_documents(texts)
    return torch.from_numpy(emb).float().cpu()


def ensure_rel_attr(
    graph: Data,
    *,
    rel2id_path: Path,
    embedding_model: str,
    embedding_device: str | None = None,
    embedding_batch_size: int = 32,
    force: bool = False,
) -> tuple[Data, int]:
    """
    Đảm bảo graph có `rel_attr` đúng kích thước theo rel2id.
    Vì QueryGNN dùng graph.rel_attr để tạo query embedding.
    """
    if getattr(graph, "rel_attr", None) is not None and not force:
        rel_attr = graph.rel_attr
        if isinstance(rel_attr, torch.Tensor) and rel_attr.dim() == 2:
            return graph, int(rel_attr.size(1))

    rel2id = _load_rel2id(rel2id_path)
    texts = build_relation_texts(rel2id)
    rel_attr = encode_texts_bge(
        texts,
        model_name=embedding_model,
        device=embedding_device,
        batch_size=embedding_batch_size,
    )

    graph.rel_attr = rel_attr
    graph.feat_dim = int(rel_attr.size(1))
    logger.info("Đã tạo rel_attr: shape=%s feat_dim=%s", tuple(rel_attr.shape), graph.feat_dim)
    return graph, int(rel_attr.size(1))

