"""Mô hình DistMult cho KGC pretraining (tương thích forward (graph, batch) như KGCTrainer)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


class DistMultKGC(nn.Module):
    """Điểm DistMult: <h, r, t> theo từng chiều embedding."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, graph: Data, batch: torch.Tensor) -> torch.Tensor:
        """
        batch: [B, num_neg+1, 3] with (h, t, r) indices.
        Trả logits [B, num_neg+1].
        """
        _ = graph
        h_idx = batch[..., 0].long().contiguous()
        t_idx = batch[..., 1].long().contiguous()
        r_idx = batch[..., 2].long().contiguous()
        h = self.entity_emb(h_idx)
        t = self.entity_emb(t_idx)
        r = self.relation_emb(r_idx)

        logits = torch.sum(h * r * t, dim=-1)
        return logits

    @torch.no_grad()
    def score_all_tails(
        self,
        graph: Data,
        head_indices: torch.Tensor,
        relation_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Điểm mọi tail cho cụm triple (batch_h, batch_r): [BH, num_nodes]."""
        _ = graph
        h = self.entity_emb(head_indices.long())
        r = self.relation_emb(relation_indices.long())
        hr = (h * r).matmul(self.entity_emb.weight.t())
        return hr

    @torch.no_grad()
    def score_all_heads(
        self,
        graph: Data,
        tail_indices: torch.Tensor,
        relation_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Điểm mọi head cho (batch_t, batch_r): [BT, num_nodes]."""
        _ = graph
        t = self.entity_emb(tail_indices.long())
        r = self.relation_emb(relation_indices.long())
        rt = (r * t).matmul(self.entity_emb.weight.t())
        return rt


def embedding_init_from_features(
    graph: Data,
    entity_emb: nn.Embedding,
    scale: float = 0.02,
) -> None:
    """Khởi tạo embedding entity từ graph.x khi có sẵn (tùy chọn)."""
    if getattr(graph, "x", None) is None:
        return
    x = graph.x.float()
    if x.size(1) != entity_emb.embedding_dim:
        # Không ép chiều để tránh ẩn chứa không xác định
        return
    with torch.no_grad():
        entity_emb.weight.copy_(scale * torch.randn_like(entity_emb.weight) + x)
