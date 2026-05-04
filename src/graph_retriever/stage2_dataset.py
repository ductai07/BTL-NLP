"""
Stage 2 Dataset Adapter cho GFM-RAG SFT fine-tuning.

Tạo đối tượng tương thích với SFTTrainer._create_task_dataset, yêu cầu:
  sft_data.train_data  → torch Dataset trả về {id, question_embeddings, start_nodes_mask, target_nodes_mask}
  sft_data.test_data   → tương tự
  sft_data.graph       → PyG Data object (đã có target_to_other_types, rel_attr, nodes_by_type, ...)
  sft_data.id2node     → {int_node_id → str_node_name} cho predict()
  sft_data.raw_test_data → list[dict] gốc cho predict()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Torch Dataset
# ─────────────────────────────────────────────────────────────

class Stage2TorchDataset(Dataset):
    """
    Dataset trả về batch dict tương thích SFTTrainer.

    Mỗi item:
        id                  : int (integer index trong dataset)
        question_embeddings : Tensor[D]   float32, D = feat_dim của text model (768 với BGE-base)
        start_nodes_mask    : Tensor[N]   float32, 1 tại entity nodes trong start_nodes
        target_nodes_mask   : Tensor[N]   float32, 1 tại chunk+entity nodes trong target_nodes
    """

    def __init__(
        self,
        samples: list[dict],
        question_embeddings: torch.Tensor,
        num_nodes: int,
        node2id: dict[str, int],
    ) -> None:
        """
        Args:
            samples: list sample đã map (từ dev_qa_stage2.json)
            question_embeddings: Tensor[n_samples, D] float32
            num_nodes: tổng số node trong graph (N = 17262)
            node2id: {node_uid → global_node_idx}
        """
        assert len(samples) == len(question_embeddings), (
            f"Mismatch samples ({len(samples)}) vs question_embeddings ({len(question_embeddings)})"
        )
        self.samples = samples
        self.question_embeddings = question_embeddings.float()
        self.num_nodes = num_nodes
        self.node2id = node2id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        # start_nodes_mask: entities trong start_nodes.entity
        start_mask = torch.zeros(self.num_nodes, dtype=torch.float32)
        for uid in sample["start_nodes"].get("entity", []):
            node_idx = self.node2id.get(uid)
            if node_idx is not None:
                start_mask[node_idx] = 1.0

        # target_nodes_mask: chunk + entity trong target_nodes
        target_mask = torch.zeros(self.num_nodes, dtype=torch.float32)
        for uid in sample["target_nodes"].get("chunk", []):
            node_idx = self.node2id.get(uid)
            if node_idx is not None:
                target_mask[node_idx] = 1.0
        for uid in sample["target_nodes"].get("entity", []):
            node_idx = self.node2id.get(uid)
            if node_idx is not None:
                target_mask[node_idx] = 1.0

        return {
            "id": torch.tensor(idx, dtype=torch.long),
            "question_embeddings": self.question_embeddings[idx],
            "start_nodes_mask": start_mask,
            "target_nodes_mask": target_mask,
        }


# ─────────────────────────────────────────────────────────────
# SFT Data Container
# ─────────────────────────────────────────────────────────────

@dataclass
class Stage2SFTData:
    """
    Container tương thích với SFTTrainer._create_task_dataset.

    Attributes:
        train_data     : Stage2TorchDataset (80% samples)
        test_data      : Stage2TorchDataset (20% samples)
        graph          : PyG Data (đã có target_to_other_types, rel_attr, nodes_by_type)
        id2node        : {int_node_id → str_node_name} cho predict()
        raw_test_data  : list[dict] gốc cho predict()
    """
    train_data: Stage2TorchDataset
    test_data: Stage2TorchDataset
    graph: Data
    id2node: dict[int, str]
    raw_test_data: list[dict]


# ─────────────────────────────────────────────────────────────
# Loader adapter tương thích với GraphDatasetLoader interface
# ─────────────────────────────────────────────────────────────

class _SingleSFTDatasetLoader:
    """
    Simple loader yield một GraphDataset duy nhất.
    Tương thích với SFTTrainer(train_graph_dataset_loader=...).
    """

    def __init__(self, name: str, sft_data: Stage2SFTData) -> None:
        self._name = name
        self._sft_data = sft_data

    def set_epoch(self, epoch: int) -> None:
        return

    def shutdown(self) -> None:
        return

    def __iter__(self):
        from gfmrag.graph_index_datasets.graph_dataset_loader import GraphDataset
        yield GraphDataset(name=self._name, data=self._sft_data)


# ─────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────

def load_stage2_sft_data(
    stage2_json: Path,
    graph: Data,
    node2id: dict[str, int],
    id2node_raw: dict[str, str],
    text_emb_model_name: str = "BAAI/bge-base-en-v1.5",
    emb_device: str | None = None,
    emb_cache: Path | None = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Stage2SFTData:
    """
    Load Stage 2 JSON, encode questions, tạo Stage2SFTData.

    Args:
        stage2_json   : đường dẫn tới dev_qa_stage2.json
        graph         : PyG Data object (đã có target_to_other_types, nodes_by_type, rel_attr)
        node2id       : {node_uid → global_node_idx}
        id2node_raw   : {str(global_node_idx) → node_uid}
        text_emb_model_name : tên SentenceTransformer model
        emb_device    : device cho encoding ('cuda', 'cpu', None=auto)
        emb_cache     : nếu không None, cache embeddings tại path này
        train_ratio   : tỷ lệ train split
        seed          : random seed cho split
    """
    # Load samples
    with open(stage2_json, encoding="utf-8") as f:
        samples: list[dict] = json.load(f)
    logger.info("Load stage2 data: %d samples từ %s", len(samples), stage2_json)

    # Encode questions
    questions = [s["question"] for s in samples]
    if emb_cache is not None and Path(emb_cache).exists():
        logger.info("Load question embeddings từ cache: %s", emb_cache)
        question_embs = torch.load(emb_cache, map_location="cpu", weights_only=False)
    else:
        logger.info("Encode %d questions bằng %s ...", len(questions), text_emb_model_name)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(text_emb_model_name, device=emb_device)
        embs = model.encode(questions, batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        question_embs = torch.tensor(embs, dtype=torch.float32)
        if emb_cache is not None:
            Path(emb_cache).parent.mkdir(parents=True, exist_ok=True)
            torch.save(question_embs, emb_cache)
            logger.info("Lưu question embeddings: %s", emb_cache)

    num_nodes = int(graph.num_nodes)

    # Train/val split
    torch.manual_seed(seed)
    n = len(samples)
    n_train = max(1, int(n * train_ratio))
    perm = torch.randperm(n).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_samples = [samples[i] for i in train_idx]
    train_embs = question_embs[train_idx]
    val_samples = [samples[i] for i in val_idx]
    val_embs = question_embs[val_idx]

    train_ds = Stage2TorchDataset(train_samples, train_embs, num_nodes, node2id)
    val_ds = Stage2TorchDataset(val_samples, val_embs, num_nodes, node2id)

    logger.info("Train: %d samples | Val: %d samples", len(train_ds), len(val_ds))

    # id2node cho predict()
    id2node_int = {int(k): v for k, v in id2node_raw.items()}

    # Sanity: log mapping stats
    start_ok = sum(1 for s in samples if s["start_nodes"].get("entity"))
    chunk_ok = sum(1 for s in samples if s["target_nodes"].get("chunk"))
    ent_ok = sum(1 for s in samples if s["target_nodes"].get("entity"))
    logger.info(
        "Mapping stats: start_entity=%d/%.0f%% | target_chunk=%d/%.0f%% | target_entity=%d/%.0f%%",
        start_ok, 100 * start_ok / max(1, n),
        chunk_ok, 100 * chunk_ok / max(1, n),
        ent_ok, 100 * ent_ok / max(1, n),
    )

    return Stage2SFTData(
        train_data=train_ds,
        test_data=val_ds,
        graph=graph,
        id2node=id2node_int,
        raw_test_data=val_samples,
    )
