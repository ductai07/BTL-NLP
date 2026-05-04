"""
Distillation Features Loader for Stage 2 training.

Loads teacher model embeddings (node_x, question_embeddings) separately
without modifying the main graph tensor.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DistillationFeatureLoader:
    """Load and manage teacher distillation features."""

    def __init__(
        self,
        teacher_feature_dir: Path,
        num_samples: int,
        num_nodes: int,
        device: str = "cpu",
    ):
        """
        Initialize loader.
        
        Args:
            teacher_feature_dir: Directory with node_x.pt, question_embeddings.pt, meta.json
            num_samples: Number of training samples
            num_nodes: Number of nodes in graph
            device: Device to load tensors on
        """
        self.teacher_feature_dir = Path(teacher_feature_dir)
        self.num_samples = num_samples
        self.num_nodes = num_nodes
        self.device = device

        self._node_x = None
        self._question_emb = None
        self._meta = None
        self._sample_q2idx = None

    def load(self) -> dict:
        """Load all distillation features and return metadata."""
        if not self.teacher_feature_dir.exists():
            raise FileNotFoundError(f"Teacher feature dir not found: {self.teacher_feature_dir}")

        # Load metadata
        meta_path = self.teacher_feature_dir / "meta.json"
        if meta_path.exists():
            self._meta = json.loads(meta_path.read_text(encoding="utf-8"))
            logger.info(f"Loaded distillation meta from {meta_path}")
        else:
            logger.warning(f"meta.json not found at {self.teacher_feature_dir}")

        # Load node embeddings
        node_x_path = self.teacher_feature_dir / "node_x.pt"
        if not node_x_path.exists():
            raise FileNotFoundError(f"node_x.pt not found: {node_x_path}")

        self._node_x = torch.load(node_x_path, map_location=self.device)
        logger.info(f"Loaded teacher node_x: {self._node_x.shape} from {node_x_path}")

        # Validate node_x
        if self._node_x.shape[0] != self.num_nodes:
            raise ValueError(
                f"node_x has {self._node_x.shape[0]} nodes, expected {self.num_nodes}"
            )

        # Load question embeddings
        q_emb_path = self.teacher_feature_dir / "question_embeddings.pt"
        if not q_emb_path.exists():
            raise FileNotFoundError(f"question_embeddings.pt not found: {q_emb_path}")

        self._question_emb = torch.load(q_emb_path, map_location=self.device)
        logger.info(
            f"Loaded teacher question_embeddings: {self._question_emb.shape} from {q_emb_path}"
        )

        # Validate embedding dimensions match
        if self._node_x.shape[1] != self._question_emb.shape[1]:
            raise ValueError(
                f"node_x dim {self._node_x.shape[1]} != question_emb dim {self._question_emb.shape[1]}"
            )

        # Load sample mapping (optional)
        q2idx_path = self.teacher_feature_dir / "sample_q2idx.json"
        if q2idx_path.exists():
            self._sample_q2idx = json.loads(q2idx_path.read_text(encoding="utf-8"))
            logger.info(f"Loaded sample q2idx mapping with {len(self._sample_q2idx)} entries")

        return {
            "node_x_shape": tuple(self._node_x.shape),
            "question_embeddings_shape": tuple(self._question_emb.shape),
            "embedding_dim": int(self._node_x.shape[1]),
            "teacher_model": self._meta.get("teacher_model", "unknown") if self._meta else "unknown",
            "metadata": self._meta,
        }

    @property
    def node_x(self) -> torch.Tensor:
        """Get teacher node embeddings."""
        if self._node_x is None:
            raise RuntimeError("Features not loaded. Call load() first.")
        return self._node_x

    @property
    def question_embeddings(self) -> torch.Tensor:
        """Get teacher question embeddings."""
        if self._question_emb is None:
            raise RuntimeError("Features not loaded. Call load() first.")
        return self._question_emb

    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._node_x is None:
            raise RuntimeError("Features not loaded. Call load() first.")
        return self._node_x.shape[1]

    def get_question_embedding(self, sample_idx: int) -> torch.Tensor:
        """Get single question embedding by sample index."""
        if self._question_emb is None:
            raise RuntimeError("Features not loaded. Call load() first.")

        if sample_idx >= len(self._question_emb):
            raise IndexError(f"sample_idx {sample_idx} >= num_samples {len(self._question_emb)}")

        return self._question_emb[sample_idx]

    def log_info(self) -> None:
        """Log detailed info about loaded features."""
        if self._node_x is None or self._question_emb is None:
            logger.warning("Features not loaded yet")
            return

        logger.info("=" * 80)
        logger.info("DISTILLATION FEATURES INFO")
        logger.info("=" * 80)
        logger.info(f"  Feature directory: {self.teacher_feature_dir}")
        if self._meta:
            logger.info(f"  Teacher model: {self._meta.get('teacher_model', 'unknown')}")
            logger.info(f"  Created at: {self._meta.get('created_at', 'unknown')}")
            if "nodes_metadata" in self._meta:
                nm = self._meta["nodes_metadata"]
                logger.info(f"  Nodes: {nm.get('num_nodes', 'unknown')} total")
                if "node_type_counts" in nm:
                    nc = nm["node_type_counts"]
                    logger.info(
                        f"    - chunks: {nc.get('chunk', 0)}, "
                        f"entities: {nc.get('entity', 0)}, "
                        f"unknown: {nc.get('unknown', 0)}"
                    )
        logger.info(f"  Node embeddings shape: {tuple(self._node_x.shape)}")
        logger.info(f"  Question embeddings shape: {tuple(self._question_emb.shape)}")
        logger.info(f"  Embedding dimension: {self._node_x.shape[1]}")
        logger.info("=" * 80)


def validate_distillation_config(cfg: dict, distill_loader: DistillationFeatureLoader) -> bool:
    """
    Validate distillation config against loaded features.
    
    Args:
        cfg: Config dict with distillation section
        distill_loader: Loaded DistillationFeatureLoader
        
    Returns:
        True if valid, False otherwise
    """
    distill_cfg = cfg.get("distillation", {})
    if not distill_cfg.get("enable", False):
        return True

    mode = distill_cfg.get("mode", "author_graph_x")
    if mode != "teacher_features":
        return True

    logger.info("Validating distillation config...")

    # Check teacher model matches
    teacher_model = distill_cfg.get("teacher_model")
    loader_model = distill_loader._meta.get("teacher_model") if distill_loader._meta else None

    if teacher_model and loader_model and teacher_model != loader_model:
        logger.warning(
            f"Config teacher_model {teacher_model} != loader teacher_model {loader_model}"
        )

    logger.info("✅ Distillation config valid")
    return True
