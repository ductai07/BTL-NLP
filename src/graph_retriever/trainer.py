"""Huấn luyện KGC một GPU, tái hiện bước train/eval của KGCTrainer (BCE + negative sampling strict)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from tqdm import tqdm

from . import tasks
from .graph_adapter import GraphBundle

logger = logging.getLogger(__name__)


def make_triples_dataloader(
    triples: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool,
    drop_last: bool,
    device: torch.device,
) -> DataLoader:
    """triples: tensor [num_edges, 3] (h, t, r_index)."""

    triples_cpu = triples.detach().clone().cpu()
    ds = TensorDataset(triples_cpu)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        pin_memory=device.type == "cuda",
    )


class LocalKGCTrainer:
    def __init__(
        self,
        model: Module,
        graph_bundle: GraphBundle,
        *,
        lr: float,
        device: torch.device,
        epochs: int,
        train_batch_size: int,
        eval_batch_size: int,
        num_negative: int,
        strict_negative: bool,
        adversarial_temperature: float,
        fast_eval_queries: int | None,
        output_dir: Path,
        log_every_steps: int,
        eval_every_epoch: bool,
        metrics: list[str] | None,
    ) -> None:
        self.model = model
        self.bundle = graph_bundle
        self.device = device
        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_negative = num_negative
        self.strict_negative = strict_negative
        self.adversarial_temperature = adversarial_temperature
        self.fast_eval_queries = fast_eval_queries
        self.output_dir = output_dir
        self.log_every_steps = log_every_steps
        self.eval_every_epoch = eval_every_epoch
        self.metrics = metrics or ["mr", "mrr", "hits@1", "hits@3", "hits@10"]

        self.graph = graph_bundle.data.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        ei = graph_bundle.data.target_edge_index
        et = graph_bundle.data.target_edge_type
        train_triples = torch.cat([ei, et.unsqueeze(0)], dim=0).t()

        drop_last_train = train_triples.size(0) > train_batch_size
        if train_batch_size % 2 != 0:
            raise ValueError(
                "train_batch_size phải chẵn (chia nửa corrupt tail vs head)."
            )
        self._train_loader = make_triples_dataloader(
            train_triples,
            train_batch_size,
            shuffle=True,
            drop_last=drop_last_train,
            device=device,
        )

    def train_step(self, batch_flat: tuple[torch.Tensor, ...]) -> dict[str, float]:
        triples_batch = batch_flat[0].to(self.device)

        corrupted = tasks.negative_sampling(
            self.graph,
            triples_batch,
            self.num_negative,
            strict=self.strict_negative,
        )
        logits = self.model(self.graph, corrupted)

        targets = torch.zeros_like(logits)
        targets[:, 0] = 1.0

        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        neg_weight = torch.ones_like(logits)
        if self.adversarial_temperature > 0:
            with torch.no_grad():
                neg_weight[:, 1:] = F.softmax(
                    logits[:, 1:] / self.adversarial_temperature, dim=-1
                )
        else:
            neg_weight[:, 1:] = 1.0 / self.num_negative

        loss = (loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)
        loss = loss.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return {"loss": float(loss.item())}

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        self.model.eval()

        ei = self.bundle.data.target_edge_index.cpu()
        et = self.bundle.data.target_edge_type.cpu()

        triples_full = torch.cat([ei, et.unsqueeze(0)], dim=0).t()
        if self.fast_eval_queries is not None and self.fast_eval_queries < triples_full.size(0):
            mask = torch.randperm(triples_full.size(0))[: self.fast_eval_queries]
            triples_eval = triples_full[mask]
            logger.info("Eval nhanh: %s / %s triple.", mask.numel(), triples_full.size(0))
        else:
            triples_eval = triples_full

        loader = make_triples_dataloader(
            triples_eval,
            self.eval_batch_size,
            shuffle=False,
            drop_last=False,
            device=self.device,
        )

        val_filtered_graph = Data(
            edge_index=self.graph.target_edge_index,
            edge_type=self.graph.target_edge_type,
            num_nodes=int(self.graph.num_nodes),
        ).to(self.device)

        rankings: list[torch.Tensor] = []

        graph_for_eval = self.graph

        for (batch_flat,) in tqdm(loader, desc="Eval KGC", leave=False):
            batch = batch_flat.to(self.device)
            t_batch, h_batch = tasks.all_negative(graph_for_eval, batch)
            t_pred = self.model(graph_for_eval, t_batch)
            h_pred = self.model(graph_for_eval, h_batch)

            t_mask, h_mask = tasks.strict_negative_mask(val_filtered_graph, batch)

            pos_h_index, pos_t_index, pos_r_index = batch.t()

            _ = pos_r_index
            t_ranking = tasks.compute_ranking(t_pred, pos_t_index, t_mask)
            h_ranking = tasks.compute_ranking(h_pred, pos_h_index, h_mask)

            rankings += [t_ranking, h_ranking]

        ranking = torch.cat(rankings)

        metrics: dict[str, float] = {}
        for metric in self.metrics:
            if "-tail" in metric:
                continue
            metric_name = metric
            tensor_rank = ranking.float()

            if metric_name == "mr":
                metrics[metric] = tensor_rank.mean().item()
            elif metric_name == "mrr":
                metrics[metric] = (1 / tensor_rank).mean().item()
            elif metric_name.startswith("hits@"):
                thresh = int(metric_name[5:].split("_")[0])
                metrics[metric] = (tensor_rank <= thresh).float().mean().item()
            elif metric_name == "mr-tail":
                pass

        self.model.train()
        return metrics

    def train(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.train()
        step = 0

        best_mrr = -1.0
        ckpt_path = self.output_dir / "kgc_checkpoint.pt"

        for epoch in range(self.epochs):
            agg_loss = 0.0
            n_batches = 0

            for batch_flat in self._train_loader:
                if batch_flat[0].size(0) % 2 != 0:
                    continue

                outs = self.train_step(batch_flat)
                agg_loss += outs["loss"]
                n_batches += 1
                step += 1

                if step % self.log_every_steps == 0:
                    logger.info("epoch=%s step=%s loss=%.6f", epoch, step, outs["loss"])

            mean_loss = agg_loss / max(n_batches, 1)
            logger.info(
                "epoch=%s kết thúc — loss trung bình=%.6f (batches=%s)",
                epoch,
                mean_loss,
                n_batches,
            )

            if self.eval_every_epoch:
                metrics = self.evaluate()
                logger.info(
                    "epoch=%s eval: %s",
                    epoch,
                    ", ".join(f"{k}={v:.6f}" for k, v in sorted(metrics.items())),
                )
                mrr = metrics.get("mrr", 0.0)
                if mrr > best_mrr:
                    best_mrr = mrr
                    self._save_checkpoint(ckpt_path, extra={"metrics": metrics, "epoch": epoch})

        if not self.eval_every_epoch:
            self._save_checkpoint(ckpt_path, extra={"epoch": self.epochs - 1})

        logger.info("Checkpoint — %s (best_mrr trong eval=%s)", ckpt_path, best_mrr)
        return ckpt_path

    def _save_checkpoint(self, path: Path, extra: dict[str, Any] | None = None) -> None:
        from .kgc_model import DistMultKGC

        if not isinstance(self.model, DistMultKGC):
            raise TypeError("Chỉ tự serialize DistMultKGC; model khác vui lòng tự lưu.")

        payload = {
            "model_cls": "DistMultKGC",
            "state_dict": self.model.state_dict(),
            "meta": {
                "num_entities": self.model.num_entities,
                "num_relations": self.model.num_relations,
                "embedding_dim": self.model.embedding_dim,
            },
            "extra": extra or {},
        }
        torch.save(payload, path)
        logger.info("Đã ghi checkpoint: %s", path)
