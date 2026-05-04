"""
Sanity gate for Stage 2 GFM-RAG SFT.

Runs hard-label-only first, then hard-label + distillation on the same
first 20 samples. Full training should not proceed if distillation is
non-finite or degrades chunk retrieval sharply.

Run from project root:
  python src/graph_retriever/sanity_stage2.py \
      --config configs/graph_retriever/stage2_sft.yaml \
      --sanity-steps 300
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.graph_retriever.graph_adapter import build_target_to_other_types, load_graph_bundle
from src.graph_retriever.gfm_bootstrap import bootstrap_gfmrag, disable_custom_rspmm
from src.graph_retriever.rel_features import ensure_rel_attr
from src.graph_retriever.stage2_common import (
    build_stage2_loss_functions,
    log_stage2_runtime_context,
    reject_external_gfmrag_path,
    validate_graph_x,
)
from src.graph_retriever.stage2_dataset import Stage2TorchDataset

logger = logging.getLogger(__name__)

SANITY_N = 20
SANITY_STEPS = 300


def _compute_mrr(pred: torch.Tensor, target: torch.Tensor) -> float:
    if target.sum() == 0:
        return 0.0
    ranks: list[float] = []
    for i in range(pred.size(0)):
        pos_ids = target[i].bool().nonzero(as_tuple=False).squeeze(1)
        if len(pos_ids) == 0:
            continue
        sorted_ids = torch.argsort(pred[i], descending=True)
        for pos_id in pos_ids:
            rank = (sorted_ids == pos_id).nonzero(as_tuple=False).item() + 1
            ranks.append(1.0 / rank)
    return float(sum(ranks) / len(ranks)) if ranks else 0.0


def _compute_recall_at(pred: torch.Tensor, target: torch.Tensor, k: int) -> float:
    scores: list[float] = []
    for i in range(pred.size(0)):
        pos_ids = target[i].bool().nonzero(as_tuple=False).squeeze(1)
        if len(pos_ids) == 0:
            continue
        topk = torch.topk(pred[i], k=min(k, pred.size(1))).indices
        hits = torch.isin(pos_ids, topk).sum().item()
        scores.append(float(hits) / float(len(pos_ids)))
    return float(sum(scores) / len(scores)) if scores else 0.0


def _load_question_embeddings(cfg: dict, samples: list[dict]) -> torch.Tensor:
    data_cfg = cfg["data"]
    emb_cache = Path(str(data_cfg["emb_cache"])) if data_cfg.get("emb_cache") else None
    if emb_cache is not None and emb_cache.exists():
        all_question_embs = torch.load(emb_cache, map_location="cpu", weights_only=False)
        if all_question_embs.size(0) < len(samples):
            raise ValueError(
                f"Question embedding cache has {all_question_embs.size(0)} rows but sanity needs {len(samples)}"
            )
        question_embs = all_question_embs[: len(samples)].float()
        logger.info("Load %d sanity question embeddings from cache: %s", len(question_embs), emb_cache)
        return question_embs

    from sentence_transformers import SentenceTransformer

    text_model_name = str(data_cfg.get("text_emb_model", "BAAI/bge-base-en-v1.5"))
    logger.info("Encode %d sanity questions with %s", len(samples), text_model_name)
    text_model = SentenceTransformer(text_model_name, device=data_cfg.get("emb_device"))
    embs = text_model.encode(
        [sample["question"] for sample in samples],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return torch.tensor(embs, dtype=torch.float32)


def _load_model(cfg: dict, feat_dim: int, device: torch.device, seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    from gfmrag.models.gfm_rag_v1 import model as gfm_model_module
    from gfmrag.models.gfm_rag_v1.rankers import SimpleRanker

    entity_model = instantiate(OmegaConf.create(cfg["model"]["entity_model"]))
    model_cfg = cfg["model"]
    model = gfm_model_module.GNNRetriever(
        entity_model=entity_model,
        feat_dim=int(feat_dim),
        ranker=SimpleRanker(),
        init_nodes_weight=bool(model_cfg.get("init_nodes_weight", True)),
        init_nodes_type=str(model_cfg.get("init_nodes_type", "chunk")),
    ).to(device)

    pretrained_path = cfg["training"].get("pretrained_model_path")
    if pretrained_path:
        pre_path = Path(str(pretrained_path))
        if not pre_path.is_absolute():
            pre_path = (_REPO_ROOT / pre_path).resolve()
        if pre_path.exists():
            payload = torch.load(pre_path, map_location="cpu", weights_only=False)
            state = payload.get("model") if isinstance(payload, dict) else payload
            missing, unexpected = model.load_state_dict(state, strict=False)
            logger.info("Pretrained load OK: missing=%d unexpected=%d", len(missing), len(unexpected))
        else:
            logger.warning("Pretrained checkpoint not found for sanity: %s", pre_path)

    return model


@torch.no_grad()
def _eval_model(
    model: torch.nn.Module,
    graph: Any,
    loader: DataLoader,
    chunk_nodes: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    all_chunk_pred: list[torch.Tensor] = []
    all_chunk_tgt: list[torch.Tensor] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        pred = model(graph, batch)
        tgt = batch["target_nodes_mask"]
        all_chunk_pred.append(pred[:, chunk_nodes].cpu())
        all_chunk_tgt.append(tgt[:, chunk_nodes].cpu())

    pred_t = torch.cat(all_chunk_pred)
    tgt_t = torch.cat(all_chunk_tgt)
    return {
        "chunk_mrr": _compute_mrr(pred_t, tgt_t),
        "chunk_recall@5": _compute_recall_at(pred_t, tgt_t, 5),
        "chunk_recall@10": _compute_recall_at(pred_t, tgt_t, 10),
        "chunk_recall@20": _compute_recall_at(pred_t, tgt_t, 20),
    }


def _compute_losses(
    *,
    loss_functions: list[Any],
    pred: torch.Tensor,
    target: torch.Tensor,
    graph: Any,
    batch: dict[str, torch.Tensor],
    require_finite_distill: bool,
) -> tuple[torch.Tensor, float, float, dict[str, float]]:
    total_loss = pred.sum() * 0.0
    hard_loss = 0.0
    distill_loss = 0.0
    metrics: dict[str, float] = {}
    distillation_targets: dict[str, torch.Tensor] = {}

    for loss in loss_functions:
        target_node_type = str(loss.target_node_type)
        target_node_ids = graph.nodes_by_type[target_node_type]
        target_node_pred = pred[:, target_node_ids]
        target_node_label = target[:, target_node_ids]

        if loss.is_distillation_loss:
            if target_node_type not in distillation_targets:
                target_node_emb = graph.x[target_node_ids]
                distillation_targets[target_node_type] = batch["question_embeddings"] @ target_node_emb.T
            single_loss = loss.loss_fn(target_node_pred, distillation_targets[target_node_type])
            if require_finite_distill and not torch.isfinite(single_loss):
                raise AssertionError(f"Distillation loss {loss.name} is non-finite: {single_loss.item()}")
        elif target_node_label.sum() == 0:
            single_loss = target_node_pred.sum() * 0.0
        else:
            single_loss = loss.loss_fn(target_node_pred, target_node_label)

        if not torch.isfinite(single_loss):
            raise AssertionError(f"Loss {loss.name} is non-finite: {single_loss.item()}")

        weighted = float(loss.weight) * single_loss
        total_loss = total_loss + weighted
        value_f = float(single_loss.detach().item())
        metrics[str(loss.name)] = value_f
        if loss.is_distillation_loss:
            distill_loss += float(loss.weight) * value_f
        else:
            hard_loss += float(loss.weight) * value_f

    if not torch.isfinite(total_loss):
        raise AssertionError(f"Total loss is non-finite: {total_loss.item()}")
    return total_loss, hard_loss, distill_loss, metrics


def _run_profile(
    *,
    profile_name: str,
    cfg: dict,
    graph: Any,
    feat_dim: int,
    loader: DataLoader,
    loss_functions: list[Any],
    chunk_nodes: torch.Tensor,
    device: torch.device,
    sanity_steps: int,
    require_finite_distill: bool,
) -> dict[str, float]:
    seed = int(cfg.get("data", {}).get("seed", 42))
    model = _load_model(cfg, feat_dim, device, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    logger.info("=== SANITY %s STEP 0 EVAL ===", profile_name)
    step0_metrics = _eval_model(model, graph, loader, chunk_nodes, device)
    logger.info(
        "  chunk_mrr=%.4f | recall@5=%.4f | recall@10=%.4f | recall@20=%.4f",
        step0_metrics["chunk_mrr"],
        step0_metrics["chunk_recall@5"],
        step0_metrics["chunk_recall@10"],
        step0_metrics["chunk_recall@20"],
    )

    model.train()
    step = 0
    recent_total: list[float] = []
    recent_hard: list[float] = []
    recent_distill: list[float] = []

    while step < sanity_steps:
        for batch in loader:
            if step >= sanity_steps:
                break
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            pred = model(graph, batch)
            total_loss, hard_loss, distill_loss, _ = _compute_losses(
                loss_functions=loss_functions,
                pred=pred,
                target=batch["target_nodes_mask"],
                graph=graph,
                batch=batch,
                require_finite_distill=require_finite_distill,
            )
            total_loss.backward()
            optimizer.step()

            recent_total.append(float(total_loss.detach().item()))
            recent_hard.append(hard_loss)
            recent_distill.append(distill_loss)
            step += 1

            if step % 50 == 0:
                logger.info(
                    "  %s step=%d | hard_loss=%.4f | distill_loss=%.4f | total_loss=%.4f",
                    profile_name,
                    step,
                    sum(recent_hard[-50:]) / len(recent_hard[-50:]),
                    sum(recent_distill[-50:]) / len(recent_distill[-50:]),
                    sum(recent_total[-50:]) / len(recent_total[-50:]),
                )

    final_metrics = _eval_model(model, graph, loader, chunk_nodes, device)
    logger.info("=== SANITY %s FINAL EVAL (step %d) ===", profile_name, step)
    logger.info(
        "  hard_loss=%.4f | distill_loss=%.4f | total_loss=%.4f",
        sum(recent_hard[-50:]) / max(1, len(recent_hard[-50:])),
        sum(recent_distill[-50:]) / max(1, len(recent_distill[-50:])),
        sum(recent_total[-50:]) / max(1, len(recent_total[-50:])),
    )
    logger.info(
        "  chunk_mrr=%.4f | recall@5=%.4f | recall@10=%.4f | recall@20=%.4f",
        final_metrics["chunk_mrr"],
        final_metrics["chunk_recall@5"],
        final_metrics["chunk_recall@10"],
        final_metrics["chunk_recall@20"],
    )
    return final_metrics


def run_sanity(
    cfg: dict,
    sanity_steps: int = SANITY_STEPS,
    *,
    already_bootstrapped: bool = False,
) -> None:
    if not already_bootstrapped:
        reject_external_gfmrag_path(cfg.get("gfmrag_path"))
        bootstrap_gfmrag(cfg.get("gfmrag_path"))
        if bool(cfg.get("disable_custom_rspmm", True)):
            disable_custom_rspmm()

    device_spec = str(cfg["training"].get("device", "cuda"))
    device = torch.device(device_spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA unavailable; using CPU for sanity.")
        device = torch.device("cpu")
    logger.info("Sanity device: %s", device)

    tensor_dir = Path(str(cfg["graph"]["tensor_dir"]))
    bundle = load_graph_bundle(
        tensor_dir,
        mention_relation_key=str(cfg["graph"].get("mention_relation_key", "is_mentioned_in")),
    )
    graph = bundle.data
    validate_graph_x(graph, context="Stage 2 sanity graph")
    graph, feat_dim = ensure_rel_attr(
        graph,
        rel2id_path=tensor_dir / "rel2id.json",
        embedding_model=str(cfg["graph"]["relation_embedding_model"]),
        embedding_device=cfg["graph"].get("relation_embedding_device"),
        embedding_batch_size=int(cfg["graph"].get("relation_embedding_batch_size", 32)),
        force=False,
    )
    validate_graph_x(graph, feat_dim=feat_dim, context="Stage 2 sanity graph")

    target_to_other = build_target_to_other_types(
        graph,
        bundle.mappings.relation2id,
        mention_key=str(cfg["graph"].get("mention_relation_key", "is_mentioned_in")),
    )
    if not target_to_other:
        raise RuntimeError("target_to_other_types is empty; check is_mentioned_in edges")
    graph.target_to_other_types = target_to_other
    graph = graph.to(device)

    stage2_json = Path(str(cfg["data"]["stage2_json"]))
    with open(stage2_json, encoding="utf-8") as f:
        all_samples = json.load(f)
    sanity_cfg = cfg.get("sanity", {}) if isinstance(cfg.get("sanity", {}), dict) else {}
    sanity_n = int(sanity_cfg.get("samples", SANITY_N))
    sanity_samples = all_samples[:sanity_n]
    logger.info("Sanity: using %d samples from %s", len(sanity_samples), stage2_json)

    question_embs = _load_question_embeddings(cfg, sanity_samples)
    validate_graph_x(
        graph,
        question_embeddings=question_embs,
        feat_dim=feat_dim,
        context="Stage 2 sanity samples",
    )

    node2id = {str(k): int(v) for k, v in bundle.mappings.entity2id.items()}
    tiny_ds = Stage2TorchDataset(sanity_samples, question_embs, int(graph.num_nodes), node2id)
    sanity_batch_size = int(cfg["training"].get("train_batch_size", 1))
    sanity_batch_size = max(1, min(sanity_batch_size, len(tiny_ds)))
    loader = DataLoader(tiny_ds, batch_size=sanity_batch_size, shuffle=False)
    logger.info("Sanity batch_size=%d", sanity_batch_size)

    chunk_nodes = graph.nodes_by_type.get("chunk", torch.tensor([], dtype=torch.long, device=device))
    if len(chunk_nodes) == 0:
        raise ValueError("graph.nodes_by_type['chunk'] is empty or missing")

    all_losses = build_stage2_loss_functions(list(cfg.get("losses", [])))
    hard_losses = [loss for loss in all_losses if not loss.is_distillation_loss]
    distill_losses = [loss for loss in all_losses if loss.is_distillation_loss]
    if not hard_losses:
        raise ValueError("Sanity requires at least one hard-label loss")
    if not distill_losses:
        logger.warning("No distillation losses configured; hard-only and distill profiles will be identical.")

    hard_metrics = _run_profile(
        profile_name="hard_only",
        cfg=cfg,
        graph=graph,
        feat_dim=feat_dim,
        loader=loader,
        loss_functions=hard_losses,
        chunk_nodes=chunk_nodes,
        device=device,
        sanity_steps=sanity_steps,
        require_finite_distill=False,
    )
    distill_metrics = _run_profile(
        profile_name="hard_plus_distill",
        cfg=cfg,
        graph=graph,
        feat_dim=feat_dim,
        loader=loader,
        loss_functions=all_losses,
        chunk_nodes=chunk_nodes,
        device=device,
        sanity_steps=sanity_steps,
        require_finite_distill=True,
    )

    hard_mrr = hard_metrics["chunk_mrr"]
    distill_mrr = distill_metrics["chunk_mrr"]
    abs_drop = hard_mrr - distill_mrr
    rel_drop = abs_drop / max(abs(hard_mrr), 1e-12)
    max_abs = float(sanity_cfg.get("max_metric_drop_abs", 0.05))
    max_rel = float(sanity_cfg.get("max_metric_drop_rel", 0.25))

    logger.info("=== SANITY RESULT ===")
    logger.info("  hard_only chunk_mrr=%.4f", hard_mrr)
    logger.info("  hard_plus_distill chunk_mrr=%.4f", distill_mrr)
    logger.info("  drop abs=%.4f rel=%.4f thresholds abs=%.4f rel=%.4f", abs_drop, rel_drop, max_abs, max_rel)

    if abs_drop > max_abs and rel_drop > max_rel:
        raise AssertionError(
            f"SANITY FAIL: distillation chunk_mrr dropped from {hard_mrr:.4f} to {distill_mrr:.4f} "
            f"(abs_drop={abs_drop:.4f}, rel_drop={rel_drop:.4f}). Not starting full train."
        )

    logger.info("SANITY PASS: hard-label + distillation is finite and within metric drop limits.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity test Stage 2 GFM-RAG.")
    parser.add_argument("--config", type=Path, default=Path("configs/graph_retriever/stage2_sft.yaml"))
    parser.add_argument("--sanity-steps", type=int, default=None)
    parser.add_argument("--gfmrag-path", type=str, default=None)
    parser.add_argument("--disable-custom-rspmm", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ns = parse_args(argv)
    if not ns.config.is_file():
        raise FileNotFoundError(f"Config not found: {ns.config}")

    raw = OmegaConf.load(ns.config)
    if ns.gfmrag_path is not None:
        raw.gfmrag_path = ns.gfmrag_path

    effective_gfmrag_path = raw.get("gfmrag_path")
    reject_external_gfmrag_path(effective_gfmrag_path)
    bootstrap_gfmrag(effective_gfmrag_path)
    if ns.disable_custom_rspmm or bool(raw.get("disable_custom_rspmm", True)):
        disable_custom_rspmm()
    log_stage2_runtime_context(ns.config)

    cfg_all = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(cfg_all, dict)
    sanity_cfg = cfg_all.get("sanity", {}) if isinstance(cfg_all.get("sanity", {}), dict) else {}
    sanity_steps = ns.sanity_steps if ns.sanity_steps is not None else int(sanity_cfg.get("steps", SANITY_STEPS))
    run_sanity(cfg_all, sanity_steps=sanity_steps, already_bootstrapped=True)


if __name__ == "__main__":
    main()
