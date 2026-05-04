"""
Đánh giá retrieval cho Stage 2 GFM-RAG model trên toàn bộ test set.

Chạy:
  python src/graph_retriever/eval_stage2.py \
      --config configs/graph_retriever/stage2_sft.yaml \
      --checkpoint outputs/graph_retriever/kgc_stage2_sft/model_best.pth \
      --data data/qa/test_qa_stage2.json \
      --output outputs/graph_retriever/kgc_stage2_sft/eval_results.json \
      --top-k 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.graph_retriever.graph_adapter import build_target_to_other_types, load_graph_bundle
from src.graph_retriever.gfm_bootstrap import bootstrap_gfmrag, disable_custom_rspmm
from src.graph_retriever.rel_features import ensure_rel_attr
from src.graph_retriever.stage2_dataset import Stage2TorchDataset

logger = logging.getLogger(__name__)

METRICS = ["mrr", "hits@1", "hits@2", "hits@5", "hits@10", "hits@20", "recall@5", "recall@10", "recall@20"]


def _compute_metrics(
    all_pred: torch.Tensor,
    all_tgt: torch.Tensor,
    metrics: list[str],
    type_name: str,
) -> dict[str, float]:
    """
    all_pred: (N_samples, N_type_nodes) logits
    all_tgt:  (N_samples, N_type_nodes) binary targets
    """
    from gfmrag.utils import batch_evaluate, evaluate as gfm_evaluate
    from gfmrag.models.ultra import query_utils

    results: dict[str, float] = {}
    # Chỉ tính trên sample có ít nhất 1 positive
    has_pos = all_tgt.sum(dim=-1) > 0
    if not has_pos.any():
        logger.warning("[%s] Không có sample nào có positive target — bỏ qua", type_name)
        return {m: 0.0 for m in metrics}

    pred_f = all_pred[has_pos]
    tgt_f = all_tgt[has_pos].bool()

    preds_list: list[tuple] = []
    targets_list: list[tuple] = []

    for i in range(0, pred_f.size(0), 64):
        p = pred_f[i:i+64]
        t = tgt_f[i:i+64]
        node_ranking, target_ranking = batch_evaluate(p, t)
        num_pred = (torch.sigmoid(p) > 0.5).sum(dim=-1)
        num_target = t.sum(dim=-1)
        preds_list.append((node_ranking, num_pred))
        targets_list.append((target_ranking, num_target))

    node_pred = query_utils.cat(preds_list)
    node_target = query_utils.cat(targets_list)

    results = gfm_evaluate(node_pred, node_target, metrics)
    return results


@torch.no_grad()
def run_eval(
    cfg: dict,
    checkpoint_path: Path,
    data_path: Path,
    output_path: Path,
    top_k: int = 20,
) -> dict:
    device_spec = str(cfg["training"].get("device", "cuda"))
    device = torch.device(device_spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA không khả dụng — chuyển sang CPU.")
        device = torch.device("cpu")

    tensor_dir = Path(str(cfg["graph"]["tensor_dir"]))

    # ── Load graph ────────────────────────────────────────────────
    logger.info("Load graph từ %s", tensor_dir)
    bundle = load_graph_bundle(
        tensor_dir,
        mention_relation_key=str(cfg["graph"].get("mention_relation_key", "is_mentioned_in")),
    )
    graph = bundle.data
    mappings = bundle.mappings

    graph, feat_dim = ensure_rel_attr(
        graph,
        rel2id_path=tensor_dir / "rel2id.json",
        embedding_model=str(cfg["graph"]["relation_embedding_model"]),
        embedding_device=cfg["graph"].get("relation_embedding_device"),
        embedding_batch_size=int(cfg["graph"].get("relation_embedding_batch_size", 32)),
        force=False,
    )
    logger.info("feat_dim: %d", feat_dim)

    target_to_other = build_target_to_other_types(
        graph, mappings.relation2id,
        mention_key=str(cfg["graph"].get("mention_relation_key", "is_mentioned_in")),
    )
    graph.target_to_other_types = target_to_other
    graph = graph.to(device)

    # ── Load data ─────────────────────────────────────────────────
    with open(data_path, encoding="utf-8") as f:
        all_samples = json.load(f)
    logger.info("Loaded %d samples từ %s", len(all_samples), data_path)

    node2id: dict[str, int] = {str(k): int(v) for k, v in mappings.entity2id.items()}
    id2node_raw: dict[str, str] = json.loads(
        (tensor_dir / "id2node.json").read_text(encoding="utf-8")
    )

    # Load question embeddings
    data_cfg = cfg["data"]
    emb_cache = Path(str(data_cfg["emb_cache"])) if data_cfg.get("emb_cache") else None
    if emb_cache and emb_cache.exists():
        all_embs = torch.load(emb_cache, map_location="cpu", weights_only=False)
        # Cache có thể chỉ có 120 sample từ training — pad nếu cần
        if all_embs.size(0) < len(all_samples):
            logger.warning(
                "Cache chỉ có %d embeddings nhưng cần %d — encode thêm phần còn thiếu",
                all_embs.size(0), len(all_samples),
            )
            extra_samples = all_samples[all_embs.size(0):]
            from sentence_transformers import SentenceTransformer
            text_model = SentenceTransformer(
                str(data_cfg.get("text_emb_model", "BAAI/bge-base-en-v1.5")),
                device=data_cfg.get("emb_device"),
            )
            extra_embs = text_model.encode(
                [s["question"] for s in extra_samples],
                normalize_embeddings=True, show_progress_bar=True,
            )
            all_embs = torch.cat([all_embs, torch.tensor(extra_embs, dtype=torch.float32)])
        question_embs = all_embs[:len(all_samples)].float()
        logger.info("Loaded question embeddings: %s", question_embs.shape)
    else:
        logger.info("Không có cache — encode %d questions bằng BGE", len(all_samples))
        from sentence_transformers import SentenceTransformer
        text_model = SentenceTransformer(
            str(data_cfg.get("text_emb_model", "BAAI/bge-base-en-v1.5")),
            device=data_cfg.get("emb_device"),
        )
        embs = text_model.encode(
            [s["question"] for s in all_samples],
            normalize_embeddings=True, show_progress_bar=True,
        )
        question_embs = torch.tensor(embs, dtype=torch.float32)

    dataset = Stage2TorchDataset(all_samples, question_embs, int(graph.num_nodes), node2id)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # ── Load model ────────────────────────────────────────────────
    from gfmrag.models.gfm_rag_v1 import model as gfm_model_module
    from gfmrag.models.gfm_rag_v1.rankers import SimpleRanker

    entity_model = instantiate(OmegaConf.create(cfg["model"]["entity_model"]))
    ranker = SimpleRanker()
    model_cfg = cfg["model"]
    GNNRetriever = gfm_model_module.GNNRetriever
    model = GNNRetriever(
        entity_model=entity_model,
        feat_dim=int(feat_dim),
        ranker=ranker,
        init_nodes_weight=bool(model_cfg.get("init_nodes_weight", True)),
        init_nodes_type=str(model_cfg.get("init_nodes_type", "chunk")),
    )

    ckpt_path = checkpoint_path
    if not ckpt_path.is_absolute():
        ckpt_path = (_REPO_ROOT / ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint không tìm thấy: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else payload
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Checkpoint: %s | missing=%d unexpected=%d", ckpt_path.name, len(missing), len(unexpected))
    if isinstance(payload, dict):
        logger.info("  epoch=%s | best_metric=%s", payload.get("epoch"), payload.get("best_metric"))
    model = model.to(device)
    model.eval()

    # ── Inference ─────────────────────────────────────────────────
    chunk_nodes = graph.nodes_by_type.get("chunk", torch.tensor([], dtype=torch.long, device=device))
    entity_nodes = graph.nodes_by_type.get("entity", torch.tensor([], dtype=torch.long, device=device))
    logger.info("chunk_nodes: %d | entity_nodes: %d", len(chunk_nodes), len(entity_nodes))

    all_chunk_pred, all_chunk_tgt = [], []
    all_ent_pred, all_ent_tgt = [], []
    retrieval_results = []

    for batch in tqdm(loader, desc="Evaluating"):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(graph, batch)  # (1, N_nodes)
        tgt = batch["target_nodes_mask"]  # (1, N_nodes)

        all_chunk_pred.append(pred[:, chunk_nodes].cpu())
        all_chunk_tgt.append(tgt[:, chunk_nodes].cpu())
        all_ent_pred.append(pred[:, entity_nodes].cpu())
        all_ent_tgt.append(tgt[:, entity_nodes].cpu())

    all_chunk_pred_t = torch.cat(all_chunk_pred)   # (N_samples, N_chunk)
    all_chunk_tgt_t = torch.cat(all_chunk_tgt)
    all_ent_pred_t = torch.cat(all_ent_pred)
    all_ent_tgt_t = torch.cat(all_ent_tgt)

    # ── Metrics ───────────────────────────────────────────────────
    logger.info("=== Tính metrics trên %d samples ===", len(all_samples))

    chunk_metrics = _compute_metrics(all_chunk_pred_t, all_chunk_tgt_t, METRICS, "chunk")
    entity_metrics = _compute_metrics(all_ent_pred_t, all_ent_tgt_t, METRICS, "entity")

    logger.info("--- CHUNK RETRIEVAL ---")
    for k, v in sorted(chunk_metrics.items()):
        logger.info("  chunk_%-15s %.4f", k, v)

    logger.info("--- ENTITY RETRIEVAL ---")
    for k, v in sorted(entity_metrics.items()):
        logger.info("  entity_%-14s %.4f", k, v)

    # ── Top-K predictions ─────────────────────────────────────────
    logger.info("=== Tạo top-%d prediction cho mỗi sample ===", top_k)
    chunk_node_list = chunk_nodes.cpu().tolist()

    for i, sample in enumerate(all_samples):
        chunk_scores = all_chunk_pred_t[i]
        top_k_local = torch.topk(chunk_scores, k=min(top_k, len(chunk_node_list))).indices
        top_k_global = [chunk_node_list[j] for j in top_k_local.tolist()]
        top_k_preds = [
            {"uid": id2node_raw.get(str(g), f"node_{g}"), "score": float(chunk_scores[j])}
            for j, g in zip(top_k_local.tolist(), top_k_global)
        ]
        # Ground truth chunks
        gt_chunks = sample.get("target_nodes", {}).get("chunk", [])
        retrieval_results.append({
            "id": sample["id"],
            "question": sample["question"],
            "ground_truth_chunks": gt_chunks,
            "predicted_chunks": top_k_preds,
            "hit@1": any(p["uid"] == gt for p in top_k_preds[:1] for gt in gt_chunks),
            "hit@5": any(p["uid"] == gt for p in top_k_preds[:5] for gt in gt_chunks),
            "hit@10": any(p["uid"] == gt for p in top_k_preds[:10] for gt in gt_chunks),
        })

    # ── Lưu output ────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "checkpoint": str(ckpt_path),
        "data": str(data_path),
        "n_samples": len(all_samples),
        "n_samples_with_chunk_target": int((all_chunk_tgt_t.sum(dim=-1) > 0).sum()),
        "n_samples_with_entity_target": int((all_ent_tgt_t.sum(dim=-1) > 0).sum()),
        "chunk_metrics": chunk_metrics,
        "entity_metrics": entity_metrics,
        "retrieval_results": retrieval_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Đã lưu kết quả: %s", output_path)

    # ── In bảng tóm tắt ───────────────────────────────────────────
    logger.info("\n" + "="*55)
    logger.info("  STAGE 2 RETRIEVAL EVALUATION — %d samples", len(all_samples))
    logger.info("="*55)
    logger.info("  %-20s %-10s %-10s", "Metric", "Chunk", "Entity")
    logger.info("  " + "-"*43)
    for m in METRICS:
        cv = chunk_metrics.get(m, 0.0)
        ev = entity_metrics.get(m, 0.0)
        logger.info("  %-20s %-10.4f %-10.4f", m, cv, ev)
    logger.info("="*55)

    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Stage 2 GFM-RAG retrieval.")
    p.add_argument("--config", type=Path, default=Path("configs/graph_retriever/stage2_sft.yaml"))
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/graph_retriever/kgc_stage2_sft/model_best.pth"))
    p.add_argument("--data", type=Path, default=Path("data/qa/test_qa_stage2.json"))
    p.add_argument("--output", type=Path, default=Path("outputs/graph_retriever/kgc_stage2_sft/eval_results.json"))
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--gfmrag-path", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ns = parse_args(argv)
    raw = OmegaConf.load(ns.config)
    if ns.device:
        raw.training.device = ns.device

    bootstrap_gfmrag(ns.gfmrag_path or raw.get("gfmrag_path"))
    if bool(raw.get("disable_custom_rspmm", True)):
        disable_custom_rspmm()

    cfg = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(cfg, dict)

    run_eval(
        cfg=cfg,
        checkpoint_path=ns.checkpoint,
        data_path=ns.data,
        output_path=ns.output,
        top_k=ns.top_k,
    )


if __name__ == "__main__":
    main()
