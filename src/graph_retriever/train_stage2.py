"""
Stage 2 GFM-RAG SFT Fine-tuning.

Load checkpoint KGC fine-tuned → GNNRetriever → SFTTrainer.
Input: question + start_nodes (entities) → score mọi node → target: chunk + entity nodes.

Chạy:
  python src/graph_retriever/train_stage2.py --config configs/graph_retriever/stage2_sft.yaml
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

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.graph_retriever.graph_adapter import build_target_to_other_types, load_graph_bundle
from src.graph_retriever.gfm_bootstrap import bootstrap_gfmrag, disable_custom_rspmm
from src.graph_retriever.rel_features import ensure_rel_attr
from src.graph_retriever.stage2_common import (
    LoggingSFTTrainerMixin,
    TeacherFeaturesDistillationMixin,
    build_stage2_loss_functions,
    log_stage2_runtime_context,
    reject_external_gfmrag_path,
    validate_graph_x,
)
from src.graph_retriever.stage2_dataset import (
    Stage2SFTData,
    _SingleSFTDatasetLoader,
    load_stage2_sft_data,
)
from src.graph_retriever.distill_features import DistillationFeatureLoader, validate_distillation_config

logger = logging.getLogger(__name__)


def _pick_device(spec: str) -> torch.device:
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA không khả dụng — chuyển sang CPU.")
        return torch.device("cpu")
    return device


def _load_kgc_checkpoint(
    model: torch.nn.Module,
    ckpt_path: Path,
    device: torch.device,
) -> None:
    """Load KGC checkpoint vào GNNRetriever (strict=False)."""
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(payload, dict):
        state = (
            payload.get("model")
            or payload.get("state_dict")
            or payload.get("model_state_dict")
        )
        if state is None:
            raise ValueError(f"Checkpoint {ckpt_path}: không có key model/state_dict")
        epoch = payload.get("epoch")
        best_metric = payload.get("best_metric")
    else:
        state = payload
        epoch = None
        best_metric = None

    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("=== CHECKPOINT LOAD ===")
    logger.info("  Source: %s", ckpt_path)
    logger.info("  epoch=%s | best_metric=%s", epoch, best_metric)
    logger.info("  missing_keys (%d): %s", len(missing), missing[:5] if missing else "none")
    logger.info("  unexpected_keys (%d): %s", len(unexpected), unexpected[:5] if unexpected else "none")


def run_stage2(*, cfg: dict) -> Path:
    device = _pick_device(str(cfg["training"]["device"]))
    tensor_dir = Path(str(cfg["graph"]["tensor_dir"]))
    output_dir = Path(str(cfg["training"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Load graph ───────────────────────────────────────────
    bundle = load_graph_bundle(
        tensor_dir,
        mention_relation_key=str(cfg["graph"].get("mention_relation_key", "is_mentioned_in")),
    )
    graph = bundle.data
    mappings = bundle.mappings
    validate_graph_x(graph, context="Stage 2 train graph")

    # ─── Đảm bảo rel_attr (BGE embeddings) ───────────────────
    rel2id_path = tensor_dir / "rel2id.json"
    graph, feat_dim = ensure_rel_attr(
        graph,
        rel2id_path=rel2id_path,
        embedding_model=str(cfg["graph"]["relation_embedding_model"]),
        embedding_device=cfg["graph"].get("relation_embedding_device"),
        embedding_batch_size=int(cfg["graph"].get("relation_embedding_batch_size", 32)),
        force=bool(cfg["graph"].get("force_rebuild_rel_attr", False)),
    )
    logger.info("feat_dim (BGE embedding dim): %d", feat_dim)
    validate_graph_x(graph, feat_dim=feat_dim, context="Stage 2 train graph")

    # ─── Build target_to_other_types (entity→chunk sparse) ────
    target_to_other = build_target_to_other_types(
        graph,
        mappings.relation2id,
        mention_key=str(cfg["graph"].get("mention_relation_key", "is_mentioned_in")),
    )
    if not target_to_other:
        raise RuntimeError(
            "build_target_to_other_types trả về rỗng — kiểm tra relation 'is_mentioned_in'"
        )
    graph.target_to_other_types = target_to_other

    # Di chuyển graph (kể cả sparse tensors) lên device trước khi tạo trainer
    # PyG recursive_apply sẽ move cả dict values của Tensor
    graph = graph.to(device)

    # ─── Load node mappings ───────────────────────────────────
    node2id: dict[str, int] = {
        str(k): int(v) for k, v in mappings.entity2id.items()
    }
    id2node_raw: dict[str, str] = json.loads(
        (tensor_dir / "id2node.json").read_text(encoding="utf-8")
    )

    # ─── Load Stage 2 data ────────────────────────────────────
    data_cfg = cfg["data"]
    sft_data: Stage2SFTData = load_stage2_sft_data(
        stage2_json=Path(str(data_cfg["stage2_json"])),
        graph=graph,
        node2id=node2id,
        id2node_raw=id2node_raw,
        text_emb_model_name=str(data_cfg.get("text_emb_model", "BAAI/bge-base-en-v1.5")),
        emb_device=data_cfg.get("emb_device"),
        emb_cache=Path(str(data_cfg["emb_cache"])) if data_cfg.get("emb_cache") else None,
        train_ratio=float(data_cfg.get("train_ratio", 0.8)),
        seed=int(data_cfg.get("seed", 42)),
    )
    validate_graph_x(
        graph,
        question_embeddings=sft_data.train_data.question_embeddings,
        feat_dim=feat_dim,
        context="Stage 2 train split",
    )
    if len(sft_data.test_data) > 0:
        validate_graph_x(
            graph,
            question_embeddings=sft_data.test_data.question_embeddings,
            feat_dim=feat_dim,
            context="Stage 2 eval split",
        )

    # ─── Khởi tạo GNNRetriever ────────────────────────────────
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
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("=== MODEL INFO ===")
    logger.info("  Class: %s", model.__class__.__name__)
    logger.info("  Total params: %d", num_params)
    logger.info("  feat_dim: %d | init_nodes_weight: %s | init_nodes_type: %s",
                feat_dim, model_cfg.get("init_nodes_weight"), model_cfg.get("init_nodes_type"))

    # ─── Load KGC checkpoint ──────────────────────────────────
    pretrained_path = cfg["training"].get("pretrained_model_path")
    if pretrained_path:
        pre_path = Path(str(pretrained_path))
        if not pre_path.is_absolute():
            pre_path = (_REPO_ROOT / pre_path).resolve()
        _load_kgc_checkpoint(model, pre_path, device)
    else:
        logger.warning("Không có pretrained_model_path — train from scratch!")

    # ─── Load Distillation Features (if enabled) ───────────────
    distill_loader = None
    distill_cfg = cfg.get("distillation", {})
    if distill_cfg.get("enable", False):
        mode = distill_cfg.get("mode", "author_graph_x")
        if mode == "teacher_features":
            teacher_feature_dir = Path(str(distill_cfg.get("teacher_feature_dir", "")))
            logger.info(f"Loading distillation features from: {teacher_feature_dir}")
            
            distill_loader = DistillationFeatureLoader(
                teacher_feature_dir=teacher_feature_dir,
                num_samples=len(sft_data.train_data) + len(sft_data.test_data),
                num_nodes=graph.num_nodes,
                device=device,
            )
            
            distill_info = distill_loader.load()
            distill_loader.log_info()
            
            # Validate config
            if not validate_distillation_config(cfg, distill_loader):
                logger.error("Distillation config validation failed")
                sys.exit(1)
            
            logger.info(f"✅ Distillation features loaded successfully")
        else:
            logger.info(f"Distillation mode: {mode} (using author_graph_x, no separate loader)")

    # ─── Optimizer ────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )

    # ─── Loss functions ───────────────────────────────────────
    from gfmrag.losses import BCELoss, KLDivLoss, ListCELoss, MSELoss  # noqa:F401
    from gfmrag.trainers.sft_trainer import SFTTrainer
    from gfmrag.trainers.training_args import TrainingArguments

    loss_cfg_list = cfg.get("losses", [
        {"name": "bce_chunk", "loss_type": "bce", "weight": 0.3, "target_node_type": "chunk", "is_distillation_loss": False},
        {"name": "listce_chunk", "loss_type": "listce", "weight": 0.7, "target_node_type": "chunk", "is_distillation_loss": False},
    ])
    loss_functions = build_stage2_loss_functions(list(loss_cfg_list))

    # ─── TrainingArguments ────────────────────────────────────
    tcfg = cfg["training"]
    args = TrainingArguments(
        num_epoch=int(tcfg["epochs"]),
        train_batch_size=int(tcfg["train_batch_size"]),
        eval_batch_size=int(tcfg["eval_batch_size"]),
        max_steps_per_epoch=tcfg.get("max_steps_per_epoch"),
        logging_steps=int(tcfg["logging_steps"]),
        save_best_only=bool(tcfg.get("save_best_only", True)),
        metric_for_best_model=str(tcfg.get("metric_for_best_model", "chunk_mrr")),
        eval_strategy=str(tcfg.get("eval_strategy", "epoch")),
        eval_steps=tcfg.get("eval_steps"),
        dtype=str(tcfg.get("dtype", "float32")),
    )

    # ─── Loaders ──────────────────────────────────────────────
    graph_name = str(cfg["graph"].get("name", "multifin_graph"))
    train_loader = _SingleSFTDatasetLoader(graph_name, sft_data)
    eval_loader = _SingleSFTDatasetLoader(graph_name, sft_data)

    # ─── SFTTrainer ───────────────────────────────────────────
    class Stage2DistillTrainer(TeacherFeaturesDistillationMixin, LoggingSFTTrainerMixin, SFTTrainer):
        """Trainer with logging and teacher features distillation support."""
        pass

    trainer = Stage2DistillTrainer(
        output_dir=str(output_dir),
        args=args,
        model=model,
        optimizer=optimizer,
        loss_functions=loss_functions,
        train_graph_dataset_loader=train_loader,
        eval_graph_dataset_loader=eval_loader,
        target_types=list(cfg.get("target_types", ["entity", "chunk"])),
        metrics=list(cfg.get("metrics", ["mrr", "hits@1", "hits@5", "hits@10"])),
    )

    # ─── Attach distillation features to trainer ───────────────
    if distill_loader is not None:
        trainer.distill_loader = distill_loader
        trainer.distill_config = distill_cfg
        logger.info("Attached distillation features to trainer")
    else:
        trainer.distill_loader = None
        trainer.distill_config = None

    logger.info("=== TRAINING START ===")
    logger.info("  output_dir: %s", output_dir)
    logger.info("  epochs: %d | batch: %d | lr: %s", args.num_epoch, args.train_batch_size, tcfg["lr"])
    logger.info("  target_types: %s | metric: %s", cfg.get("target_types"), args.metric_for_best_model)
    
    # Log distillation info
    if distill_loader is not None:
        logger.info("=== DISTILLATION INFO ===")
        logger.info("  enabled: true")
        logger.info("  mode: teacher_features")
        logger.info("  teacher_model: %s", distill_cfg.get("teacher_model", "unknown"))
        logger.info("  teacher_feature_dir: %s", distill_cfg.get("teacher_feature_dir", "unknown"))
        logger.info("  teacher node_x: %s", distill_loader.node_x.shape)
        logger.info("  teacher question_emb: %s", distill_loader.question_embeddings.shape)
        logger.info("  teacher embedding_dim: %d", distill_loader.embedding_dim)
    else:
        logger.info("  distillation: %s", "disabled" if not distill_cfg.get("enable") else f"mode={distill_cfg.get('mode')}")

    trainer.train()

    # ─── Report ───────────────────────────────────────────────
    best_path = output_dir / "model_best.pth"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        best_metric = ckpt.get("best_metric", "N/A")
        best_epoch = ckpt.get("best_epoch", "N/A")
        logger.info("=== TRAINING REPORT ===")
        logger.info("  KGC pretrained metric: typed_mrr=0.2666 (epoch 13)")
        logger.info("  Stage2 best %s: %s (epoch %s)", args.metric_for_best_model, best_metric, best_epoch)
        logger.info("  Checkpoint: %s", best_path)
        return best_path

    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 GFM-RAG SFT Fine-tuning.")
    p.add_argument("--config", type=Path, default=Path("configs/graph_retriever/stage2_sft.yaml"))
    p.add_argument("--gfmrag-path", type=str, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--pretrained", type=Path, default=None)
    p.add_argument("--stage2-json", type=Path, default=None)
    p.add_argument("--disable-custom-rspmm", action="store_true")
    p.add_argument("--run-sanity-first", action="store_true")
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
    if not ns.config.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {ns.config}")

    raw = OmegaConf.load(ns.config)
    if ns.output_dir is not None:
        raw.training.output_dir = str(ns.output_dir.resolve())
    if ns.epochs is not None:
        raw.training.epochs = ns.epochs
    if ns.pretrained is not None:
        raw.training.pretrained_model_path = str(ns.pretrained)
    if ns.stage2_json is not None:
        raw.data.stage2_json = str(ns.stage2_json)

    effective_gfmrag_path = ns.gfmrag_path or raw.get("gfmrag_path")
    reject_external_gfmrag_path(effective_gfmrag_path)
    bootstrap_gfmrag(effective_gfmrag_path)
    if ns.disable_custom_rspmm or bool(raw.get("disable_custom_rspmm", True)):
        disable_custom_rspmm()
    log_stage2_runtime_context(ns.config)

    cfg_all = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(cfg_all, dict)

    if ns.run_sanity_first:
        from src.graph_retriever.sanity_stage2 import run_sanity

        sanity_cfg = cfg_all.get("sanity", {})
        sanity_steps = int(sanity_cfg.get("steps", 300)) if isinstance(sanity_cfg, dict) else 300
        logger.info("=== RUN SANITY FIRST ===")
        run_sanity(cfg_all, sanity_steps=sanity_steps, already_bootstrapped=True)

    ckpt = run_stage2(cfg=cfg_all)
    logger.info("Hoàn thành Stage 2 — checkpoint: %s", ckpt)


if __name__ == "__main__":
    main()
