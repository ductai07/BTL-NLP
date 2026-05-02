"""
Entrypoint huấn luyện KGC pretraining bằng model gốc của tác giả:
QueryGNN + QueryNBFNet, trainer KGCTrainer, negative sampling từ gfmrag/models/ultra/tasks.py

Chạy ví dụ:
  python -m src.graph_retriever.train_kgc --config configs/graph_retriever/kgc_gfm_training.yaml --gfmrag-path d:/Project/gfm-rag
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Cho phép chạy từ thư mục gốc project
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.graph_retriever.graph_adapter import load_graph_bundle  # noqa:E402
from src.graph_retriever.gfm_bootstrap import (  # noqa:E402
    bootstrap_gfmrag,
    disable_custom_rspmm,
)
from src.graph_retriever.rel_features import ensure_rel_attr  # noqa:E402

logger = logging.getLogger(__name__)


def _pick_device(spec: str) -> torch.device:
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA không khả dụng — chọn CPU.")
        return torch.device("cpu")
    return device


class _SingleGraphDatasetLoader:
    """
    Adapter nhỏ để dùng KGCTrainer của tác giả mà không cần GraphIndexDataset loader.
    KGCTrainer chỉ cần iterate ra object có `.name` và `.data.graph`.
    """

    def __init__(self, name: str, graph) -> None:
        self._name = name
        self._graph = graph

    def set_epoch(self, epoch: int) -> None:  # noqa:ARG002
        return

    def shutdown(self) -> None:
        return

    def __iter__(self):
        from types import SimpleNamespace

        yield SimpleNamespace(name=self._name, data=SimpleNamespace(graph=self._graph))


def run_training(*, cfg: dict) -> Path:
    device = _pick_device(str(cfg["training"]["device"]))
    tensor_dir = Path(str(cfg["graph"]["tensor_dir"]))
    graph_dir = Path(str(cfg["graph"]["graph_dir"]))
    output_dir = Path(str(cfg["training"]["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_graph_bundle(
        tensor_dir,
        build_relation_graph=bool(cfg["graph"].get("build_relation_graph", False)),
        mention_relation_key=str(cfg["graph"].get("mention_relation_key", "is_mentioned_in")),
    )
    graph = bundle.data

    # Đảm bảo rel_attr tồn tại (QueryGNN bắt buộc)
    rel2id_path = tensor_dir / "rel2id.json"
    graph, feat_dim = ensure_rel_attr(
        graph,
        rel2id_path=rel2id_path,
        embedding_model=str(cfg["graph"]["relation_embedding_model"]),
        embedding_device=cfg["graph"].get("relation_embedding_device"),
        embedding_batch_size=int(cfg["graph"].get("relation_embedding_batch_size", 32)),
        force=bool(cfg["graph"].get("force_rebuild_rel_attr", False)),
    )

    graph = graph.to(device)

    # Import đúng các class từ repo tác giả sau khi bootstrap
    from gfmrag.models.gfm_rag_v1 import QueryGNN
    from gfmrag.trainers.kgc_trainer import KGCTrainer
    from gfmrag.trainers.training_args import TrainingArguments

    # Instantiate entity_model (QueryNBFNet) theo YAML
    entity_model = instantiate(OmegaConf.create(cfg["model"]["entity_model"]))
    model = QueryGNN(entity_model=entity_model, feat_dim=int(feat_dim)).to(device)

    # Tùy chọn: load pretrained của tác giả trước khi finetune
    pretrained_path = cfg["training"].get("pretrained_model_path")
    if pretrained_path:
        pre_path = Path(str(pretrained_path))
        if not pre_path.is_absolute():
            pre_path = (_REPO_ROOT / pre_path).resolve()
        if not pre_path.exists():
            raise FileNotFoundError(f"Không tìm thấy pretrained_model_path: {pre_path}")
        payload = torch.load(pre_path, map_location=device, weights_only=False)
        if isinstance(payload, dict):
            state = (
                payload.get("model")
                or payload.get("state_dict")
                or payload.get("model_state_dict")
            )
            if state is None:
                raise ValueError(
                    f"Checkpoint {pre_path} không có key model/state_dict/model_state_dict"
                )
        else:
            state = payload
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            "Đã nạp pretrained: %s (missing=%s, unexpected=%s)",
            pre_path,
            len(missing),
            len(unexpected),
        )

    num_params = sum(p.numel() for p in model.parameters())
    logger.info("So tham so model (QueryGNN+QueryNBFNet): %s", num_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))

    args = TrainingArguments(
        num_epoch=int(cfg["training"]["epochs"]),
        train_batch_size=int(cfg["training"]["train_batch_size"]),
        eval_batch_size=int(cfg["training"]["eval_batch_size"]),
        max_steps_per_epoch=cfg["training"].get("max_steps_per_epoch"),
        logging_steps=int(cfg["training"]["logging_steps"]),
        save_best_only=bool(cfg["training"].get("save_best_only", True)),
        metric_for_best_model=str(cfg["training"].get("metric_for_best_model", "mrr")),
        eval_strategy=str(cfg["training"].get("eval_strategy", "epoch")),
        eval_steps=cfg["training"].get("eval_steps"),
        dtype=str(cfg["training"].get("dtype", "float32")),
    )

    train_loader = _SingleGraphDatasetLoader(str(cfg["graph"].get("name", "local_graph")), graph)
    trainer = KGCTrainer(
        output_dir=str(output_dir),
        args=args,
        model=model,
        optimizer=optimizer,
        train_graph_dataset_loader=train_loader,
        eval_graph_dataset_loader=train_loader,
        num_negative=int(cfg["training"]["num_negative"]),
        strict_negative=bool(cfg["training"]["strict_negative"]),
        adversarial_temperature=float(cfg["training"]["adversarial_temperature"]),
        fast_test=int(cfg["training"].get("fast_test", 500)),
        metrics=list(cfg["training"].get("metrics", ["mr", "mrr", "hits@10"])),
    )

    trainer.train()

    # checkpoint best nằm tại output_dir/model_best.pth (theo BaseTrainer)
    best = output_dir / "model_best.pth"
    if best.exists():
        return best
    # fallback: trả về output_dir nếu best không được ghi vì cấu hình
    return output_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="KGC pretraining (QueryGNN/QueryNBFNet) trên graph tensor.")
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/graph_retriever/kgc_gfm_training.yaml"),
        help="Tệp YAML cấu hình (OmegaConf).",
    )
    p.add_argument(
        "--gfmrag-path",
        type=str,
        default=None,
        help="Đường dẫn repo tác giả (chứa thư mục gfmrag/). Ví dụ: d:/Project/gfm-rag",
    )
    p.add_argument("--tensor-dir", type=Path, default=None, help="Ghi đè graph.tensor_dir.")
    p.add_argument("--output-dir", type=Path, default=None, help="Ghi đè training.output_dir.")
    p.add_argument("--epochs", type=int, default=None, help="Ghi đè training.epochs.")
    p.add_argument(
        "--max-steps-per-epoch",
        type=int,
        default=None,
        help="Ghi đè training.max_steps_per_epoch để chạy smoke/short run.",
    )
    p.add_argument(
        "--pretrained",
        type=Path,
        default=None,
        help="Đường dẫn model pretrained (.pth) để finetune.",
    )
    p.add_argument("--disable-custom-rspmm", action="store_true", help="Patch để không dùng rspmm extension.")
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
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {ns.config}")

    raw = OmegaConf.load(ns.config)
    if ns.tensor_dir is not None:
        raw.graph.tensor_dir = str(ns.tensor_dir.resolve())
    if ns.output_dir is not None:
        raw.training.output_dir = str(ns.output_dir.resolve())
    if ns.epochs is not None:
        raw.training.epochs = ns.epochs
    if ns.max_steps_per_epoch is not None:
        raw.training.max_steps_per_epoch = ns.max_steps_per_epoch
    if ns.pretrained is not None:
        raw.training.pretrained_model_path = str(ns.pretrained)

    bootstrap_gfmrag(ns.gfmrag_path or raw.get("gfmrag_path"))
    if ns.disable_custom_rspmm or bool(raw.get("disable_custom_rspmm", True)):
        disable_custom_rspmm()

    cfg_all = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(cfg_all, dict)
    ckpt = run_training(cfg=cfg_all)
    logger.info("Hoàn thành huấn luyện — checkpoint: %s", ckpt)


if __name__ == "__main__":
    main()
