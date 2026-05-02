from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG = Path("configs/gfmrag_reasoning/sft_training.yaml")

logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def bootstrap_gfmrag(gfmrag_path: str | None) -> None:
    path = gfmrag_path or os.environ.get("GFM_RAG_PATH")
    if path:
        resolved_path = Path(path).resolve()
        package_dir = resolved_path / "gfmrag"
        if package_dir.exists():
            package = types.ModuleType("gfmrag")
            package.__path__ = [str(package_dir)]  # type: ignore[attr-defined]
            sys.modules["gfmrag"] = package
            install_lightweight_text_emb_module()
            return
        resolved = str(resolved_path)
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    try:
        import gfmrag  # noqa:F401
    except ModuleNotFoundError as exc:
        if exc.name != "gfmrag":
            raise ModuleNotFoundError(
                f"gfmrag was found, but dependency '{exc.name}' is missing. "
                "Install GFM-RAG training dependencies, for example:\n"
                "  python -m pip install -r requirements-rag.txt\n"
                "  python -m pip install -e d:/Project/gfm-rag"
            ) from exc
        raise ModuleNotFoundError(
            "Cannot import gfmrag. Install your gfm-rag repo or pass "
            "--gfmrag-path d:/Project/gfm-rag."
        ) from exc


def install_lightweight_text_emb_module() -> None:
    """Provide the BaseTextEmbModel used by training without importing NV/Qwen backends."""

    if "gfmrag.text_emb_models" in sys.modules:
        return

    module = types.ModuleType("gfmrag.text_emb_models")

    class BaseTextEmbModel:
        def __init__(
            self,
            text_emb_model_name: str,
            normalize: bool = False,
            batch_size: int = 32,
            query_instruct: str | None = None,
            passage_instruct: str | None = None,
            model_kwargs: dict[str, Any] | None = None,
        ) -> None:
            from sentence_transformers import SentenceTransformer

            self.text_emb_model_name = text_emb_model_name
            self.normalize = normalize
            self.batch_size = batch_size
            self.query_instruct = query_instruct
            self.passage_instruct = passage_instruct
            self.text_emb_model = SentenceTransformer(
                text_emb_model_name,
                trust_remote_code=True,
                model_kwargs=model_kwargs,
            )

        def encode(
            self,
            text: list[str],
            is_query: bool = False,
            show_progress_bar: bool = True,
        ):
            import torch

            prompt = self.query_instruct if is_query else self.passage_instruct
            encoded = self.text_emb_model.encode(
                text,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                prompt=prompt,
                show_progress_bar=show_progress_bar,
                convert_to_tensor=True,
            )
            if isinstance(encoded, torch.Tensor):
                return encoded.float().cpu()
            return torch.tensor(encoded, dtype=torch.float32)

    module.BaseTextEmbModel = BaseTextEmbModel
    module.__all__ = ["BaseTextEmbModel"]
    sys.modules["gfmrag.text_emb_models"] = module


def disable_custom_rspmm() -> None:
    """Force Ultra layers to use PyG message passing instead of compiled rspmm."""

    from torch_geometric.nn.conv import MessagePassing

    from gfmrag.models.ultra.layers import GeneralizedRelationalConv

    def pyg_propagate(self, edge_index, size=None, **kwargs):
        return MessagePassing.propagate(self, edge_index, size=size, **kwargs)

    GeneralizedRelationalConv.propagate = pyg_propagate  # type: ignore[method-assign]


def train(config_path: Path, gfmrag_path: str | None = None) -> Path:
    raw_cfg = load_yaml(config_path)
    bootstrap_gfmrag(gfmrag_path or raw_cfg.get("gfmrag_path"))

    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from gfmrag import utils
    from gfmrag.graph_index_datasets import GraphDatasetLoader
    from gfmrag.trainers.sft_trainer import SFTLoss

    cfg = OmegaConf.create(raw_cfg)
    if cfg.get("disable_custom_rspmm", False):
        disable_custom_rspmm()
    utils.init_distributed_mode(cfg.get("timeout", None))
    torch.manual_seed(int(cfg.seed) + utils.get_rank())

    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    if utils.is_main_process():
        (output_dir / "resolved_config.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True),
            encoding="utf-8",
        )

    if cfg.datasets.init_datasets:
        feat_dim_list = utils.init_multi_dataset(
            cfg,
            utils.get_world_size(),
            utils.get_rank(),
        )
        feat_dims = set(feat_dim_list)
        if len(feat_dims) != 1:
            raise ValueError(f"All datasets must share feat_dim, got {feat_dims}")
        feat_dim = feat_dims.pop()
    else:
        feat_dim = int(cfg.datasets.feat_dim)

    if cfg.get("load_model_from_pretrained"):
        model, _ = utils.load_model_from_pretrained(cfg.load_model_from_pretrained)
    else:
        model = instantiate(cfg.model, feat_dim=feat_dim)

    optimizer = instantiate(cfg.optimizer, model.parameters())
    loss_functions: list[SFTLoss] = []
    for loss_cfg in cfg.losses:
        loss_functions.append(
            SFTLoss(
                name=loss_cfg.name,
                loss_fn=instantiate(loss_cfg.loss),
                weight=float(loss_cfg.weight),
                target_node_type=loss_cfg.target_node_type,
                is_distillation_loss=loss_cfg.get("is_distillation_loss", False),
            )
        )

    train_loader = GraphDatasetLoader(
        cfg.datasets,
        cfg.datasets.train_names,
        max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
        data_loading_workers=cfg.datasets.data_loading_workers,
    )
    valid_loader = GraphDatasetLoader(
        cfg.datasets,
        cfg.datasets.valid_names,
        shuffle=False,
        max_datasets_in_memory=cfg.datasets.max_datasets_in_memory,
        data_loading_workers=cfg.datasets.data_loading_workers,
    )

    trainer = instantiate(
        cfg.trainer,
        output_dir=str(output_dir),
        model=model,
        optimizer=optimizer,
        loss_functions=loss_functions,
        train_graph_dataset_loader=train_loader,
        eval_graph_dataset_loader=valid_loader,
    )

    try:
        trainer.train()
        if utils.is_main_process() and cfg.get("save_pretrained", False):
            pretrained_dir = output_dir / "pretrained"
            utils.save_model_to_pretrained(model, cfg, str(pretrained_dir))
        if trainer.args.do_predict:
            predictions = trainer.predict()
            if utils.is_main_process():
                for data_name, preds in predictions.items():
                    pred_path = output_dir / f"predictions_{data_name}.json"
                    pred_path.write_text(
                        json.dumps(preds, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
    finally:
        train_loader.shutdown()
        valid_loader.shutdown()
        utils.synchronize()
        utils.cleanup()

    return output_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Train a GFM-RAG graph reasoning model on the local financial graph."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gfmrag-path", default=None)
    args = parser.parse_args()

    output_dir = train(args.config, args.gfmrag_path)
    print(f"Training output: {output_dir}")


if __name__ == "__main__":
    main()
