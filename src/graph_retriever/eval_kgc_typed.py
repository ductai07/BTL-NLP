"""Compare pretrained and fine-tuned KGC checkpoints with typed ranking metrics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.graph_retriever.debug_kgc import _eval_ranking, _load_real_model_and_graph
from src.graph_retriever.gfm_bootstrap import bootstrap_gfmrag, disable_custom_rspmm


def _print_metrics(label: str, metrics: dict[str, float]) -> None:
    pretty = " ".join(f"{key}={value:.6f}" for key, value in metrics.items())
    print(f"{label}: {pretty}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Typed/all-node KGC eval for pretrained vs fine-tuned checkpoints."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/graph_retriever/kgc_gfm_training_typed.yaml"),
    )
    parser.add_argument("--pretrained", type=Path, default=Path("model/model.pth"))
    parser.add_argument(
        "--finetuned",
        type=Path,
        default=Path("outputs/graph_retriever/kgc_gfm_typed_after_patch/model_best.pth"),
    )
    parser.add_argument("--eval-sample", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    bootstrap_gfmrag(None)
    disable_custom_rspmm()

    for label, checkpoint in [
        ("pretrained", args.pretrained),
        ("fine_tuned", args.finetuned),
    ]:
        model, graph, triples = _load_real_model_and_graph(
            config=args.config,
            checkpoint=checkpoint,
            eval_sample=args.eval_sample,
            seed=args.seed,
        )
        print(f"\n=== {label} checkpoint={checkpoint} eval_triples={triples.size(0)} ===")
        _print_metrics("all-node", _eval_ranking(model, graph, triples, typed=False))
        _print_metrics("typed", _eval_ranking(model, graph, triples, typed=True))


if __name__ == "__main__":
    main()
