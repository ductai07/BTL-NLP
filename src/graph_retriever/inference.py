"""
Infer top-k entities từ checkpoint KGC (QueryGNN/QueryNBFNet) của tác giả.

Chế độ inference hiện tại: KGC tail prediction cho (head, relation).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.graph_retriever.graph_adapter import (  # noqa:E402
    load_graph_bundle,
    resolve_entity_uid,
    resolve_relation_key,
)
from src.graph_retriever.gfm_bootstrap import bootstrap_gfmrag, disable_custom_rspmm  # noqa:E402
from src.graph_retriever.rel_features import ensure_rel_attr  # noqa:E402

logger = logging.getLogger(__name__)

CandidateScope = Literal["all", "entity_only"]
Direction = Literal["tail"]  # inference KGC dùng tail prediction mode


def _infer_entity_indices(
    bundle,
    *,
    candidate_scope: CandidateScope,
) -> torch.Tensor | None:
    if candidate_scope != "entity_only":
        return None
    nb = getattr(bundle.data, "nodes_by_type", None)
    if not isinstance(nb, dict):
        return None
    ents = nb.get("entity")
    if ents is None:
        logger.warning("Thiếu nodes_by_type['entity']; dùng toàn bộ node làm ứng viên.")
        return None
    return ents.detach().clone().cpu().long()


class GraphRetrieverInference:
    def __init__(
        self,
        tensor_dir: Path,
        checkpoint_path: Path,
        device: torch.device,
        *,
        build_relation_graph: bool = False,
        gfmrag_path: str | None = None,
        disable_rspmm: bool = True,
        relation_embedding_model: str = "BAAI/bge-base-en-v1.5",
        model_config_path: Path | None = None,
    ) -> None:
        bootstrap_gfmrag(gfmrag_path)
        if disable_rspmm:
            disable_custom_rspmm()

        self.bundle = load_graph_bundle(
            tensor_dir,
            build_relation_graph=build_relation_graph,
        )
        self.device = device
        self.graph = self.bundle.data
        self.mappings = self.bundle.mappings

        # QueryGNN cần rel_attr
        self.graph, _ = ensure_rel_attr(
            self.graph,
            rel2id_path=tensor_dir / "rel2id.json",
            embedding_model=relation_embedding_model,
        )
        self.graph = self.graph.to(device)

        from gfmrag.models.gfm_rag_v1 import QueryGNN

        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = payload.get("model") or payload.get("state_dict") or payload.get("model_state_dict")
        if state is None:
            raise ValueError("Checkpoint không có key `model` (state_dict).")

        if model_config_path is None:
            model_config_path = Path("configs/graph_retriever/kgc_gfm_training.yaml")
        cfg = OmegaConf.load(model_config_path)
        entity_model = instantiate(cfg.model.entity_model)
        self.model = QueryGNN(
            entity_model=entity_model,
            feat_dim=int(self.graph.rel_attr.size(1)),
        ).to(device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def retrieve(
        self,
        *,
        head_uid: str | None,
        relation_name: str,
        direction: Direction = "tail",
        top_k: int,
        candidate_scope: CandidateScope = "all",
    ) -> dict[str, Any]:
        """
        Predict tail given (head, r).
        """
        rk = resolve_relation_key(self.mappings, relation_name)
        if rk is None:
            raise ValueError(f"Relation không có trong relation2id: {relation_name}")

        rid = int(self.mappings.relation2id[rk])

        cand_filter = _infer_entity_indices(self.bundle, candidate_scope=candidate_scope)

        if head_uid is None:
            raise ValueError("Cần head_uid cho KGC tail prediction.")
        h_res = resolve_entity_uid(self.mappings, head_uid)
        if h_res is None:
            raise ValueError(f"Không map được head entity: {head_uid}")

        # Dùng tasks.all_negative để tạo batch [1, num_nodes, 3] rồi forward QueryGNN
        from gfmrag.models.ultra import tasks

        batch = torch.tensor(
            [[self.mappings.entity2id[h_res], 0, rid]],
            dtype=torch.long,
            device=self.device,
        )
        t_batch, _ = tasks.all_negative(self.graph, batch)
        logits = self.model(self.graph, t_batch)[0]

        scores = logits.float().detach().cpu()
        indices = torch.arange(scores.numel())

        if cand_filter is not None:
            logits_cf = scores[cand_filter]
            order = logits_cf.argsort(descending=True)[:top_k]
            ranked_ids = cand_filter[order]
            ranked_scores = logits_cf[order]
        else:
            order = scores.argsort(descending=True)
            ranked_ids = indices[order][:top_k]
            ranked_scores = scores[ranked_ids]

        top_entities: list[dict[str, Any]] = []
        for rank, (nid_tensor, scr) in enumerate(
            zip(ranked_ids.tolist(), ranked_scores.tolist()),
            start=1,
        ):
            uid = self.mappings.id2entity[str(int(nid_tensor))]
            row: dict[str, Any] = {
                "rank": rank,
                "node_id": int(nid_tensor),
                "uid": uid,
                "score": float(scr),
            }

            docs = self.mappings.entity_to_documents.get(uid)
            if docs:
                row["linked_documents"] = docs

            top_entities.append(row)

        top_documents = self._merge_document_scores(top_entities)

        return {
            "query": {
                "head_uid": head_uid,
                "relation_resolved": rk,
                "relation_id": rid,
                "direction": direction,
                "candidate_scope": candidate_scope,
            },
            "top_entities": top_entities,
            "top_documents": top_documents[:top_k],
        }

    def _merge_document_scores(self, top_entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best: dict[str, float] = {}
        sources: dict[str, str] = {}
        for row in top_entities:
            uid = row["uid"]
            score = row["score"]
            for cid in row.get("linked_documents", []) or []:
                sid = str(cid)
                if sid not in best or score > best[sid]:
                    best[sid] = score
                    sources[sid] = uid
        merged = [{"chunk_uid": c, "score": s, "best_entity_uid": sources[c]} for c, s in best.items()]
        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged


def cli() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    p = argparse.ArgumentParser(description="Graph retriever inference (QueryGNN/QueryNBFNet KGC)")
    p.add_argument("--tensor-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--gfmrag-path", type=str, default=None)
    p.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/graph_retriever/kgc_gfm_training.yaml"),
        help="Config YAML để khởi tạo đúng entity_model (QueryNBFNet).",
    )
    p.add_argument("--relation", type=str, required=True)
    p.add_argument("--head", type=str, default=None)
    p.add_argument(
        "--direction",
        choices=("tail",),
        default="tail",
        help="tail — dự đoán đuôi từ (head,r).",
    )
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument(
        "--candidates",
        choices=("all", "entity_only"),
        default="all",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--json-out", type=Path, default=None)
    args = p.parse_args()

    device = torch.device(args.device)
    inf = GraphRetrieverInference(
        args.tensor_dir,
        args.checkpoint,
        device,
        gfmrag_path=args.gfmrag_path,
        model_config_path=args.model_config,
    )

    payload = inf.retrieve(
        head_uid=args.head,
        relation_name=args.relation,
        direction=args.direction,
        top_k=args.top_k,
        candidate_scope=args.candidates,
    )
    txt = json.dumps(payload, indent=2, ensure_ascii=False)
    print(txt)
    if args.json_out:
        args.json_out.write_text(txt, encoding="utf-8")
        logger.info("Đã ghi %s", args.json_out)


if __name__ == "__main__":
    cli()
