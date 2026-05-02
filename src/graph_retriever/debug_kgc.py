"""Focused KGC debugging utilities.

This script intentionally uses tiny runs. It is for checking mechanics, not for
doing another long fine-tune.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, deque
from pathlib import Path
from types import MethodType
from typing import Any

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Data

from src.graph_retriever import tasks
from src.graph_retriever.gfm_bootstrap import bootstrap_gfmrag, disable_custom_rspmm
from src.graph_retriever.graph_adapter import load_graph_bundle
from src.graph_retriever.rel_features import ensure_rel_attr

logger = logging.getLogger(__name__)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_query_gnn(*, feat_dim: int, input_dim: int, hidden_dims: list[int]) -> nn.Module:
    from gfmrag.models.gfm_rag_v1 import QueryGNN
    from gfmrag.models.ultra.models import QueryNBFNet

    entity_model = QueryNBFNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        message_func="distmult",
        aggregate_func="sum",
        short_cut=True,
        layer_norm=True,
    )
    return QueryGNN(entity_model=entity_model, feat_dim=feat_dim)


def _toy_graph(device: torch.device) -> tuple[Data, dict[int, str]]:
    # A -r1-> B, B -r2-> C, A -r3-> C, plus inverse edges.
    target_edge_index = torch.tensor([[0, 1, 0], [1, 2, 2]], dtype=torch.long)
    target_edge_type = torch.tensor([0, 1, 2], dtype=torch.long)
    num_base_rel = 3
    edge_index = torch.cat([target_edge_index, target_edge_index.flip(0)], dim=1)
    edge_type = torch.cat([target_edge_type, target_edge_type + num_base_rel])
    rel_attr = torch.eye(num_base_rel * 2, 6).float()
    graph = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        target_edge_index=target_edge_index,
        target_edge_type=target_edge_type,
        num_nodes=3,
        num_relations=num_base_rel * 2,
        rel_attr=rel_attr,
        node_type=torch.tensor([0, 0, 0], dtype=torch.long),
        node_type_names=["entity"],
        nodes_by_type={"entity": torch.arange(3, dtype=torch.long)},
    )
    graph = tasks.build_relation_graph(graph)
    return graph.to(device), {0: "A", 1: "B", 2: "C"}


def _triples_from_graph(graph: Data) -> torch.Tensor:
    return torch.cat(
        [graph.target_edge_index, graph.target_edge_type.unsqueeze(0)], dim=0
    ).t()


def _install_remove_easy_logger(model: nn.Module, *, enabled: bool) -> None:
    original = model.entity_model.remove_easy_edges

    def logged_remove_easy_edges(self: Any, data: Data, h_index, t_index, r_index=None):
        if not enabled:
            print("remove_easy_edges: disabled, removed=0")
            return data
        before = int(data.edge_index.size(1))
        out = original(data, h_index, t_index, r_index)
        after = int(out.edge_index.size(1))
        print(
            "remove_easy_edges: enabled "
            f"before={before} after={after} removed={before - after} "
            f"batch_edges={h_index.numel()}"
        )
        return out

    model.entity_model.remove_easy_edges = MethodType(logged_remove_easy_edges, model.entity_model)


def _bce_loss(model: nn.Module, graph: Data, batch: torch.Tensor, *, num_negative: int) -> torch.Tensor:
    sampled = tasks.negative_sampling(graph, batch, num_negative, strict=True)
    pred = model(graph, sampled)
    target = torch.zeros_like(pred)
    target[:, 0] = 1
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    neg_weight = torch.ones_like(pred)
    with torch.no_grad():
        neg_weight[:, 1:] = F.softmax(pred[:, 1:], dim=-1)
    return ((loss * neg_weight).sum(dim=-1) / neg_weight.sum(dim=-1)).mean()


@torch.no_grad()
def _log_sample_logits(model: nn.Module, graph: Data, triples: torch.Tensor, *, num_negative: int, tag: str) -> None:
    model.eval()
    batch = triples[:2]
    sampled = tasks.negative_sampling(graph, batch, num_negative, strict=True)
    print(f"{tag}: negative_sampling batch shape={tuple(sampled.shape)}")
    print(f"{tag}: positive column correct={bool(torch.equal(sampled[:, 0, :], batch))}")
    print(f"{tag}: positive triples col0={sampled[:, 0, :].detach().cpu().tolist()}")
    pred = model(graph, sampled)
    print(f"{tag}: logits positive={pred[:, 0].detach().cpu().tolist()}")
    print(f"{tag}: logits negatives={pred[:, 1:].detach().cpu().tolist()}")
    model.train()


@torch.no_grad()
def _eval_ranking(
    model: nn.Module,
    graph: Data,
    triples: torch.Tensor,
    *,
    typed: bool = False,
) -> dict[str, float]:
    model.eval()
    filtered = Data(
        edge_index=graph.target_edge_index,
        edge_type=graph.target_edge_type,
        num_nodes=int(graph.num_nodes),
    ).to(graph.edge_index.device)
    ranks: list[torch.Tensor] = []
    for tri in triples:
        batch = tri.view(1, 3)
        t_batch, h_batch = tasks.all_negative(graph, batch)
        t_pred = model(graph, t_batch)
        h_pred = model(graph, h_batch)
        t_mask, h_mask = tasks.strict_negative_mask(filtered, batch)
        if typed and hasattr(graph, "node_type"):
            h, t, _r = [int(x) for x in tri.detach().cpu().tolist()]
            tail_type = int(graph.node_type[t].item())
            head_type = int(graph.node_type[h].item())
            t_mask &= graph.node_type.view(1, -1).to(t_mask.device).eq(tail_type)
            h_mask &= graph.node_type.view(1, -1).to(h_mask.device).eq(head_type)
        pos_h, pos_t, _pos_r = batch.t()
        ranks.append(tasks.compute_ranking(t_pred, pos_t, t_mask).detach().cpu())
        ranks.append(tasks.compute_ranking(h_pred, pos_h, h_mask).detach().cpu())
    rank = torch.cat(ranks).float()
    model.train()
    return {
        "mr": float(rank.mean().item()),
        "mrr": float((1 / rank).mean().item()),
        "hits@1": float((rank <= 1).float().mean().item()),
        "hits@3": float((rank <= 3).float().mean().item()),
        "hits@10": float((rank <= 10).float().mean().item()),
    }


def run_toy_debug(*, remove_easy_edges: bool, steps: int, seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    device = _device()
    graph, id2node = _toy_graph(device)
    triples = _triples_from_graph(graph).to(device)
    model = _make_query_gnn(feat_dim=6, input_dim=16, hidden_dims=[16, 16]).to(device)
    _install_remove_easy_logger(model, enabled=remove_easy_edges)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    print("\n=== TOY GRAPH ===")
    print(f"device={device} remove_easy_edges={remove_easy_edges}")
    print(f"nodes={id2node}")
    print(f"target triples={triples.detach().cpu().tolist()}")
    print(
        "relation_graph="
        f"nodes={graph.relation_graph.num_nodes} "
        f"rels={graph.relation_graph.num_relations} "
        f"edges={graph.relation_graph.edge_index.size(1)}"
    )
    _log_sample_logits(model, graph, triples, num_negative=2, tag="before")
    print(f"before ranking={_eval_ranking(model, graph, triples)}")

    losses: list[float] = []
    for step in range(1, steps + 1):
        # Keep the batch even: first row is tail-corrupted, second head-corrupted.
        batch = triples[torch.tensor([(step - 1) % 3, step % 3], device=device)]
        loss = _bce_loss(model, graph, batch, num_negative=2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        if step <= 5 or step % 25 == 0:
            print(f"step={step:03d} loss={losses[-1]:.6f}")

    print(
        f"loss first5={sum(losses[:5]) / min(5, len(losses)):.6f} "
        f"last5={sum(losses[-5:]) / min(5, len(losses)):.6f}"
    )
    _log_sample_logits(model, graph, triples, num_negative=2, tag="after")
    print(f"after ranking={_eval_ranking(model, graph, triples)}")


def _load_real_model_and_graph(
    *,
    config: Path,
    checkpoint: Path,
    eval_sample: int,
    seed: int,
) -> tuple[nn.Module, Data, torch.Tensor]:
    cfg = OmegaConf.load(config)
    bundle = load_graph_bundle(Path(str(cfg.graph.tensor_dir)))
    graph, feat_dim = ensure_rel_attr(
        bundle.data,
        rel2id_path=Path(str(cfg.graph.tensor_dir)) / "rel2id.json",
        embedding_model=str(cfg.graph.relation_embedding_model),
        embedding_device="cpu",
        embedding_batch_size=int(cfg.graph.get("relation_embedding_batch_size", 32)),
        force=False,
    )
    device = _device()
    graph = graph.to(device)
    model = _make_query_gnn(
        feat_dim=int(feat_dim),
        input_dim=int(cfg.model.entity_model.input_dim),
        hidden_dims=list(cfg.model.entity_model.hidden_dims),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    state = payload.get("model") or payload.get("state_dict") or payload.get("model_state_dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"checkpoint={checkpoint} missing={len(missing)} unexpected={len(unexpected)}")

    triples = _triples_from_graph(graph).to(device)
    if eval_sample and eval_sample < triples.size(0):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randperm(triples.size(0), generator=generator)[:eval_sample].to(device)
        triples = triples[idx]
    return model, graph, triples


def run_real_typed_eval(*, config: Path, checkpoint: Path, eval_sample: int, seed: int) -> None:
    model, graph, triples = _load_real_model_and_graph(
        config=config, checkpoint=checkpoint, eval_sample=eval_sample, seed=seed
    )
    print("\n=== REAL GRAPH TYPED EVAL ===")
    print(f"eval triples={triples.size(0)}")
    print(f"all-node={_eval_ranking(model, graph, triples, typed=False)}")
    print(f"typed={_eval_ranking(model, graph, triples, typed=True)}")


def _connected_components(num_nodes: int, edge_index: torch.Tensor) -> list[int]:
    adj = [[] for _ in range(num_nodes)]
    for h, t in edge_index.t().tolist():
        adj[h].append(t)
        adj[t].append(h)
    seen = [False] * num_nodes
    sizes: list[int] = []
    for start in range(num_nodes):
        if seen[start]:
            continue
        seen[start] = True
        q = deque([start])
        size = 0
        while q:
            node = q.popleft()
            size += 1
            for nxt in adj[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    q.append(nxt)
        sizes.append(size)
    return sizes


def _has_path_after_removing_direct(
    adj: list[list[tuple[int, int]]],
    h: int,
    t: int,
    r: int,
    *,
    base_rel: int,
    max_hops: int,
) -> bool:
    blocked = {(h, t, r), (t, h, r + base_rel)}
    q = deque([(h, 0)])
    seen = {h}
    while q:
        node, depth = q.popleft()
        if depth >= max_hops:
            continue
        for nxt, rel in adj[node]:
            if (node, nxt, rel) in blocked:
                continue
            if nxt == t:
                return True
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, depth + 1))
    return False


def run_real_distribution(*, tensor_dir: Path, path_sample: int, seed: int) -> None:
    graph = torch.load(tensor_dir / "graph.pt", map_location="cpu", weights_only=False)
    rel2id = json.loads((tensor_dir / "rel2id.json").read_text(encoding="utf-8"))
    id2rel = {int(v): k for k, v in rel2id.items()}
    triples = _triples_from_graph(graph).cpu()
    rel_counts = Counter(int(r) for r in triples[:, 2].tolist())
    print("\n=== REAL GRAPH DISTRIBUTION ===")
    print("top relation counts:")
    for rid, count in rel_counts.most_common(12):
        mask = triples[:, 2].eq(rid)
        heads = int(triples[mask, 0].unique().numel())
        tails = int(triples[mask, 1].unique().numel())
        print(f"  {rid:02d} {id2rel.get(rid, str(rid))}: count={count} heads={heads} tails={tails}")

    deg = torch.bincount(graph.edge_index.flatten(), minlength=int(graph.num_nodes)).float()
    print(
        "degree: "
        f"median={deg.median().item():.1f} mean={deg.mean().item():.2f} "
        f"max={deg.max().item():.0f} zero={int((deg == 0).sum().item())}"
    )
    comps = _connected_components(int(graph.num_nodes), graph.edge_index.cpu())
    comps_sorted = sorted(comps, reverse=True)
    print(
        f"connected_components={len(comps_sorted)} "
        f"largest={comps_sorted[:5]} isolated={sum(1 for x in comps_sorted if x == 1)}"
    )

    base_rel = int(graph.num_relations) // 2
    adj: list[list[tuple[int, int]]] = [[] for _ in range(int(graph.num_nodes))]
    for h, t, r in zip(graph.edge_index[0].tolist(), graph.edge_index[1].tolist(), graph.edge_type.tolist(), strict=False):
        adj[int(h)].append((int(t), int(r)))

    if path_sample and path_sample < triples.size(0):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        sample_idx = torch.randperm(triples.size(0), generator=generator)[:path_sample]
        triples_for_path = triples[sample_idx]
        sample_note = f"sample={path_sample}/{triples.size(0)}"
    else:
        triples_for_path = triples
        sample_note = f"all={triples.size(0)}"
    ok = 0
    for h, t, r in triples_for_path.tolist():
        ok += int(
            _has_path_after_removing_direct(
                adj, int(h), int(t), int(r), base_rel=base_rel, max_hops=6
            )
        )
    pct = ok / max(1, triples_for_path.size(0)) * 100
    print(f"path<=6 after removing direct edge: {ok}/{triples_for_path.size(0)} = {pct:.2f}% ({sample_note})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused KGC debug checks.")
    parser.add_argument("--config", type=Path, default=Path("configs/graph_retriever/kgc_gfm_training.yaml"))
    parser.add_argument("--tensor-dir", type=Path, default=Path("data/graph_tensor"))
    parser.add_argument("--checkpoint", type=Path, default=Path("model/model.pth"))
    parser.add_argument("--toy-steps", type=int, default=150)
    parser.add_argument("--eval-sample", type=int, default=50)
    parser.add_argument("--path-sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    bootstrap_gfmrag(None)
    disable_custom_rspmm()
    args = parse_args()
    run_toy_debug(remove_easy_edges=True, steps=args.toy_steps, seed=args.seed)
    run_toy_debug(remove_easy_edges=False, steps=args.toy_steps, seed=args.seed)
    run_real_typed_eval(
        config=args.config,
        checkpoint=args.checkpoint,
        eval_sample=args.eval_sample,
        seed=args.seed,
    )
    run_real_distribution(tensor_dir=args.tensor_dir, path_sample=args.path_sample, seed=args.seed)


if __name__ == "__main__":
    main()
