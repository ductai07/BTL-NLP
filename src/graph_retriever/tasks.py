"""Logic negative sampling và ranking của tác giả (GFM-RAG / ULTRA), giữ để bám đúng hành vi."""
# mypy: ignore-errors

from functools import reduce

import torch
from torch_geometric.data import Data


def edge_match(edge_index: torch.Tensor, query_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    base = edge_index.max(dim=1)[0] + 1
    assert reduce(int.__mul__, base.tolist()) < torch.iinfo(torch.long).max
    scale = base.cumprod(0)
    scale = scale[-1] // scale

    edge_hash = (edge_index * scale.unsqueeze(-1)).sum(dim=0)
    edge_hash, order = edge_hash.sort()
    query_hash = (query_index * scale.unsqueeze(-1)).sum(dim=0)

    start = torch.bucketize(query_hash, edge_hash)
    end = torch.bucketize(query_hash, edge_hash, right=True)
    num_match = end - start

    offset = num_match.cumsum(0) - num_match
    range_arr = torch.arange(num_match.sum(), device=edge_index.device)
    range_arr = range_arr + (start - offset).repeat_interleave(num_match)

    return order[range_arr], num_match


def negative_sampling(
    data: Data,
    batch: torch.Tensor,
    num_negative: int,
    strict: bool = True,
) -> torch.Tensor:
    batch_size = len(batch)
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    if strict:
        t_mask, h_mask = strict_negative_mask(data, batch)
        t_mask = t_mask[: batch_size // 2]
        neg_t_candidate = t_mask.nonzero()[:, 1]
        num_t_candidate = t_mask.sum(dim=-1)
        rand = torch.rand(len(t_mask), num_negative, device=batch.device)
        index = (rand * num_t_candidate.unsqueeze(-1)).long()
        index = index + (num_t_candidate.cumsum(0) - num_t_candidate).unsqueeze(-1)
        neg_t_index = neg_t_candidate[index]

        h_mask = h_mask[batch_size // 2 :]
        neg_h_candidate = h_mask.nonzero()[:, 1]
        num_h_candidate = h_mask.sum(dim=-1)
        rand = torch.rand(len(h_mask), num_negative, device=batch.device)
        index = (rand * num_h_candidate.unsqueeze(-1)).long()
        index = index + (num_h_candidate.cumsum(0) - num_h_candidate).unsqueeze(-1)
        neg_h_index = neg_h_candidate[index]
    else:
        neg_index = torch.randint(
            data.num_nodes, (batch_size, num_negative), device=batch.device
        )
        neg_t_index, neg_h_index = (
            neg_index[: batch_size // 2],
            neg_index[batch_size // 2 :],
        )

    h_index = pos_h_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index = pos_t_index.unsqueeze(-1).repeat(1, num_negative + 1)
    r_index = pos_r_index.unsqueeze(-1).repeat(1, num_negative + 1)
    t_index[: batch_size // 2, 1:] = neg_t_index
    h_index[batch_size // 2 :, 1:] = neg_h_index

    return torch.stack([h_index, t_index, r_index], dim=-1)


def all_negative(data: Data, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos_h_index, pos_t_index, pos_r_index = batch.t()
    r_index = pos_r_index.unsqueeze(-1).expand(-1, data.num_nodes)
    all_index = torch.arange(data.num_nodes, device=batch.device)
    h_index, t_index = torch.meshgrid(pos_h_index, all_index, indexing="ij")
    t_batch = torch.stack([h_index, t_index, r_index], dim=-1)
    all_index = torch.arange(data.num_nodes, device=batch.device)
    t_index, h_index = torch.meshgrid(pos_t_index, all_index, indexing="ij")
    h_batch = torch.stack([h_index, t_index, r_index], dim=-1)

    return t_batch, h_batch


def strict_negative_mask(data: Data, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pos_h_index, pos_t_index, pos_r_index = batch.t()

    edge_index = torch.stack([data.edge_index[0], data.edge_type])
    query_index = torch.stack([pos_h_index, pos_r_index])
    edge_id, num_t_truth = edge_match(edge_index, query_index)
    t_truth_index = data.edge_index[1, edge_id]
    sample_id = torch.arange(len(num_t_truth), device=batch.device).repeat_interleave(
        num_t_truth
    )
    t_mask = torch.ones(
        len(num_t_truth), data.num_nodes, dtype=torch.bool, device=batch.device
    )
    t_mask[sample_id, t_truth_index] = 0
    t_mask.scatter_(1, pos_t_index.unsqueeze(-1), 0)

    edge_index = torch.stack([data.edge_index[1], data.edge_type])
    query_index = torch.stack([pos_t_index, pos_r_index])
    edge_id, num_h_truth = edge_match(edge_index, query_index)
    h_truth_index = data.edge_index[0, edge_id]
    sample_id = torch.arange(len(num_h_truth), device=batch.device).repeat_interleave(
        num_h_truth
    )
    h_mask = torch.ones(
        len(num_h_truth), data.num_nodes, dtype=torch.bool, device=batch.device
    )
    h_mask[sample_id, h_truth_index] = 0
    h_mask.scatter_(1, pos_h_index.unsqueeze(-1), 0)

    return t_mask, h_mask


def compute_ranking(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    pos_pred = pred.gather(-1, target.unsqueeze(-1))
    if mask is not None:
        ranking = torch.sum((pos_pred <= pred) & mask, dim=-1) + 1
    else:
        ranking = torch.sum(pos_pred <= pred, dim=-1) + 1
    return ranking


def build_relation_graph(graph: Data) -> Data:
    """Xây relation graph trên toàn bộ cạnh (đã gồm inverse) — giống tác giả."""
    edge_index, edge_type = graph.edge_index, graph.edge_type
    num_nodes, num_rels = graph.num_nodes, graph.num_relations
    device = edge_index.device

    Eh = torch.vstack([edge_index[0], edge_type]).T.unique(dim=0)
    Dh = torch.zeros(Eh[:, 0].max() + 1, device=Eh.device, dtype=Eh.dtype)
    Dh.scatter_add_(dim=0, index=Eh[:, 0], src=torch.ones_like(Eh[:, 0]))

    EhT = torch.sparse_coo_tensor(
        torch.flip(Eh, dims=[1]).T,
        torch.ones(Eh.shape[0], device=device) / Dh[Eh[:, 0]],
        (num_rels, num_nodes),
    )
    Eh = torch.sparse_coo_tensor(
        Eh.T, torch.ones(Eh.shape[0], device=device), (num_nodes, num_rels)
    )
    Et = torch.vstack([edge_index[1], edge_type]).T.unique(dim=0)

    Dt = torch.zeros(Et[:, 0].max() + 1, device=Et.device, dtype=Et.dtype)
    Dt.scatter_add_(dim=0, index=Et[:, 0], src=torch.ones_like(Et[:, 0]))
    assert not (Dt[Et[:, 0]] == 0).any()

    EtT = torch.sparse_coo_tensor(
        torch.flip(Et, dims=[1]).T,
        torch.ones(Et.shape[0], device=device) / Dt[Et[:, 0]],
        (num_rels, num_nodes),
    )
    Et = torch.sparse_coo_tensor(
        Et.T, torch.ones(Et.shape[0], device=device), (num_nodes, num_rels)
    )

    Ahh = torch.sparse.mm(EhT, Eh).coalesce()
    Att = torch.sparse.mm(EtT, Et).coalesce()
    Aht = torch.sparse.mm(EhT, Et).coalesce()
    Ath = torch.sparse.mm(EtT, Eh).coalesce()

    hh_edges = torch.cat(
        [
            Ahh.indices().T,
            torch.zeros(Ahh.indices().T.shape[0], 1, dtype=torch.long).fill_(0),
        ],
        dim=1,
    )
    tt_edges = torch.cat(
        [
            Att.indices().T,
            torch.zeros(Att.indices().T.shape[0], 1, dtype=torch.long).fill_(1),
        ],
        dim=1,
    )
    ht_edges = torch.cat(
        [
            Aht.indices().T,
            torch.zeros(Aht.indices().T.shape[0], 1, dtype=torch.long).fill_(2),
        ],
        dim=1,
    )
    th_edges = torch.cat(
        [
            Ath.indices().T,
            torch.zeros(Ath.indices().T.shape[0], 1, dtype=torch.long).fill_(3),
        ],
        dim=1,
    )

    rel_graph = Data(
        edge_index=torch.cat(
            [
                hh_edges[:, [0, 1]].T,
                tt_edges[:, [0, 1]].T,
                ht_edges[:, [0, 1]].T,
                th_edges[:, [0, 1]].T,
            ],
            dim=1,
        ),
        edge_type=torch.cat(
            [hh_edges[:, 2], tt_edges[:, 2], ht_edges[:, 2], th_edges[:, 2]], dim=0
        ),
        num_nodes=num_rels,
        num_relations=4,
    )

    graph.relation_graph = rel_graph
    return graph
