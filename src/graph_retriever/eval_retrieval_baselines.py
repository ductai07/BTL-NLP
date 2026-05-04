"""
Đánh giá retrieval baseline kiểu HippoRAG và LightRAG trên Stage 2 data.

Mục tiêu:
- Không phụ thuộc chạy code trực tiếp từ repo khác.
- Tái sử dụng graph + mapping + metric trong project hiện tại.

Chạy ví dụ:
  python src/graph_retriever/eval_retrieval_baselines.py ^
      --config configs/graph_retriever/stage2_sft.yaml ^
      --data data/qa/test_qa_stage2.json ^
      --output outputs/graph_retriever/baseline_eval_results.json ^
      --top-k 20
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env", override=False)
except ImportError:
    pass

from src.graph_retriever.graph_adapter import build_target_to_other_types, load_graph_bundle
from src.graph_retriever.gfm_bootstrap import bootstrap_gfmrag, disable_custom_rspmm

logger = logging.getLogger(__name__)

METRICS = [
    "mrr",
    "hits@1",
    "hits@2",
    "hits@5",
    "hits@10",
    "hits@20",
    "recall@5",
    "recall@10",
    "recall@20",
]


def _compute_metrics(
    all_pred: torch.Tensor,
    all_tgt: torch.Tensor,
    metrics: list[str],
    type_name: str,
) -> dict[str, float]:
    """Tính metric giống pipeline gốc của GFM-RAG."""
    from gfmrag.utils import batch_evaluate, evaluate as gfm_evaluate
    from gfmrag.models.ultra import query_utils

    has_pos = all_tgt.sum(dim=-1) > 0
    if not has_pos.any():
        logger.warning("[%s] Không có sample positive target.", type_name)
        return {m: 0.0 for m in metrics}

    pred_f = all_pred[has_pos]
    tgt_f = all_tgt[has_pos].bool()

    preds_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    targets_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    for i in range(0, pred_f.size(0), 64):
        p = pred_f[i : i + 64]
        t = tgt_f[i : i + 64]
        node_ranking, target_ranking = batch_evaluate(p, t)
        num_pred = (torch.sigmoid(p) > 0.5).sum(dim=-1)
        num_target = t.sum(dim=-1)
        preds_list.append((node_ranking, num_pred))
        targets_list.append((target_ranking, num_target))

    node_pred = query_utils.cat(preds_list)
    node_target = query_utils.cat(targets_list)
    return gfm_evaluate(node_pred, node_target, metrics)


def _load_node_texts(graph_dir: Path, node_type: str) -> dict[str, str]:
    """Đọc nội dung text đại diện cho node_type từ nodes.csv."""
    nodes_path = graph_dir / "nodes.csv"
    if not nodes_path.exists():
        raise FileNotFoundError(f"Thiếu file: {nodes_path}")

    df = pd.read_csv(nodes_path, keep_default_na=False)
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        if str(row.get("type", "")) != node_type:
            continue
        uid = str(row.get("uid", ""))
        if not uid:
            continue

        attrs_raw = row.get("attributes", "")
        text = str(row.get("name", ""))
        if isinstance(attrs_raw, str) and attrs_raw:
            try:
                attrs = ast.literal_eval(attrs_raw)
                if isinstance(attrs, dict):
                    text = str(
                        attrs.get("text_preview")
                        or attrs.get("text")
                        or attrs.get("content")
                        or text
                    )
            except Exception:
                pass
        out[uid] = text
    return out


def _build_start_mask(samples: list[dict[str, Any]], node2id: dict[str, int], num_nodes: int) -> torch.Tensor:
    """Tạo start mask toàn cục [B, N] từ start_nodes.entity."""
    start_mask = torch.zeros((len(samples), num_nodes), dtype=torch.float32)
    for i, sample in enumerate(samples):
        for uid in sample.get("start_nodes", {}).get("entity", []):
            nid = node2id.get(uid)
            if nid is not None:
                start_mask[i, nid] = 1.0
    return start_mask


def _build_targets(
    samples: list[dict[str, Any]],
    node2id: dict[str, int],
    chunk_nodes: torch.Tensor,
    entity_nodes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tạo target binary cho chunk/entity."""
    bsz = len(samples)
    chunk_index = {int(n.item()): i for i, n in enumerate(chunk_nodes)}
    entity_index = {int(n.item()): i for i, n in enumerate(entity_nodes)}
    tgt_chunk = torch.zeros((bsz, len(chunk_nodes)), dtype=torch.float32)
    tgt_entity = torch.zeros((bsz, len(entity_nodes)), dtype=torch.float32)

    for i, sample in enumerate(samples):
        for uid in sample.get("target_nodes", {}).get("chunk", []):
            nid = node2id.get(uid)
            if nid is not None and nid in chunk_index:
                tgt_chunk[i, chunk_index[nid]] = 1.0
        for uid in sample.get("target_nodes", {}).get("entity", []):
            nid = node2id.get(uid)
            if nid is not None and nid in entity_index:
                tgt_entity[i, entity_index[nid]] = 1.0
    return tgt_chunk, tgt_entity


def _hipporag_scores(
    start_mask: torch.Tensor,
    ent2chunk_sparse: torch.Tensor,
    alpha: float = 0.7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    HippoRAG-lite:
    1) entity anchor -> chunk
    2) chunk -> entity feedback
    3) entity propagated -> chunk
    """
    # [B, N] x [N, N] -> [B, N]
    chunk_seed = torch.sparse.mm(start_mask, ent2chunk_sparse)
    ent_feedback = torch.sparse.mm(chunk_seed, ent2chunk_sparse.transpose(0, 1))

    # Chuẩn hóa theo mỗi sample để tránh sample có nhiều start node bị lệch biên độ.
    denom = ent_feedback.sum(dim=-1, keepdim=True).clamp_min(1.0)
    ent_feedback = ent_feedback / denom
    ent_prop = start_mask + alpha * ent_feedback
    chunk_final = torch.sparse.mm(ent_prop, ent2chunk_sparse)
    return ent_prop, chunk_final


def _lightrag_scores(
    questions: list[str],
    target_uids: list[str],
    target_text_map: dict[str, str],
    emb_cache: Path | None,
    target_emb_cache: Path | None,
    text_emb_model: str,
    emb_device: str | None,
) -> torch.Tensor:
    """LightRAG-lite: dense retrieval query->target theo cosine."""
    from sentence_transformers import SentenceTransformer

    if emb_cache is not None and emb_cache.exists():
        q_emb = torch.load(emb_cache, map_location="cpu", weights_only=False).float()
        if q_emb.size(0) < len(questions):
            raise ValueError("Question embedding cache không đủ số lượng sample.")
        q_emb = q_emb[: len(questions)]
    else:
        model = SentenceTransformer(text_emb_model, device=emb_device)
        q_np = model.encode(
            questions,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        q_emb = torch.tensor(q_np, dtype=torch.float32)

    target_emb: torch.Tensor
    if target_emb_cache is not None and target_emb_cache.exists():
        target_emb = torch.load(target_emb_cache, map_location="cpu", weights_only=False).float()
        if target_emb.size(0) != len(target_uids):
            raise ValueError("Target embedding cache không khớp số lượng target nodes.")
    else:
        model = SentenceTransformer(text_emb_model, device=emb_device)
        target_texts = [target_text_map.get(uid, uid) for uid in target_uids]
        target_np = model.encode(
            target_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        target_emb = torch.tensor(target_np, dtype=torch.float32)
        if target_emb_cache is not None:
            target_emb_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(target_emb, target_emb_cache)

    # Cosine similarity vì embeddings đã normalize.
    return q_emb @ target_emb.transpose(0, 1)


def _topk_chunk_predictions(
    chunk_scores: torch.Tensor,
    chunk_uids: list[str],
    top_k: int,
) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    k = min(top_k, len(chunk_uids))
    for i in range(chunk_scores.size(0)):
        vals, idx = torch.topk(chunk_scores[i], k=k)
        out.append(
            [
                {"uid": chunk_uids[int(j)], "score": float(v)}
                for v, j in zip(vals.tolist(), idx.tolist())
            ]
        )
    return out


def _topk_entity_predictions(
    entity_scores: torch.Tensor,
    entity_uids: list[str],
    top_k: int,
) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    k = min(top_k, len(entity_uids))
    for i in range(entity_scores.size(0)):
        vals, idx = torch.topk(entity_scores[i], k=k)
        out.append(
            [
                {"uid": entity_uids[int(j)], "score": float(v)}
                for v, j in zip(vals.tolist(), idx.tolist())
            ]
        )
    return out


def _build_hipporag_full_resources(
    graph_dir: Path,
    mention_relation_key: str,
) -> dict[str, Any]:
    """Load nodes/edges và tạo resources cần cho HippoRAG-full (no-LLM rerank)."""
    nodes_df = pd.read_csv(graph_dir / "nodes.csv", keep_default_na=False)
    edges_df = pd.read_csv(graph_dir / "edges.csv", keep_default_na=False)

    uid2name: dict[str, str] = {}
    uid2type: dict[str, str] = {}
    for _, row in nodes_df.iterrows():
        uid = str(row.get("uid", ""))
        if not uid:
            continue
        uid2name[uid] = str(row.get("name", uid))
        uid2type[uid] = str(row.get("type", ""))

    # entity -> chunk mentions
    ent_to_chunks: dict[str, set[str]] = {}
    mention_df = edges_df[edges_df["relation"] == mention_relation_key]
    for _, row in mention_df.iterrows():
        src = str(row["source"])
        tgt = str(row["target"])
        if uid2type.get(src) != "entity" or uid2type.get(tgt) != "chunk":
            continue
        ent_to_chunks.setdefault(src, set()).add(tgt)

    # facts: bo mention edges, uu tien facts co 2 dau mut la entity
    fact_df = edges_df[edges_df["relation"] != mention_relation_key].copy()
    keep_mask = fact_df["source"].map(uid2type).eq("entity") & fact_df["target"].map(uid2type).eq("entity")
    fact_df = fact_df[keep_mask]

    facts: list[tuple[str, str, str]] = []
    fact_texts: list[str] = []
    for _, row in fact_df.iterrows():
        src_uid = str(row["source"])
        rel = str(row["relation"])
        tgt_uid = str(row["target"])
        src_name = uid2name.get(src_uid, src_uid)
        tgt_name = uid2name.get(tgt_uid, tgt_uid)
        facts.append((src_uid, rel, tgt_uid))
        fact_texts.append(f"{src_name} [SEP] {rel} [SEP] {tgt_name}")

    return {
        "uid2name": uid2name,
        "uid2type": uid2type,
        "ent_to_chunks": ent_to_chunks,
        "facts": facts,
        "fact_texts": fact_texts,
    }


def _hipporag_full_scores(
    questions: list[str],
    chunk_uids: list[str],
    entity_uids: list[str],
    chunk_scores_dense: torch.Tensor,
    resources: dict[str, Any],
    text_emb_model: str,
    emb_device: str | None,
    top_k: int,
    passage_node_weight: float = 0.05,
    llm_rerank: bool = False,
    llm_input_topk: int = 80,
    llm_output_topk: int = 20,
    llm_model: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    HippoRAG-full style (khong LLM rerank):
    - retrieve facts theo query/fact embeddings
    - aggregate score ve entity tu facts
    - merge them dense chunk score vao chunk nodes
    """
    from sentence_transformers import SentenceTransformer

    facts: list[tuple[str, str, str]] = resources["facts"]
    fact_texts: list[str] = resources["fact_texts"]
    ent_to_chunks: dict[str, set[str]] = resources["ent_to_chunks"]

    if len(facts) == 0:
        raise RuntimeError("HippoRAG-full: khong co fact nao (non-mention edges).")

    model = SentenceTransformer(text_emb_model, device=emb_device)
    q_np = model.encode(
        questions,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    f_np = model.encode(
        fact_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    q_emb = torch.tensor(q_np, dtype=torch.float32)
    f_emb = torch.tensor(f_np, dtype=torch.float32)
    fact_scores = q_emb @ f_emb.transpose(0, 1)  # [B, F]

    llm_client = None
    llm_base_url: str | None = None
    if llm_rerank:
        import requests as _requests

        api_key = (os.getenv("OPENAI_API_KEY") or os.getenv("NVIDIA_API_KEY") or "").strip()
        llm_base_url = (os.getenv("OPENAI_BASE_URL") or "").strip().rstrip("/") or "https://api.openai.com/v1"
        if not api_key:
            raise RuntimeError(
                "Thieu OPENAI_API_KEY hoac NVIDIA_API_KEY trong moi truong (co the them file .env o goc repo)."
            )
        llm_client = {"api_key": api_key, "base_url": llm_base_url, "_requests": _requests}
        llm_model = llm_model or os.getenv("OPENAI_MODEL", "openai/gpt-oss-20b")

    ent_pos = {uid: i for i, uid in enumerate(entity_uids)}
    chunk_pos = {uid: i for i, uid in enumerate(chunk_uids)}

    ent_scores = torch.zeros((len(questions), len(entity_uids)), dtype=torch.float32)
    chunk_scores = torch.zeros((len(questions), len(chunk_uids)), dtype=torch.float32)

    top_fact_k = min(max(1, top_k * 4), fact_scores.size(1))
    llm_input_topk = min(max(1, llm_input_topk), fact_scores.size(1))
    llm_output_topk = min(max(1, llm_output_topk), llm_input_topk)
    for i in range(len(questions)):
        vals, idxs = torch.topk(fact_scores[i], k=top_fact_k)
        selected_pairs = list(zip(vals.tolist(), idxs.tolist()))

        if llm_client is not None:
            cand_vals, cand_idxs = torch.topk(fact_scores[i], k=llm_input_topk)
            cand_pairs = list(zip(cand_vals.tolist(), cand_idxs.tolist()))
            fact_lines = []
            for j, (score, fid) in enumerate(cand_pairs, start=1):
                s_uid, rel, o_uid = facts[int(fid)]
                fact_lines.append(
                    f"{j}. ({resources['uid2name'].get(s_uid, s_uid)}) -[{rel}]-> ({resources['uid2name'].get(o_uid, o_uid)}) | score={score:.4f}"
                )
            prompt = (
                "Chon cac fact lien quan nhat de tra loi cau hoi.\n"
                f"Cau hoi: {questions[i]}\n"
                f"Tra ve JSON object duy nhat: {{\"top_indices\": [..]}} voi chi so 1-based, toi da {llm_output_topk} phan tu.\n"
                "Khong giai thich them.\n"
                "Danh sach fact:\n"
                + "\n".join(fact_lines)
            )
            try:
                _req = llm_client["_requests"]
                _payload: dict = {
                    "model": llm_model,
                    "messages": [
                        {"role": "system", "content": "Ban la bo loc fact cho retrieval graph."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": 300,
                    "stream": False,
                }
                if "gpt-oss" in (llm_model or ""):
                    _payload["reasoning_effort"] = "low"
                _resp = _req.post(
                    f"{llm_client['base_url']}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {llm_client['api_key']}",
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    json=_payload,
                    timeout=60,
                )
                _resp.raise_for_status()
                content = (_resp.json()["choices"][0]["message"].get("content") or "").strip()
                match = re.search(r"\{[\s\S]*\}", content)
                if match:
                    obj = json.loads(match.group(0))
                    picks = obj.get("top_indices", [])
                    reranked: list[tuple[float, int]] = []
                    for p in picks:
                        if isinstance(p, int) and 1 <= p <= len(cand_pairs):
                            reranked.append(cand_pairs[p - 1])
                    if len(reranked) > 0:
                        selected_pairs = reranked[:llm_output_topk]
            except Exception as exc:
                logger.warning("LLM rerank fail sample %d: %s", i, exc)

        phrase_acc: dict[str, list[float]] = {}
        for v, fid in selected_pairs:
            s_uid, _, o_uid = facts[int(fid)]
            for ent_uid in (s_uid, o_uid):
                w = float(v)
                deg = len(ent_to_chunks.get(ent_uid, set()))
                if deg > 0:
                    w /= deg
                phrase_acc.setdefault(ent_uid, []).append(w)

        # score entity tu facts
        for ent_uid, ws in phrase_acc.items():
            pos = ent_pos.get(ent_uid)
            if pos is not None:
                ent_scores[i, pos] = float(np.mean(ws))

        # score chunk tu entity facts (entity -> mentioned chunks)
        for ent_uid, ws in phrase_acc.items():
            s = float(np.mean(ws))
            for chunk_uid in ent_to_chunks.get(ent_uid, set()):
                cpos = chunk_pos.get(chunk_uid)
                if cpos is not None:
                    chunk_scores[i, cpos] += s

        # merge dense chunk retrieval nhu HippoRAG2 constructor
        chunk_scores[i] += passage_node_weight * chunk_scores_dense[i]

    return ent_scores, chunk_scores


@torch.no_grad()
def run_eval(
    cfg: dict[str, Any],
    data_path: Path,
    output_path: Path,
    top_k: int,
    hipporag_alpha: float,
    methods: list[str],
    max_samples: int | None = None,
    hipporag_full_llm_rerank: bool = False,
    hipporag_full_llm_input_topk: int = 80,
    hipporag_full_llm_output_topk: int = 20,
    hipporag_full_llm_model: str | None = None,
) -> dict[str, Any]:
    tensor_dir = Path(str(cfg["graph"]["tensor_dir"]))
    graph_dir = Path(str(cfg["graph"]["graph_dir"]))
    data_cfg = cfg["data"]

    bundle = load_graph_bundle(
        tensor_dir,
        mention_relation_key=str(cfg["graph"].get("mention_relation_key", "is_mentioned_in")),
    )
    graph = bundle.data
    mappings = bundle.mappings

    with open(data_path, encoding="utf-8") as f:
        samples: list[dict[str, Any]] = json.load(f)
    if max_samples is not None and max_samples > 0:
        samples = samples[:max_samples]

    node2id = {str(k): int(v) for k, v in mappings.entity2id.items()}
    chunk_nodes = graph.nodes_by_type["chunk"].cpu()
    entity_nodes = graph.nodes_by_type["entity"].cpu()
    id2node = json.loads((tensor_dir / "id2node.json").read_text(encoding="utf-8"))
    chunk_uids = [id2node[str(int(n.item()))] for n in chunk_nodes]
    entity_uids = [id2node[str(int(n.item()))] for n in entity_nodes]

    start_mask = _build_start_mask(samples, node2id, int(graph.num_nodes))
    tgt_chunk, tgt_entity = _build_targets(samples, node2id, chunk_nodes, entity_nodes)
    questions = [s["question"] for s in samples]

    results: dict[str, Any] = {}

    if "hipporag" in methods:
        target_to_other = build_target_to_other_types(
            graph,
            mappings.relation2id,
            mention_key=str(cfg["graph"].get("mention_relation_key", "is_mentioned_in")),
        )
        if "chunk" not in target_to_other:
            raise RuntimeError("Không tìm được mapping entity->chunk cho HippoRAG.")
        ent2chunk = target_to_other["chunk"].coalesce().float()

        ent_scores_global, chunk_scores_global = _hipporag_scores(
            start_mask=start_mask,
            ent2chunk_sparse=ent2chunk,
            alpha=hipporag_alpha,
        )
        chunk_scores = chunk_scores_global[:, chunk_nodes]
        ent_scores = ent_scores_global[:, entity_nodes]

        chunk_metrics = _compute_metrics(chunk_scores, tgt_chunk, METRICS, "hipporag_chunk")
        entity_metrics = _compute_metrics(ent_scores, tgt_entity, METRICS, "hipporag_entity")
        results["hipporag"] = {
            "chunk_metrics": chunk_metrics,
            "entity_metrics": entity_metrics,
            "predicted_chunks": _topk_chunk_predictions(chunk_scores, chunk_uids, top_k),
            "predicted_entities": _topk_entity_predictions(ent_scores, entity_uids, top_k),
            "params": {"alpha": hipporag_alpha},
        }

    if "hipporag_full" in methods:
        chunk_text_map = _load_node_texts(graph_dir, node_type="chunk")
        chunk_emb_cache = Path(
            str(data_cfg.get("chunk_emb_cache", "data/qa/chunk_embeddings_lightrag.pt"))
        )
        emb_cache = Path(str(data_cfg["emb_cache"])) if data_cfg.get("emb_cache") else None
        chunk_scores_dense = _lightrag_scores(
            questions=questions,
            target_uids=chunk_uids,
            target_text_map=chunk_text_map,
            emb_cache=emb_cache,
            target_emb_cache=chunk_emb_cache,
            text_emb_model=str(data_cfg.get("text_emb_model", "BAAI/bge-base-en-v1.5")),
            emb_device=data_cfg.get("emb_device"),
        )
        resources = _build_hipporag_full_resources(
            graph_dir=graph_dir,
            mention_relation_key=str(cfg["graph"].get("mention_relation_key", "is_mentioned_in")),
        )
        ent_scores, chunk_scores = _hipporag_full_scores(
            questions=questions,
            chunk_uids=chunk_uids,
            entity_uids=entity_uids,
            chunk_scores_dense=chunk_scores_dense,
            resources=resources,
            text_emb_model=str(data_cfg.get("text_emb_model", "BAAI/bge-base-en-v1.5")),
            emb_device=data_cfg.get("emb_device"),
            top_k=top_k,
            llm_rerank=hipporag_full_llm_rerank,
            llm_input_topk=hipporag_full_llm_input_topk,
            llm_output_topk=hipporag_full_llm_output_topk,
            llm_model=hipporag_full_llm_model,
        )
        chunk_metrics = _compute_metrics(chunk_scores, tgt_chunk, METRICS, "hipporag_full_chunk")
        entity_metrics = _compute_metrics(ent_scores, tgt_entity, METRICS, "hipporag_full_entity")
        results["hipporag_full"] = {
            "chunk_metrics": chunk_metrics,
            "entity_metrics": entity_metrics,
            "predicted_chunks": _topk_chunk_predictions(chunk_scores, chunk_uids, top_k),
            "predicted_entities": _topk_entity_predictions(ent_scores, entity_uids, top_k),
            "params": {
                "passage_node_weight": 0.05,
                "fact_topk_multiplier": 4,
                "llm_rerank": hipporag_full_llm_rerank,
                "llm_input_topk": hipporag_full_llm_input_topk,
                "llm_output_topk": hipporag_full_llm_output_topk,
                "llm_model": hipporag_full_llm_model,
            },
        }

    if "lightrag" in methods:
        chunk_text_map = _load_node_texts(graph_dir, node_type="chunk")
        entity_text_map = _load_node_texts(graph_dir, node_type="entity")
        chunk_emb_cache = Path(
            str(
                data_cfg.get(
                    "chunk_emb_cache",
                    "data/qa/chunk_embeddings_lightrag.pt",
                )
            )
        )
        entity_emb_cache = Path(
            str(
                data_cfg.get(
                    "entity_emb_cache",
                    "data/qa/entity_embeddings_lightrag.pt",
                )
            )
        )
        emb_cache = Path(str(data_cfg["emb_cache"])) if data_cfg.get("emb_cache") else None
        chunk_scores = _lightrag_scores(
            questions=questions,
            target_uids=chunk_uids,
            target_text_map=chunk_text_map,
            emb_cache=emb_cache,
            target_emb_cache=chunk_emb_cache,
            text_emb_model=str(data_cfg.get("text_emb_model", "BAAI/bge-base-en-v1.5")),
            emb_device=data_cfg.get("emb_device"),
        )
        entity_scores = _lightrag_scores(
            questions=questions,
            target_uids=entity_uids,
            target_text_map=entity_text_map,
            emb_cache=emb_cache,
            target_emb_cache=entity_emb_cache,
            text_emb_model=str(data_cfg.get("text_emb_model", "BAAI/bge-base-en-v1.5")),
            emb_device=data_cfg.get("emb_device"),
        )
        chunk_metrics = _compute_metrics(chunk_scores, tgt_chunk, METRICS, "lightrag_chunk")
        entity_metrics = _compute_metrics(entity_scores, tgt_entity, METRICS, "lightrag_entity")
        results["lightrag"] = {
            "chunk_metrics": chunk_metrics,
            "entity_metrics": entity_metrics,
            "predicted_chunks": _topk_chunk_predictions(chunk_scores, chunk_uids, top_k),
            "predicted_entities": _topk_entity_predictions(entity_scores, entity_uids, top_k),
            "params": {"embedding_model": str(data_cfg.get("text_emb_model", "BAAI/bge-base-en-v1.5"))},
        }

    merged_predictions: list[dict[str, Any]] = []
    for i, sample in enumerate(samples):
        row = {
            "id": sample.get("id", i),
            "question": sample.get("question", ""),
            "ground_truth_chunks": sample.get("target_nodes", {}).get("chunk", []),
            "ground_truth_entities": sample.get("target_nodes", {}).get("entity", []),
            "predictions": {},
        }
        for method in results.keys():
            row["predictions"][method] = {
                "chunk": results[method]["predicted_chunks"][i],
                "entity": (
                    results[method]["predicted_entities"][i]
                    if "predicted_entities" in results[method]
                    else []
                ),
            }
        merged_predictions.append(row)

    output = {
        "data": str(data_path),
        "n_samples": len(samples),
        "n_samples_with_chunk_target": int((tgt_chunk.sum(dim=-1) > 0).sum().item()),
        "n_samples_with_entity_target": int((tgt_entity.sum(dim=-1) > 0).sum().item()),
        "chunk_count": len(chunk_uids),
        "entity_count": len(entity_uids),
        "methods": results,
        "per_sample": merged_predictions,
    }
    # Không cần giữ toàn bộ prediction lặp hai lần trong output.
    for method in output["methods"].values():
        method.pop("predicted_chunks", None)
        method.pop("predicted_entities", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info("Đã lưu baseline eval: %s", output_path)
    for method_name, m in results.items():
        logger.info("=== %s ===", method_name.upper())
        for k, v in sorted(m["chunk_metrics"].items()):
            logger.info("  chunk_%s: %.4f", k, v)
        if "entity_metrics" in m:
            for k, v in sorted(m["entity_metrics"].items()):
                logger.info("  entity_%s: %.4f", k, v)

    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HippoRAG/LightRAG baselines on Stage2 data.")
    p.add_argument("--config", type=Path, default=Path("configs/graph_retriever/stage2_sft.yaml"))
    p.add_argument("--data", type=Path, default=Path("data/qa/test_qa_stage2.json"))
    p.add_argument("--output", type=Path, default=Path("outputs/graph_retriever/baseline_eval_results.json"))
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-samples", type=int, default=None, help="Chay thu tren N sample dau.")
    p.add_argument("--hipporag-alpha", type=float, default=0.7)
    p.add_argument("--hipporag-full-llm-rerank", action="store_true")
    p.add_argument("--hipporag-full-llm-input-topk", type=int, default=80)
    p.add_argument("--hipporag-full-llm-output-topk", type=int, default=20)
    p.add_argument("--hipporag-full-llm-model", type=str, default=None)
    p.add_argument("--gfmrag-path", type=str, default=None)
    p.add_argument("--disable-custom-rspmm", action="store_true")
    p.add_argument(
        "--methods",
        type=str,
        default="hipporag,lightrag",
        help="Danh sách baseline, ví dụ: hipporag,lightrag",
    )
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
    bootstrap_gfmrag(ns.gfmrag_path or raw.get("gfmrag_path"))
    if ns.disable_custom_rspmm or bool(raw.get("disable_custom_rspmm", True)):
        disable_custom_rspmm()

    cfg = OmegaConf.to_container(raw, resolve=True)
    assert isinstance(cfg, dict)

    methods = [m.strip().lower() for m in ns.methods.split(",") if m.strip()]
    valid = {"hipporag", "hipporag_full", "lightrag"}
    invalid = [m for m in methods if m not in valid]
    if invalid:
        raise ValueError(f"Method không hợp lệ: {invalid}. Hợp lệ: {sorted(valid)}")
    if not methods:
        raise ValueError("Cần ít nhất 1 method để đánh giá.")

    run_eval(
        cfg=cfg,
        data_path=ns.data,
        output_path=ns.output,
        top_k=ns.top_k,
        hipporag_alpha=ns.hipporag_alpha,
        methods=methods,
        max_samples=ns.max_samples,
        hipporag_full_llm_rerank=bool(ns.hipporag_full_llm_rerank),
        hipporag_full_llm_input_topk=int(ns.hipporag_full_llm_input_topk),
        hipporag_full_llm_output_topk=int(ns.hipporag_full_llm_output_topk),
        hipporag_full_llm_model=ns.hipporag_full_llm_model,
    )


if __name__ == "__main__":
    main()
