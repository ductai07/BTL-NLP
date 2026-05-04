"""
Chuẩn bị dữ liệu Stage 2 cho GFM-RAG fine-tuning.

Pipeline bám sát logic gốc GFMRAGConstructor:
  1. NER bằng GPT-OSS-20B API → entity surface forms
  2. Entity Linking: exact → lowercase → embedding similarity → graph nodes
  3. Chunk mapping: source_pdf filter → evidence embedding similarity
  4. target_nodes.entity: lấy từ cạnh is_mentioned_in của positive chunks
  5. Output format: {id, question, start_nodes: {entity: [...]}, target_nodes: {chunk: [...], entity: [...]}}

Chạy:
  python src/graph_retriever/prepare_stage2_data.py \
      --config configs/graph_retriever/stage2_data_prep.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────
# NER
# ────────────────────────────────────────────────────────────────

def _ner_with_gpt(
    question: str,
    client: Any,
    model: str,
) -> list[str]:
    """Trích entity surface forms từ câu hỏi tài chính bằng LLM API."""
    system_prompt = (
        "You are a financial NER assistant. "
        "Extract all named entities (company names, financial metrics, products, "
        "dates, amounts, people, locations) from the given question. "
        "Return a JSON object with key 'entities' containing a list of strings. "
        "Only include entity mentions present in the question verbatim."
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0,
            max_tokens=500,
        )
        msg = response.choices[0].message
        # Reasoning models (gpt-oss-20b) đôi khi trả kết quả vào reasoning_content
        raw_content = msg.content or getattr(msg, "reasoning_content", None) or ""
        raw = raw_content.strip()
        # Parse JSON từ response
        raw = raw.strip("```json").strip("```").strip()
        parsed = json.loads(raw)
        entities = parsed.get("entities", [])
        if isinstance(entities, list):
            return [str(e).strip() for e in entities if str(e).strip()]
        return []
    except Exception as exc:
        logger.warning("NER API lỗi cho question: %s | error: %s", question[:80], exc)
        return []


def _load_ner_cache(cache_path: Path) -> dict[str, list[str]]:
    cache: dict[str, list[str]] = {}
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    cache[item["id"]] = item["ner_ents"]
                except Exception:
                    pass
    logger.info("Đọc NER cache: %d entries", len(cache))
    return cache


def _append_ner_cache(cache_path: Path, sample_id: str, ner_ents: list[str]) -> None:
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": sample_id, "ner_ents": ner_ents}, ensure_ascii=False) + "\n")


# ────────────────────────────────────────────────────────────────
# Entity Linking
# ────────────────────────────────────────────────────────────────

def _build_entity_lookup(nodes_df: pd.DataFrame) -> dict[str, str]:
    """Tạo {entity_name → entity_uid} từ nodes.csv."""
    entities = nodes_df[nodes_df["type"] == "entity"]
    return {row["name"]: row["uid"] for _, row in entities.iterrows()}


def _entity_link_exact(
    surface: str,
    name2uid: dict[str, str],
) -> str | None:
    """Exact match → lowercase match."""
    if surface in name2uid:
        return name2uid[surface]
    lower = surface.lower()
    for name, uid in name2uid.items():
        if name.lower() == lower:
            return uid
    return None


def _entity_link_embed(
    surfaces: list[str],
    entity_names: list[str],
    entity_uids: list[str],
    emb_model: Any,
    threshold: float = 0.8,
) -> dict[str, str]:
    """Embedding similarity fallback cho entity linking."""
    if not surfaces or not entity_names:
        return {}

    surf_embs = emb_model.encode(surfaces, normalize_embeddings=True, show_progress_bar=False)
    ent_embs = emb_model.encode(entity_names, normalize_embeddings=True, show_progress_bar=False)

    scores = np.dot(surf_embs, ent_embs.T)  # (n_surf, n_ent)
    result: dict[str, str] = {}
    for i, surface in enumerate(surfaces):
        best_j = int(np.argmax(scores[i]))
        if scores[i, best_j] >= threshold:
            result[surface] = entity_uids[best_j]
    return result


def link_entities(
    ner_entities: list[str],
    name2uid: dict[str, str],
    emb_model: Any,
    el_embed_threshold: float = 0.82,
) -> list[str]:
    """
    Entity linking: exact → lowercase → embedding similarity.
    Chỉ trả về uid tồn tại trong graph.
    """
    linked_uids: list[str] = []
    unmatched: list[str] = []

    for surface in ner_entities:
        uid = _entity_link_exact(surface, name2uid)
        if uid:
            linked_uids.append(uid)
        else:
            unmatched.append(surface)

    # Embedding fallback cho unmatched
    if unmatched and emb_model is not None:
        entity_names = list(name2uid.keys())
        entity_uids = list(name2uid.values())
        embed_matches = _entity_link_embed(
            unmatched, entity_names, entity_uids, emb_model, el_embed_threshold
        )
        for surface in unmatched:
            if surface in embed_matches:
                linked_uids.append(embed_matches[surface])

    return list(set(linked_uids))


# Mapping ticker → tên công ty ưu tiên tra trong graph
_TICKER_TO_NAMES: dict[str, list[str]] = {
    "AAPL": ["Apple", "Apple Inc.", "Apple Inc"],
    "HD":   ["Home Depot", "The Home Depot"],
    "INTU": ["Intuit", "Intuit Inc."],
    "MS":   ["Morgan Stanley"],
    "NVDA": ["NVIDIA", "NVIDIA Corporation"],
}


def company_fallback_entities(
    source_pdf: str,
    name2uid: dict[str, str],
) -> list[str]:
    """
    Khi start_nodes rỗng, trích ticker từ source_pdf (vd: AAPL_10-Q_...)
    rồi link tên công ty vào graph.
    """
    basename = source_pdf.replace("\\", "/").split("/")[-1]
    ticker = basename.split("_")[0].upper()
    names = _TICKER_TO_NAMES.get(ticker, [])
    uids: list[str] = []
    for name in names:
        uid = _entity_link_exact(name, name2uid)
        if uid:
            uids.append(uid)
    return uids


# ────────────────────────────────────────────────────────────────
# Chunk Mapping
# ────────────────────────────────────────────────────────────────

def _build_chunk_lookup(nodes_df: pd.DataFrame) -> dict[str, list[dict]]:
    """
    Tạo {source_pdf → list[{uid, text_preview}]} từ nodes.csv.
    """
    chunks = nodes_df[nodes_df["type"] == "chunk"]
    lookup: dict[str, list[dict]] = defaultdict(list)
    for _, row in chunks.iterrows():
        try:
            attr = json.loads(row["attributes"])
            pdf = attr.get("source_pdf", "")
            if pdf:
                lookup[pdf].append({
                    "uid": row["uid"],
                    "text_preview": attr.get("text_preview", ""),
                })
        except Exception:
            pass
    return dict(lookup)


def map_qa_to_chunks(
    source_pdf: str,
    evidence: str,
    chunk_lookup: dict[str, list[dict]],
    emb_model: Any,
) -> list[str]:
    """
    Ánh xạ một QA sample sang danh sách positive chunk UIDs.
    - Lọc theo source_pdf
    - Embedding similarity evidence vs text_preview
    """
    candidates = chunk_lookup.get(source_pdf, [])
    if not candidates:
        return []
    if len(candidates) == 1:
        return [candidates[0]["uid"]]

    previews = [c["text_preview"] for c in candidates]

    if emb_model is not None and evidence.strip():
        # Dùng embedding similarity nếu có model
        ev_emb = emb_model.encode([evidence], normalize_embeddings=True, show_progress_bar=False)
        prev_embs = emb_model.encode(previews, normalize_embeddings=True, show_progress_bar=False)
        scores = np.dot(ev_emb, prev_embs.T)[0]
    else:
        # Fallback: word overlap (Jaccard) — không cần RAM/GPU
        ev_tokens = set(evidence.lower().split())
        scores_list = []
        for prev in previews:
            p_tokens = set(prev.lower().split())
            if not ev_tokens and not p_tokens:
                scores_list.append(0.0)
            else:
                inter = len(ev_tokens & p_tokens)
                union = len(ev_tokens | p_tokens)
                scores_list.append(inter / union if union > 0 else 0.0)
        scores = np.array(scores_list)

    # Lấy top-k theo chunk_top_k trong cfg (mặc định 1)
    top_k = 1
    if len(scores) <= top_k:
        return [c["uid"] for c in candidates]
    top_idxs = np.argsort(scores)[::-1][:top_k]
    return [candidates[i]["uid"] for i in top_idxs]


# ────────────────────────────────────────────────────────────────
# Target Entity Extraction từ is_mentioned_in edges
# ────────────────────────────────────────────────────────────────

def _build_chunk_to_entities(
    graph_pt_path: Path,
    id2node: dict[str, str],
    relation2id: dict[str, int],
    mention_key: str = "is_mentioned_in",
) -> dict[str, list[str]]:
    """
    Từ graph.pt, lấy mapping chunk_uid → [entity_uid] qua cạnh is_mentioned_in (entity→chunk).
    Giải phóng graph khỏi memory ngay sau khi trích xong edges.
    """
    import gc

    data = torch.load(graph_pt_path, map_location="cpu", weights_only=False)

    if mention_key not in relation2id:
        logger.warning("Không tìm thấy relation '%s' trong rel2id", mention_key)
        del data
        gc.collect()
        return {}

    rid = int(relation2id[mention_key])
    ei = data.target_edge_index.cpu().clone()
    rt = data.target_edge_type.cpu().clone()
    # Giải phóng graph ngay sau khi lấy xong edges
    del data
    gc.collect()

    mask = rt == rid
    edges = ei[:, mask]

    chunk_to_ents: dict[str, list[str]] = defaultdict(list)
    for i in range(edges.size(1)):
        h_idx, t_idx = int(edges[0, i]), int(edges[1, i])
        entity_uid = id2node.get(str(h_idx), f"unk_{h_idx}")
        chunk_uid = id2node.get(str(t_idx), f"unk_{t_idx}")
        chunk_to_ents[chunk_uid].append(entity_uid)

    return {c: sorted(set(ents)) for c, ents in chunk_to_ents.items()}


# ────────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────────

def prepare_stage2_data(cfg: dict) -> Path:
    """Chạy toàn bộ pipeline data prep, trả về đường dẫn output."""
    # Paths
    qa_path = Path(cfg["qa_path"])
    nodes_csv_path = Path(cfg["nodes_csv"])
    tensor_dir = Path(cfg["tensor_dir"])
    output_path = Path(cfg["output_path"])
    cache_dir = Path(cfg.get("cache_dir", "tmp/stage2_data_prep"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ner_cache_path = cache_dir / "ner_cache.jsonl"
    mention_key = cfg.get("mention_relation_key", "is_mentioned_in")
    el_threshold = float(cfg.get("el_embed_threshold", 0.82))
    skip_no_chunk = bool(cfg.get("skip_no_chunk", True))
    skip_no_start = bool(cfg.get("skip_no_start", False))

    # Load mappings
    id2node: dict[str, str] = json.loads((tensor_dir / "id2node.json").read_text(encoding="utf-8"))
    relation2id: dict[str, int] = {
        k: int(v) for k, v in json.loads((tensor_dir / "rel2id.json").read_text(encoding="utf-8")).items()
    }

    # Embedding model chỉ load khi use_embedding=true trong config (mặc định tắt để tiết kiệm RAM)
    use_embedding = bool(cfg.get("use_embedding", False))
    emb_model = None
    if use_embedding:
        emb_model_name = cfg.get("embedding_model", "BAAI/bge-base-en-v1.5")
        emb_device = cfg.get("embedding_device", None)
        logger.info("Load embedding model: %s", emb_model_name)
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer(emb_model_name, device=emb_device)
        logger.info("Embedding model loaded OK")
    else:
        logger.info("use_embedding=false: dùng exact/fuzzy match, không load BGE (tiết kiệm ~440MB RAM)")

    # Load nodes.csv
    logger.info("Đọc nodes.csv từ %s", nodes_csv_path)
    nodes_df = pd.read_csv(nodes_csv_path, keep_default_na=False)
    name2uid = _build_entity_lookup(nodes_df)
    chunk_lookup = _build_chunk_lookup(nodes_df)
    logger.info("Đã build entity lookup: %d entities | chunk_lookup: %d PDFs", len(name2uid), len(chunk_lookup))

    # Build chunk→entities mapping từ graph edges (load rồi giải phóng graph.pt)
    logger.info("Build chunk→entities từ %s", mention_key)
    chunk_to_entities = _build_chunk_to_entities(
        tensor_dir / "graph.pt", id2node, relation2id, mention_key
    )
    logger.info("chunk_to_entities: %d chunks có entity", len(chunk_to_entities))

    # Khởi tạo GPT NER client
    ner_api_cfg = cfg.get("ner_api", {})
    ner_client = None
    if ner_api_cfg.get("base_url") or ner_api_cfg.get("api_key"):
        try:
            import openai
            ner_client = openai.OpenAI(
                base_url=ner_api_cfg.get("base_url"),
                api_key=ner_api_cfg.get("api_key", "not-needed"),
            )
            logger.info("GPT NER client: base_url=%s model=%s", ner_api_cfg.get("base_url"), ner_api_cfg.get("model"))
        except Exception as e:
            logger.warning("Không thể init NER client: %s", e)

    # Load QA data
    qa_samples: list[dict] = []
    with open(qa_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_samples.append(json.loads(line))
    logger.info("Đọc %d QA samples từ %s", len(qa_samples), qa_path)

    # Load NER cache
    ner_cache = _load_ner_cache(ner_cache_path)

    # ────── Xử lý từng sample ──────
    stats = {
        "total": len(qa_samples),
        "mapped_chunk": 0,
        "mapped_start_entity": 0,
        "mapped_target_entity": 0,
        "skipped_no_chunk": 0,
        "skipped_no_start": 0,
    }
    output_samples: list[dict] = []
    sample_logs: list[str] = []

    for idx, sample in enumerate(qa_samples):
        sample_id = str(sample["id"])
        question = sample["question"]
        evidence = sample.get("evidence", "")
        source_pdf = sample.get("source_pdf", "")

        # ── Bước 1: NER ──
        if sample_id in ner_cache:
            ner_ents = ner_cache[sample_id]
        elif ner_client is not None:
            ner_ents = _ner_with_gpt(question, ner_client, ner_api_cfg.get("model", "gpt-4o-mini"))
            _append_ner_cache(ner_cache_path, sample_id, ner_ents)
            ner_cache[sample_id] = ner_ents
        else:
            # Fallback: dùng question words làm surface forms (không có API)
            logger.debug("Sample %s: không có NER client, bỏ qua NER", sample_id)
            ner_ents = []

        # ── Bước 2: Entity Linking ──
        linked_entities = link_entities(ner_ents, name2uid, emb_model, el_threshold)

        # Fallback: nếu không link được entity nào, dùng company entity từ source_pdf ticker
        if not linked_entities and source_pdf:
            linked_entities = company_fallback_entities(source_pdf, name2uid)
            if linked_entities:
                logger.debug("Sample %s: dùng company fallback → %s", sample_id, linked_entities)

        # ── Bước 3: Chunk Mapping ──
        positive_chunks = map_qa_to_chunks(source_pdf, evidence, chunk_lookup, emb_model)

        if not positive_chunks:
            stats["skipped_no_chunk"] += 1
            if skip_no_chunk:
                continue

        if not linked_entities and skip_no_start:
            stats["skipped_no_start"] += 1
            continue

        # ── Bước 4: Target entities từ is_mentioned_in ──
        target_entities: list[str] = []
        for chunk_uid in positive_chunks:
            target_entities.extend(chunk_to_entities.get(chunk_uid, []))
        target_entities = list(set(target_entities))

        # Cập nhật stats
        if positive_chunks:
            stats["mapped_chunk"] += 1
        if linked_entities:
            stats["mapped_start_entity"] += 1
        if target_entities:
            stats["mapped_target_entity"] += 1

        out_sample = {
            "id": sample_id,
            "question": question,
            "answer": sample.get("answer", ""),
            "start_nodes": {"entity": linked_entities},
            "target_nodes": {
                "chunk": positive_chunks,
                "entity": target_entities,
            },
        }
        output_samples.append(out_sample)

        # Log decode cho 5 sample đầu
        if len(sample_logs) < 5:
            sample_logs.append(
                f"  [{sample_id}] Q: {question[:80]}\n"
                f"    NER: {ner_ents}\n"
                f"    start_nodes.entity: {linked_entities}\n"
                f"    target_nodes.chunk: {positive_chunks}\n"
                f"    target_nodes.entity: {target_entities[:3]}...\n"
            )

    # ────── Lưu output ──────
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_samples, f, ensure_ascii=False, indent=2)

    # ────── Log thống kê ──────
    logger.info("=" * 60)
    logger.info("STAGE 2 DATA PREP STATS")
    logger.info("  Total QA samples: %d", stats["total"])
    logger.info("  Output samples:   %d", len(output_samples))
    logger.info("  Mapped positive chunk: %d (%.1f%%)", stats["mapped_chunk"], 100 * stats["mapped_chunk"] / max(1, stats["total"]))
    logger.info("  Has start_nodes.entity: %d (%.1f%%)", stats["mapped_start_entity"], 100 * stats["mapped_start_entity"] / max(1, len(output_samples)))
    logger.info("  Has target_nodes.entity: %d (%.1f%%)", stats["mapped_target_entity"], 100 * stats["mapped_target_entity"] / max(1, len(output_samples)))
    logger.info("  Skipped (no chunk): %d", stats["skipped_no_chunk"])
    logger.info("  Skipped (no start): %d", stats["skipped_no_start"])
    logger.info("=" * 60)
    logger.info("5 sample decode:")
    for log in sample_logs:
        logger.info(log)

    logger.info("Đã lưu Stage 2 data: %s (%d samples)", output_path, len(output_samples))
    return output_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chuẩn bị dữ liệu Stage 2 GFM-RAG.")
    p.add_argument("--config", type=Path, default=Path("configs/graph_retriever/stage2_data_prep.yaml"))
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ns = parse_args(argv)

    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(OmegaConf.load(ns.config), resolve=True)
    assert isinstance(cfg, dict)

    out = prepare_stage2_data(cfg)
    logger.info("Hoàn thành: %s", out)


if __name__ == "__main__":
    main()
