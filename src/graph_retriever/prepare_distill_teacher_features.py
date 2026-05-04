"""
Prepare teacher embeddings for distillation without modifying main graph.pt.

Usage:
  python src/graph_retriever/prepare_distill_teacher_features.py \
    --config configs/graph_retriever/stage2_sft.yaml \
    --teacher-model BAAI/bge-m3 \
    --output-dir data/distill_features/bge-m3 \
    --force

Output:
  data/distill_features/<teacher_name>/
    ├── node_x.pt          (node embeddings, shape: [num_nodes, dim])
    ├── question_embeddings.pt (question embeddings for stage2 samples)
    ├── meta.json          (metadata: model, dim, timestamps)
    └── sample_q2idx.json  (mapping sample_id -> embedding index)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def load_graph_nodes_csv(nodes_csv_path: Path):
    """Load and parse nodes.csv from graph."""
    import pandas as pd
    nodes_df = pd.read_csv(nodes_csv_path, keep_default_na=False)
    return nodes_df


def load_embedding_model(model_name: str, device: str = "cpu"):
    """Load embedding model (BGE or similar)."""
    logger.info(f"Loading embedding model: {model_name} on device: {device}")
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        return model
    except ImportError:
        logger.error("transformers not installed. Install: pip install transformers")
        raise


def encode_texts(
    texts: list[str],
    model,
    batch_size: int = 32,
    device: str = "cpu",
    normalize: bool = True,
) -> np.ndarray:
    """Encode list of texts to embeddings."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers not installed")
        raise

    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts", leave=False):
        batch = texts[i : i + batch_size]
        
        # Remove empty texts
        batch = [t if t and isinstance(t, str) else "[UNKNOWN]" for t in batch]

        with torch.no_grad():
            # Use modern tokenizer API (works with all transformers versions)
            encoded = tokenizer(
                batch,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(**encoded)
            batch_emb = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            embeddings.append(batch_emb.detach().cpu().numpy())

    embeddings = np.vstack(embeddings)

    # Normalize
    if normalize:
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)

    return embeddings


def prepare_node_embeddings(
    nodes_csv_path: Path,
    node2id_path: Path,
    model,
    device: str = "cpu",
    batch_size: int = 32,
) -> tuple[np.ndarray, dict]:
    """
    Prepare node embeddings (entities + chunks) in node2id order.
    
    Returns:
        (embeddings, metadata)
    """
    logger.info(f"Loading nodes from: {nodes_csv_path}")
    nodes_df = load_graph_nodes_csv(nodes_csv_path)
    logger.info(f"Loaded {len(nodes_df)} nodes")

    logger.info(f"Loading node2id from: {node2id_path}")
    node2id = json.loads(node2id_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(node2id)} node2id mappings")

    # Build ordered list by node2id
    id2node = {v: k for k, v in node2id.items()}
    num_nodes = len(node2id)
    
    node_texts = []
    node_types = []
    
    for node_idx in range(num_nodes):
        node_id = id2node[node_idx]
        
        # Find node in dataframe
        row = nodes_df[nodes_df["uid"] == node_id]
        if row.empty:
            logger.warning(f"Node {node_id} not found in nodes.csv, using [UNKNOWN]")
            text = "[UNKNOWN]"
            node_type = "unknown"
        else:
            node_type = row["type"].iloc[0]
            node_name = row["name"].iloc[0]
            
            # Parse attributes JSON to extract text
            try:
                attrs_str = row["attributes"].iloc[0]
                if isinstance(attrs_str, str):
                    attrs = json.loads(attrs_str)
                    # Try to extract text_preview or chunk_id or any text field
                    text = (
                        attrs.get("text_preview", "")
                        or attrs.get("chunk_id", "")
                        or attrs.get("summary", "")
                        or node_name
                        or "[UNKNOWN]"
                    )
                else:
                    text = node_name or "[UNKNOWN]"
            except Exception as e:
                logger.debug(f"Failed to parse attributes for {node_id}: {e}")
                text = node_name or "[UNKNOWN]"
            
            # Ensure string
            if not isinstance(text, str) or not text.strip():
                text = f"[{node_type}:{node_id}]"
        
        node_texts.append(text[:512] if len(text) > 512 else text)  # Truncate
        node_types.append(node_type)

    logger.info(f"Encoding {num_nodes} nodes...")
    embeddings = encode_texts(
        node_texts,
        model,
        batch_size=batch_size,
        device=device,
        normalize=True,
    )

    metadata = {
        "num_nodes": num_nodes,
        "embedding_dim": int(embeddings.shape[1]),
        "node_type_counts": {
            "chunk": sum(1 for t in node_types if t == "chunk"),
            "entity": sum(1 for t in node_types if t == "entity"),
            "unknown": sum(1 for t in node_types if t == "unknown"),
        },
        "source_nodes_csv": str(nodes_csv_path),
        "source_node2id": str(node2id_path),
    }

    return embeddings, metadata


def prepare_question_embeddings(
    stage2_json_path: Path,
    model,
    device: str = "cpu",
    batch_size: int = 32,
) -> tuple[np.ndarray, dict]:
    """
    Prepare question embeddings from stage2 JSON.
    
    Returns:
        (embeddings, sample_q2idx mapping)
    """
    logger.info(f"Loading stage2 data from: {stage2_json_path}")
    stage2_data = json.loads(stage2_json_path.read_text(encoding="utf-8"))

    if not isinstance(stage2_data, list):
        logger.warning("stage2_json is dict, extracting as list of samples")
        stage2_data = [stage2_data]

    logger.info(f"Loaded {len(stage2_data)} samples")

    questions = []
    sample_ids = []

    for idx, sample in enumerate(stage2_data):
        q = sample.get("question", "[UNKNOWN]")
        if not q or not isinstance(q, str):
            q = "[UNKNOWN]"
        questions.append(q)
        sample_ids.append(idx)

    logger.info(f"Encoding {len(questions)} questions...")
    embeddings = encode_texts(
        questions,
        model,
        batch_size=batch_size,
        device=device,
        normalize=True,
    )

    sample_q2idx = {str(sid): int(idx) for idx, sid in enumerate(sample_ids)}

    return embeddings, sample_q2idx


def main():
    parser = argparse.ArgumentParser(
        description="Prepare teacher embeddings for distillation.",
        epilog="""
Examples:
  python src/graph_retriever/prepare_distill_teacher_features.py \
    --config configs/graph_retriever/stage2_sft.yaml \
    --teacher-model BAAI/bge-base-en-v1.5 \
    --output-dir data/distill_features/bge-base-en-v1.5 \
    --force

  python src/graph_retriever/prepare_distill_teacher_features.py \
    --config configs/graph_retriever/stage2_sft.yaml \
    --teacher-model BAAI/bge-m3 \
    --output-dir data/distill_features/bge-m3 \
    --force
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Stage 2 config YAML path",
    )
    parser.add_argument(
        "--teacher-model",
        type=str,
        required=True,
        help="Teacher embedding model (e.g., BAAI/bge-m3)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for teacher features",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing features",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device (cpu or cuda:0)",
    )

    args = parser.parse_args()

    # Validate paths
    if not args.config.exists():
        logger.error(f"Config not found: {args.config}")
        sys.exit(1)

    # Load config
    import yaml
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    data_cfg = cfg.get("data", {})
    graph_cfg = cfg.get("graph", {})

    # Resolve paths
    stage2_json = Path(str(data_cfg.get("stage2_json", "data/qa/test_qa_stage2.json")))
    tensor_dir = Path(str(graph_cfg.get("tensor_dir", "data/graph_tensor")))
    graph_dir = Path(str(graph_cfg.get("graph_dir", "data/graph")))

    nodes_csv = graph_dir / "nodes.csv"
    node2id_json = tensor_dir / "node2id.json"

    if not nodes_csv.exists():
        logger.error(f"nodes.csv not found: {nodes_csv}")
        sys.exit(1)

    if not node2id_json.exists():
        logger.error(f"node2id.json not found: {node2id_json}")
        sys.exit(1)

    if not stage2_json.exists():
        logger.error(f"stage2_json not found: {stage2_json}")
        sys.exit(1)

    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already exists
    existing_node_x = args.output_dir / "node_x.pt"
    if existing_node_x.exists() and not args.force:
        logger.error(f"Features already exist at {args.output_dir}. Use --force to overwrite.")
        sys.exit(1)

    logger.info(f"Project root: {_REPO_ROOT}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Teacher model: {args.teacher_model}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")

    # Load model
    model = load_embedding_model(args.teacher_model, device=args.device)
    logger.info(f"Model loaded, embedding dim: {model.config.hidden_size}")

    # Prepare node embeddings
    logger.info("\n=== Preparing Node Embeddings ===")
    node_emb, node_meta = prepare_node_embeddings(
        nodes_csv,
        node2id_json,
        model,
        device=args.device,
        batch_size=args.batch_size,
    )
    logger.info(f"Node embeddings shape: {node_emb.shape}")
    logger.info(f"Node metadata: {json.dumps(node_meta, indent=2)}")

    # Prepare question embeddings
    logger.info("\n=== Preparing Question Embeddings ===")
    q_emb, q2idx = prepare_question_embeddings(
        stage2_json,
        model,
        device=args.device,
        batch_size=args.batch_size,
    )
    logger.info(f"Question embeddings shape: {q_emb.shape}")

    # Save
    logger.info("\n=== Saving Features ===")
    node_x_path = args.output_dir / "node_x.pt"
    q_emb_path = args.output_dir / "question_embeddings.pt"
    q2idx_path = args.output_dir / "sample_q2idx.json"
    meta_path = args.output_dir / "meta.json"

    torch.save(torch.from_numpy(node_emb).float(), node_x_path)
    logger.info(f"Saved node embeddings: {node_x_path}")

    torch.save(torch.from_numpy(q_emb).float(), q_emb_path)
    logger.info(f"Saved question embeddings: {q_emb_path}")

    q2idx_path.write_text(json.dumps(q2idx, indent=2), encoding="utf-8")
    logger.info(f"Saved sample q2idx mapping: {q2idx_path}")

    # Create metadata
    meta = {
        "teacher_model": args.teacher_model,
        "node_x_shape": list(node_emb.shape),
        "question_embeddings_shape": list(q_emb.shape),
        "num_samples": len(q_emb),
        "embedding_dim": int(node_emb.shape[1]),
        "nodes_metadata": node_meta,
        "source_config": str(args.config),
        "source_nodes_csv": str(nodes_csv),
        "source_node2id": str(node2id_json),
        "source_stage2_json": str(stage2_json),
        "created_at": datetime.now().isoformat(),
    }

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"Saved metadata: {meta_path}")

    logger.info("\n✅ Teacher features prepared successfully!")
    logger.info(f"   Features saved to: {args.output_dir}")
    logger.info(f"   Next: Update config to use these features and run training")


if __name__ == "__main__":
    main()
