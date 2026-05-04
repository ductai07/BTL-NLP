#!/usr/bin/env python3
"""
Tiện ích chuyển đổi embedding model cho Stage 2 training.

Chuyển từ bge-base → bge-m3 / bge-small / bge-large.
Tự động:
  1. Cập nhật config YAML
  2. Xóa cache embedding
  3. Rebuild graph tensor nếu cần

Chạy:
  python src/graph_retriever/switch_embedding_model.py --model BAAI/bge-m3 --auto-rebuild
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


EMBEDDING_MODELS = {
    "bge-base": {
        "model_id": "BAAI/bge-base-en-v1.5",
        "dim": 768,
        "params": "109M",
        "memory_mb": 1100,
        "speed_t4": "baseline",
        "quality": "baseline",
    },
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "dim": 384,
        "params": "84M",
        "memory_mb": 400,
        "speed_t4": "+40%",
        "quality": "+2%",
    },
    "bge-small": {
        "model_id": "BAAI/bge-small-en-v1.5",
        "dim": 384,
        "params": "33M",
        "memory_mb": 150,
        "speed_t4": "+60%",
        "quality": "-10%",
    },
    "bge-large": {
        "model_id": "BAAI/bge-large-en-v1.5",
        "dim": 1024,
        "params": "335M",
        "memory_mb": 2500,
        "speed_t4": "-30%",
        "quality": "+5%",
    },
}


def update_yaml_config(
    config_path: Path,
    new_model: str,
    dry_run: bool = False,
) -> bool:
    """Cập nhật embedding model trong YAML config."""
    logger.info(f"Cập nhật config: {config_path}")

    if not config_path.exists():
        logger.error(f"Config không tồn tại: {config_path}")
        return False

    content = config_path.read_text(encoding="utf-8")

    # Thay thế relation_embedding_model
    old_rel = None
    for key in EMBEDDING_MODELS:
        model_id = EMBEDDING_MODELS[key]["model_id"]
        if model_id in content:
            old_rel = model_id
            break

    if old_rel:
        logger.info(f"  Tìm thấy relation_embedding_model: {old_rel}")
        content = content.replace(
            f"relation_embedding_model: {old_rel}",
            f"relation_embedding_model: {new_model}",
        )
    else:
        logger.warning("  Không tìm thấy relation_embedding_model trong config")

    # Thay thế text_emb_model
    old_text = None
    for key in EMBEDDING_MODELS:
        model_id = EMBEDDING_MODELS[key]["model_id"]
        if f"text_emb_model: {model_id}" in content:
            old_text = model_id
            break

    if old_text:
        logger.info(f"  Tìm thấy text_emb_model: {old_text}")
        content = content.replace(
            f"text_emb_model: {old_text}",
            f"text_emb_model: {new_model}",
        )
    else:
        logger.warning("  Không tìm thấy text_emb_model trong config")

    if dry_run:
        logger.info("  [DRY RUN] Không ghi file")
        return True

    config_path.write_text(content, encoding="utf-8")
    logger.info(f"✅ Config updated: {config_path}")
    return True


def clear_embedding_cache(cache_dir: Path, dry_run: bool = False) -> bool:
    """Xóa cache embedding cũ."""
    if not cache_dir.exists():
        logger.info(f"  Cache không tồn tại: {cache_dir}")
        return True

    cache_file = cache_dir / "test_question_embeddings.pt"
    if cache_file.exists():
        logger.info(f"Xóa cache: {cache_file}")
        if not dry_run:
            cache_file.unlink()
            logger.info(f"✅ Cache cleared: {cache_file}")
        else:
            logger.info("  [DRY RUN] Không xóa file")
    else:
        logger.info(f"  Cache không tồn tại: {cache_file}")

    return True


def clear_graph_tensor(tensor_dir: Path, dry_run: bool = False) -> bool:
    """Xóa graph tensor để rebuild (nếu chiều thay đổi)."""
    if not tensor_dir.exists():
        logger.info(f"Graph tensor không tồn tại: {tensor_dir}")
        return True

    # Giữ lại id2node, node2id, rel2id (không phụ thuộc embedding)
    rel_attr_file = tensor_dir / "rel_attr.pt"
    if rel_attr_file.exists():
        logger.info(f"Xóa relation attributes cũ: {rel_attr_file}")
        if not dry_run:
            rel_attr_file.unlink()
            logger.info(f"✅ Relation attributes cleared: {rel_attr_file}")
        else:
            logger.info("  [DRY RUN] Không xóa file")

    return True


def rebuild_graph_tensor(
    config_path: Path,
    model_name: str,
    dry_run: bool = False,
) -> bool:
    """Chạy build_graph_tensor.py để rebuild với model mới."""
    logger.info(f"Rebuild graph tensor với model: {model_name}")

    cmd = [
        sys.executable,
        "src/graph_extraction/build_graph_tensor.py",
        "--graph-dir", "data/graph",
        "--output-dir", "data/graph_tensor",
        "--embed-features",
        "--embedding-model", model_name,
    ]

    if dry_run:
        logger.info(f"  [DRY RUN] Lệnh sẽ chạy: {' '.join(cmd)}")
        return True

    try:
        logger.info(f"  Chạy: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ Graph tensor rebuilt: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Lỗi rebuild: {e.stderr}")
        return False


def show_model_info(model_key: str | None = None) -> None:
    """Hiển thị thông tin các model."""
    logger.info("\n📊 Available Embedding Models:")
    logger.info("-" * 100)

    for key in EMBEDDING_MODELS:
        info = EMBEDDING_MODELS[key]
        marker = " <-- hiện tại" if key == "bge-base" else ""
        logger.info(
            f"  {key:15} | {info['model_id']:35} | "
            f"Dim: {info['dim']:4} | Params: {info['params']:6} | "
            f"Memory: {info['memory_mb']:5}MB | T4: {info['speed_t4']:6} | Quality: {info['quality']:5}{marker}"
        )

    logger.info("-" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Chuyển đổi embedding model cho Stage 2 training.",
        epilog="""
Ví dụ:
  python src/graph_retriever/switch_embedding_model.py --model bge-m3 --auto-rebuild
  python src/graph_retriever/switch_embedding_model.py --model BAAI/bge-large-en-v1.5
  python src/graph_retriever/switch_embedding_model.py --list-models
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model mới (vd: bge-m3, BAAI/bge-m3, bge-small, bge-large)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/graph_retriever/stage2_sft.yaml"),
        help="Config YAML path",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/qa"),
        help="Data directory path",
    )
    parser.add_argument(
        "--tensor-dir",
        type=Path,
        default=Path("data/graph_tensor"),
        help="Graph tensor directory path",
    )
    parser.add_argument(
        "--auto-rebuild",
        action="store_true",
        help="Tự động rebuild graph tensor (chậm, ~5 phút)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ hiển thị, không thực thi",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Liệt kê các model khả dụng",
    )

    args = parser.parse_args()

    if args.list_models:
        show_model_info()
        return

    if not args.model:
        logger.error("❌ Phải chỉ định --model hoặc --list-models")
        parser.print_help()
        sys.exit(1)

    # Resolve model name
    model_id = None
    for key, info in EMBEDDING_MODELS.items():
        if args.model == key or args.model == info["model_id"]:
            model_id = info["model_id"]
            break

    if not model_id:
        logger.error(f"❌ Model không tìm thấy: {args.model}")
        show_model_info()
        sys.exit(1)

    logger.info(f"🔄 Chuyển đổi embedding model sang: {model_id}")
    logger.info(f"   Config: {args.config}")
    if args.dry_run:
        logger.info("   [DRY RUN MODE]")

    # Bước 1: Cập nhật config
    if not update_yaml_config(args.config, model_id, dry_run=args.dry_run):
        sys.exit(1)

    # Bước 2: Xóa cache embedding
    if not clear_embedding_cache(args.data_dir, dry_run=args.dry_run):
        sys.exit(1)

    # Bước 3: Rebuild graph tensor nếu được yêu cầu
    if args.auto_rebuild:
        if not clear_graph_tensor(args.tensor_dir, dry_run=args.dry_run):
            sys.exit(1)

        if not rebuild_graph_tensor(args.config, model_id, dry_run=args.dry_run):
            logger.warning("⚠️  Rebuild graph tensor thất bại")
            logger.info("   Chạy thủ công:")
            logger.info(
                f"   python src/graph_extraction/build_graph_tensor.py "
                f"--graph-dir data/graph --output-dir data/graph_tensor "
                f"--embed-features --embedding-model {model_id}"
            )
            sys.exit(1)
    else:
        logger.info("\n📝 Bước tiếp theo (manual):")
        logger.info(f"   python src/graph_extraction/build_graph_tensor.py \\")
        logger.info(f"     --graph-dir data/graph \\")
        logger.info(f"     --output-dir data/graph_tensor \\")
        logger.info(f"     --embed-features \\")
        logger.info(f"     --embedding-model {model_id}")

    logger.info("\n✅ Hoàn tất! Bây giờ chạy training:")
    logger.info("   python src/graph_retriever/train_stage2.py \\")
    logger.info("     --config configs/graph_retriever/stage2_sft.yaml")


if __name__ == "__main__":
    main()
