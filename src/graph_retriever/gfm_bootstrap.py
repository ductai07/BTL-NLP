from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def bootstrap_gfmrag(gfmrag_path: str | None) -> None:
    """
    Nạp repo tác giả gfm-rag như một package `gfmrag` mà không cần pip install.

    gfmrag_path: đường dẫn tới thư mục clone, ví dụ `d:/Project/gfm-rag`.
    """
    candidates: list[Path] = []
    if gfmrag_path:
        candidates.append(Path(gfmrag_path).resolve())
    env_path = os.environ.get("GFM_RAG_PATH")
    if env_path:
        candidates.append(Path(env_path).resolve())

    # fallback cho trường hợp đã copy `gfmrag/` vào project hiện tại
    repo_root = Path(__file__).resolve().parents[2]
    candidates.extend([repo_root, Path.cwd()])

    resolved: Path | None = None
    package_dir: Path | None = None
    for cand in candidates:
        cand_pkg = cand if cand.name == "gfmrag" else cand / "gfmrag"
        if cand_pkg.is_dir():
            resolved = cand_pkg.parent
            package_dir = cand_pkg
            break

    if resolved is None or package_dir is None:
        raise FileNotFoundError(
            "Không tìm thấy thư mục `gfmrag/`. Hãy copy `gfmrag` vào project hoặc truyền --gfmrag-path."
        )

    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))

    if "gfmrag" not in sys.modules:
        package = types.ModuleType("gfmrag")
        package.__path__ = [str(package_dir)]
        sys.modules["gfmrag"] = package

    # Tạo stub `gfmrag.text_emb_models` để tránh kéo các model phụ thuộc nặng (einops, ...)
    # KGC pretraining chỉ cần core model/trainer/tasks; không cần NVEmbed.
    if "gfmrag.text_emb_models" not in sys.modules:
        module = types.ModuleType("gfmrag.text_emb_models")

        import torch

        class BaseTextEmbModel:  # type: ignore
            def __init__(
                self,
                *,
                model_name: str = "BAAI/bge-base-en-v1.5",
                batch_size: int = 32,
                device: str | None = None,
                normalize_embeddings: bool = True,
                **_: object,
            ) -> None:
                self.model_name = model_name
                self.batch_size = batch_size
                self.device = device
                self.normalize = normalize_embeddings
                self._model = None

            def _load(self):
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(self.model_name, device=self.device)
                return self._model

            def encode(self, texts, *, is_query: bool):  # noqa:ARG002
                model = self._load()
                emb = model.encode(
                    list(texts),
                    batch_size=self.batch_size,
                    normalize_embeddings=self.normalize,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                )
                if isinstance(emb, torch.Tensor):
                    return emb.float().cpu()
                return torch.tensor(emb, dtype=torch.float32)

        module.BaseTextEmbModel = BaseTextEmbModel
        module.__all__ = ["BaseTextEmbModel"]
        sys.modules["gfmrag.text_emb_models"] = module

    # Smoke import để fail sớm nếu thiếu dependency
    import gfmrag  # noqa:F401


def disable_custom_rspmm() -> None:
    """
    Trên Windows/CPU thường không build được extension rspmm.
    Patch Ultra layer để luôn dùng propagate mặc định của PyG.
    """
    from torch_geometric.nn.conv import MessagePassing

    from gfmrag.models.ultra.layers import GeneralizedRelationalConv

    def pyg_propagate(self, edge_index, size=None, **kwargs):
        return MessagePassing.propagate(self, edge_index, size=size, **kwargs)

    GeneralizedRelationalConv.propagate = pyg_propagate  # type: ignore[method-assign]

