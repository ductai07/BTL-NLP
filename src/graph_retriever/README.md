# Graph Retriever - GFM-RAG KGC Finetune

Module này dùng **đúng luồng tác giả** cho KG Completion:
- Model: `QueryGNN + QueryNBFNet` (từ thư mục `gfmrag/` đã copy vào project)
- Trainer: `KGCTrainer`
- Negative sampling/ranking: `gfmrag.models.ultra.tasks`

Không rewrite pipeline RAG hiện có, chỉ thêm module riêng cho pretrain/finetune graph retriever.

Điều kiện quan trọng:
- Trong repo hiện tại đã có thư mục `gfmrag/` (copy từ repo tác giả).
- Vì vậy trên cloud/Kaggle có thể chạy trực tiếp, không cần đường dẫn Windows cũ.

## 1) Chuẩn bị dữ liệu graph

Yêu cầu tối thiểu trong `data/graph_tensor`:
- `graph.pt`
- `node2id.json`
- `id2node.json`
- `rel2id.json`

Nếu chưa có:

```bash
python src/graph_extraction/build_graph_tensor.py --graph-dir data/graph --output-dir data/graph_tensor
```

Lưu ý:
- `graph.pt` phải có `edge_index`, `edge_type`, `target_edge_index`, `target_edge_type`.
- Inverse edges đã cần sẵn để `strict_negative_mask` hoạt động đúng.
- `rel_attr` sẽ được module tự tạo từ `rel2id.json` bằng BGE nếu graph chưa có.

## 2) File config chính

File: `configs/graph_retriever/kgc_gfm_training.yaml`

Nhóm tham số quan trọng:
- `gfmrag_path`: đường dẫn repo tác giả (`d:/Project/gfm-rag`)
- `disable_custom_rspmm`: nên để `true` trên Windows/CPU
- `graph.tensor_dir`: thư mục graph tensor local
- `graph.relation_embedding_model`: model sinh `rel_attr` (mặc định BGE)
- `model.entity_model`: cấu hình `QueryNBFNet`
- `training.pretrained_model_path`: checkpoint pretrained để finetune (`model/model.pth`)
- `training.epochs`, `train_batch_size`, `num_negative`, `strict_negative`, `fast_test`

## 3) Finetune từ pretrained của tác giả

```bash
python -m src.graph_retriever.train_kgc \
  --config configs/graph_retriever/kgc_gfm_training.yaml \
  --disable-custom-rspmm
```

Điểm bắt buộc khi dùng pretrained:
- Kiến trúc trong `model.entity_model` phải **khớp tuyệt đối** với checkpoint (`input_dim`, `hidden_dims`, ...).
- Nếu mismatch shape khi load weights, hãy chỉnh lại YAML cho trùng pretrained.

Output:
- Best checkpoint: `outputs/graph_retriever/kgc_gfm_pretrained/model_best.pth`

## 4) Inference (tail prediction cho link prediction)

```bash
python -m src.graph_retriever.inference \
  --tensor-dir data/graph_tensor \
  --checkpoint outputs/graph_retriever/kgc_gfm_pretrained/model_best.pth \
  --model-config configs/graph_retriever/kgc_gfm_training.yaml \
  --gfmrag-path d:/Project/gfm-rag \
  --relation is_mentioned_in \
  --head entity:YOUR_ENTITY_HASH \
  --direction tail \
  --top-k 10 \
  --candidates entity_only
```

Output JSON:
- `top_entities`: top-k node dự đoán
- `top_documents`: map entity -> chunk/document qua cạnh `is_mentioned_in` (nếu có)

## 5) Gợi ý tune khi dữ liệu ít

- Luôn bắt đầu từ `training.pretrained_model_path`.
- Tăng `epochs` (10-20), giảm `lr` nếu dao động.
- Với CPU:
  - giữ kiến trúc khớp pretrained
  - giảm `train_batch_size`, `num_negative`
  - giữ `disable_custom_rspmm: true`
