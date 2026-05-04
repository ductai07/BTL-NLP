# 🔬 Distillation Ablation Quick Start Guide

## Tl;dr

```bash
# 1. Chuẩn bị teacher embeddings
python src/graph_retriever/prepare_distill_teacher_features.py \
  --config configs/graph_retriever/stage2_sft.yaml \
  --teacher-model BAAI/bge-m3 \
  --output-dir data/distill_features/bge-m3 \
  --force

# 2. Chạy training với BGE-M3 distillation
python src/graph_retriever/train_stage2.py \
  --config configs/graph_retriever/stage2_sft_distill_bge_m3.yaml \
  --run-sanity-first
```

---

## Workflow Distillation Lengkap

### 1️⃣ Chuẩn bị Teacher Embeddings

**Vì sao?** 
- Teacher embeddings (node_x, question_embeddings) disiapkan terlebih dahulu
- Disimpan di folder terpisah (`data/distill_features/<teacher_model>/`)
- **Tidak** mengubah graph tensor chính

**Bagaimana?**

```bash
# BGE-base teacher (default)
python src/graph_retriever/prepare_distill_teacher_features.py \
  --config configs/graph_retriever/stage2_sft.yaml \
  --teacher-model BAAI/bge-base-en-v1.5 \
  --output-dir data/distill_features/bge-base-en-v1.5 \
  --force

# BGE-M3 teacher (recommended for T4)
python src/graph_retriever/prepare_distill_teacher_features.py \
  --config configs/graph_retriever/stage2_sft.yaml \
  --teacher-model BAAI/bge-m3 \
  --output-dir data/distill_features/bge-m3 \
  --force

# BGE-small teacher (minimum resources)
python src/graph_retriever/prepare_distill_teacher_features.py \
  --config configs/graph_retriever/stage2_sft.yaml \
  --teacher-model BAAI/bge-small-en-v1.5 \
  --output-dir data/distill_features/bge-small-en-v1.5 \
  --force
```

**Output:**
```
data/distill_features/bge-m3/
  ├── node_x.pt                    # Shape: [17262, 1024] untuk BGE-M3
  ├── question_embeddings.pt       # Shape: [120, 1024]
  ├── sample_q2idx.json            # Mapping sample_id -> embedding_idx
  └── meta.json                    # Metadata (model name, dims, timestamps, etc)
```

**Flags:**
- `--force`: Overwrite existing features
- `--batch-size 32`: Adjust batch size untuk embedding (default 32)
- `--device cpu`: Load ke CPU (GPU juga bisa, but CPU lebih aman untuk embedding)

---

### 2️⃣ Pilih Config untuk Training

#### Option A: Hard Labels Saja (Baseline)

```bash
# Create config atau edit temp file
cp configs/graph_retriever/stage2_sft.yaml /tmp/stage2_hard_only.yaml
# Edit: distillation.enable = false

python src/graph_retriever/train_stage2.py \
  --config /tmp/stage2_hard_only.yaml \
  --run-sanity-first
```

**Output:** `outputs/graph_retriever/kgc_stage2_sft/`

Expected metrics:
- chunk_mrr: 0.15-0.17
- Tidak ada distillation loss

#### Option B: Hard + BGE-base Distillation (Default)

```bash
python src/graph_retriever/train_stage2.py \
  --config configs/graph_retriever/stage2_sft.yaml \
  --run-sanity-first
```

**Output:** `outputs/graph_retriever/kgc_stage2_sft/`

Expected metrics:
- chunk_mrr: 0.1828
- entity_mrr: 0.1073

#### Option C: Hard + BGE-M3 Distillation (Recommended)

Sebelumnya sudah prepare embeddings di step 1.

```bash
python src/graph_retriever/train_stage2.py \
  --config configs/graph_retriever/stage2_sft_distill_bge_m3.yaml \
  --run-sanity-first
```

**Output:** `outputs/graph_retriever/kgc_stage2_sft_distill_bge_m3/`

Latest metrics (run 2026-05-04 03:28 UTC):
- chunk_mrr: 0.1994
- chunk_hits@10: 0.5238
- chunk_hits@20: 0.5714
- entity_mrr: 0.1221
- entity_hits@10: 0.2287
- entity_recall@20: 0.2591

---

### 3️⃣ Monitoring Training

**Sanity Mode (Default, cepat):**
```bash
--run-sanity-first
# Runs:
# - 20 samples
# - 300 steps
# - 50-60 detik
# Validates distillation setup trước full train
```

**Full Training:**
```bash
# Tanpa --run-sanity-first
python src/graph_retriever/train_stage2.py \
  --config configs/graph_retriever/stage2_sft_distill_bge_m3.yaml
```

Waktu: ~2-3 jam untuk 35 epochs pada T4

---

## Architecture: Mengapa Tidak Modify graph.pt?

### ❌ Approach Lama (Dangerous)

```
graph.pt (768 dim)
  ↓
Try BGE-M3 (1024 dim)
  ↓
Dimension mismatch!
❌ Semua checkpoint KGC corrupt
```

### ✅ Approach Baru (Safe Teacher Features)

```
graph.pt (768 dim) ← student model, tidak pernah berubah
  ↓
data/distill_features/bge-base/     ← teacher 768 dim
data/distill_features/bge-m3/       ← teacher 1024 dim
data/distill_features/bge-small/    ← teacher 384 dim
  ↓
Trainer loads teacher embeddings secara dinamis
  ↓
MSE loss: student_pred vs teacher_pred
  ✅ Fleksibel! Bisa test banyak teacher models
  ✅ Aman! Graph chính tidak di-modify
  ✅ Checkpoint KGC tetap valid
```

---

## Perbandingan Kualitas vs Speed

| Teacher Model | Dim | Memory | Speed (T4) | Quality | Rekomendasi |
|---------------|-----|--------|-----------|---------|------------|
| bge-base | 768 | 1.1GB | baseline | baseline | Default |
| bge-m3 | 1024 | 1.5GB | -20% | +5% | ✅ Best balance |
| bge-small | 384 | 400MB | +60% | -10% | Minimum resource |
| bge-large | 1024 | 2.5GB | -40% | +8% | Jika VRAM cukup |

---

## Troubleshooting

### Error: "node_x.pt not found"

```
❌ FileNotFoundError: node_x.pt not found: data/distill_features/bge-m3/node_x.pt
```

**Fix:** Jalankan prepare_distill_teacher_features.py terlebih dahulu

### Error: "node_x has X nodes, expected Y"

```
❌ ValueError: node_x has 17252 nodes, expected 17262
```

**Possible causes:**
- source nodes.csv berbeda dari graph yang sekarang
- graph di-rebuild tanpa update nodes.csv

**Fix:** Gunakan nodes.csv dan graph tensor yang sama saat prepare

### Error: "embedding dims mismatch"

```
❌ ValueError: node_x dim 1024 != question_emb dim 768
```

**Reason:** Corruption dalam teacher features (sangat jarang)

**Fix:**
```bash
rm -rf data/distill_features/bge-m3/
python src/graph_retriever/prepare_distill_teacher_features.py \
  --teacher-model BAAI/bge-m3 \
  --output-dir data/distill_features/bge-m3 \
  --force
```

### Training sangat lambat dengan distillation

**Causes:**
- Full graph forward pass dengan batch_size=1
- Teacher feature loading overhead

**Mitigations:**
- Pastikan teacher features sudah di-cache (`.pt` files)
- Tidak perlu ulang prepare setiap epoch
- Normal untuk T4 16GB: ~2-3 jam per 35 epochs

---

## Next Steps

1. **Run Ablations:**
   - Hard only
   - Hard + BGE-base
   - Hard + BGE-M3

2. **Compare Results:**
   - Metrics improvement
   - Training time
   - Memory usage

3. **Deploy Best Model:**
   ```bash
   # Copy best checkpoint ke production
   cp outputs/graph_retriever/kgc_stage2_sft_distill_bge_m3/model_best.pth \
      models/stage2_production/
   ```

---

## References

- Prepare script: [`src/graph_retriever/prepare_distill_teacher_features.py`](../src/graph_retriever/prepare_distill_teacher_features.py)
- Loader: [`src/graph_retriever/distill_features.py`](../src/graph_retriever/distill_features.py)
- Config: [`configs/graph_retriever/stage2_sft_distill_bge_m3.yaml`](../configs/graph_retriever/stage2_sft_distill_bge_m3.yaml)
- README Section: [Ablation Studies](../README.md#ablation-studies-hard-labels--distillation)
