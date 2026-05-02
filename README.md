# 📊 MultiFinRAG SEC Filing RAG 
pipeline **RAG multimodal** cho các báo cáo SEC. Ý tưởng cốt lõi theo hướng **MultiFinRAG**:

* Xem **text, bảng (table), hình ảnh (image)** là các object độc lập để truy xuất
* Chuyển bảng và ảnh → **semantic text (summary)**
* Embed tất cả bằng **BGE**
* Lưu trữ và truy xuất bằng **FAISS**

Pipeline hiện tại:

* SEC HTML filings
* Convert sang PDF để xử lý text
* Table và image lấy trực tiếp từ HTML để đảm bảo độ chính xác

---

# 🧠 Tổng quan Pipeline

```text
PDF/HTML filings
|
+-- Text
|   +-- PyMuPDF extract text theo page
|   +-- tách câu (sentence split)
|   +-- sliding window + overlap
|   +-- embedding bằng BGE
|   +-- cosine distance giữa các câu
|   +-- breakpoint tại 95 percentile
|   +-- merge nếu similarity >= 0.85
|   +-- embed chunk cuối bằng BAAI/bge-base-en-v1.5
|
+-- Table
|   +-- parse HTML <table> bằng BeautifulSoup
|   +-- 1 bảng = 1 chunk
|   +-- giữ table_json từ HTML (source of truth)
|   +-- render bảng thành ảnh
|   +-- Gemma 3 27B Vision tạo semantic summary
|   +-- embed summary bằng BGE
|
+-- Image/chart
    +-- HTML <img>
    +-- tải ảnh gốc từ SEC
    +-- lọc ảnh meaningful
    +-- Gemma 3 27B Vision tạo summary
    +-- embed summary bằng BGE

FAISS
+-- text.faiss
+-- table.faiss
+-- image.faiss
+-- all.faiss (gộp tất cả)
```

---

# 📁 Cấu trúc Project

```text
src/
  common/
    bge_embedder.py
  ingestion/
    html_to_pdf.py
  chunking/
    rebuild_text_chunks_bge.py
    build_pdfminer_image_chunks.py
    render_table_assets.py
  vlm/
    nvidia_vlm_client.py
    download_sec_image_assets.py
    enrich_image_chunks_vlm.py
    enrich_table_chunks_vlm.py
  indexing/
    build_bge_index.py
    combine_bge_indexes.py
  retrieval/
    retrieve_bge.py
  semantic_tagging/
    tag_pipeline.py
    semantic_tagger.py
  graph_extraction/
    extract_chunk_graph.py
    build_graph_nodes_edges.py
    build_graph_tensor.py
    llm_openie_model.py
    prompts.py
    base_openie_model.py
  evaluation/
    generate_eval_qa.py

data/
  raw_filings/
  pdfs/
  chunks/
  qa/
  index_bge/
  graph/
```


# 📦 Dữ liệu chính

### Chunk files

```text
data/chunks/text_chunks.jsonl
data/chunks/table_chunks.jsonl
data/chunks/image_chunks.jsonl
data/chunks/all_chunks.jsonl
```

### Số lượng chunk

```text
text: 1318
table: 1847
image: 369
total: 3534
```

---

#  Evaluation QA

```text
eval: 150 câu
dev: 30 câu
test: 120 câu
```

Phân loại:

```text
text: 75
table: 35
image: 25
multimodal: 15
```

---

# Semantic Tagging

Semantic tagging currently follows only the paper schema from
`Multi-Document Financial Question Answering using LLMs`.

The `semantic_tags` schema for chunks and queries is exactly:

```json
{
  "named_entities": [],
  "dates": [],
  "industries": [],
  "domains": [],
  "sectors": [],
  "organizations": [],
  "partnerships": [],
  "partners": [],
  "dividends": [],
  "products": [],
  "locations": []
}
```

No extra semantic fields are added. The paper-schema output does not use
`chunk_role`, `evidence_type`, `section_tags`, `financial_metrics`,
`business_topics`, `risk_topics`, or `retrieval_keywords`.

## Tagged Files

Chunk tags:

```text
data/chunks/all_chunks_tagged_paper_schema.jsonl
```

Query tags for QA test split:

```text
data/qa/test_qa_tagged.jsonl
```

Each tagged chunk keeps the original chunk fields and adds:

```json
"semantic_tags": { "...": [] }
```

Each tagged QA item keeps the original QA fields and adds:

```json
"query_semantic_tags": { "...": [] }
```

## API Configuration

The taggers use an OpenAI-compatible chat API. For NVIDIA:

```powershell
$env:OPENAI_API_KEY="your_nvidia_key"
$env:OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1"
$env:OPENAI_MODEL="openai/gpt-oss-20b"
```

The local `.env` file can contain the same values:

```text
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
OPENAI_MODEL=openai/gpt-oss-20b
```

## Tag All Chunks

Run from scratch:

```bash
python src/semantic_tagging/tag_pipeline.py --input data/chunks/all_chunks.jsonl --output data/chunks/all_chunks_tagged_paper_schema.jsonl --model openai/gpt-oss-20b --overwrite
```

Resume:

```bash
python src/semantic_tagging/tag_pipeline.py --input data/chunks/all_chunks.jsonl --output data/chunks/all_chunks_tagged_paper_schema.jsonl --model openai/gpt-oss-20b --resume
```

Dry run writes empty paper-schema tags without calling the LLM:

```bash
python src/semantic_tagging/tag_pipeline.py --input data/chunks/all_chunks.jsonl --output data/chunks/all_chunks_tagged_paper_schema.jsonl --dry-run --overwrite
```

## Tag QA Queries

Tag all test QA queries:

```bash
python src/semantic_tagging/tag_query_pipeline.py --input data/qa/test_qa.jsonl --output data/qa/test_qa_tagged.jsonl --model openai/gpt-oss-20b --overwrite
```

Resume:

```bash
python src/semantic_tagging/tag_query_pipeline.py --input data/qa/test_qa.jsonl --output data/qa/test_qa_tagged.jsonl --model openai/gpt-oss-20b --resume
```

Tag one query for inspection:

```bash
python src/semantic_tagging/query_tagger.py --question "What risks does Apple mention about tariffs in 2025?" --model openai/gpt-oss-20b
```

Logs:

```text
tagging_errors.log
query_tagging_errors.log
```

---

# Knowledge Graph từ chunk (OpenIE + LLM)

Module `src/graph_extraction` trích **thực thể** và **triple quan hệ** từ nội dung chunk báo cáo tài chính (OpenIE hai bước: NER rồi trích triple), gọi API **OpenAI-compatible** giống semantic tagging (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` trong `.env` hoặc biến môi trường).

Văn bản đưa vào LLM ưu tiên theo thứ tự: `text` → `summary` → `embed_text`. Một đoạn “focus” có thể được cắt quanh các từ mang tính biến động (vd. increased/decreased) khi passage quá dài; **không bắt buộc** chunk phải có quan hệ đó.

## Schema trong `chunk_graph.jsonl`

Mỗi dòng là một chunk gốc kèm trường `graph`:

```json
{
  "id": "...",
  "text": "...",
  "graph": {
    "entities": ["..."],
    "triples": [["subject", "relation_canonical", "object"]],
    "clean_triples": [["...", "...", "..."]],
    "noisy_triples": [{"triple": [...], "reason": "..."}]
  }
}
```

- `clean_triples`: triple đã qua kiểm tra (grounding trong đoạn, chuẩn hóa relation, và ít nhất một đầu mút trùng NER).
- `noisy_triples`: các triple bị loại kèm lý do để audit.
- `triples`: trùng nội dung chính của `clean_triples` sau bước xử lý trong model.

## Bước 1 — Trích graph (LLM)

Chạy từ đầu (ghi đè file output):

```bash
python src/graph_extraction/extract_chunk_graph.py --input data/chunks/all_chunks.jsonl --output data/graph/chunk_graph.jsonl --summary-out data/graph/graph_summary.json --overwrite
```

Tiếp tục sau khi dừng giữa chừng (giữ các dòng đã có `graph`, chỉ xử lý chunk thiếu):

```bash
python src/graph_extraction/extract_chunk_graph.py --input data/chunks/all_chunks.jsonl --output data/graph/chunk_graph.jsonl --summary-out data/graph/graph_summary.json --resume
```

Tham số hữu ích:

```text
--model <ten_model>          # ghi đè OPENAI_MODEL
--max-chunks N               # thử nhanh trên N chunk đầu
--enable-delta-heuristic     # tùy chọn: chỉ khi LLM ra quá ít triple, fallback nhẹ cho câu có từ khóa tăng/giảm (mặc định tắt để không ép buộc)
--log-file graph_extraction_errors.log
```

Output:

```text
data/graph/chunk_graph.jsonl
data/graph/graph_summary.json
graph_extraction_errors.log
```

## Bước 2 — Gộp nodes / relations / edges (định dạng gần GFM-RAG)

Từ `chunk_graph.jsonl` sinh các file JSONL và CSV (cột CSV tương thích ý đồ `nodes.csv`, `relations.csv`, `edges.csv` của luồng bring-your-own-graph trong GFM-RAG: `uid`/`name`/`type`/`attributes`, `source`/`relation`/`target`/`attributes`, v.v.). Node loại chunk dùng `chunk:<chunk_id>`; node entity có id băm cố định từ tên đã normalize.

Khuyến nghị để có graph “đầy đủ” nhưng không lọc quá gắt (chỉ bỏ self-loop và object boolean yes/no và vài relation trạng thái nội bộ như `filer_status` / `shell_company_status` khi dùng `--drop-low-value-edges`; **đã không** loại triple chỉ vì object là số liệu):

```bash
python src/graph_extraction/build_graph_nodes_edges.py \
  --input data/graph/chunk_graph.jsonl \
  --nodes-out data/graph/nodes.jsonl \
  --edges-out data/graph/edges.jsonl \
  --relations-out data/graph/relations.jsonl \
  --nodes-csv-out data/graph/nodes.csv \
  --edges-csv-out data/graph/edges.csv \
  --relations-csv-out data/graph/relations.csv \
  --summary-out data/graph/graph_nodes_edges_summary.json \
  --add-equivalent-edges \
  --drop-low-value-edges \
  --use-default-relation-whitelist
```

Nếu muốn giữ **mọi** relation do LLM trích (không whitelist), bỏ `--use-default-relation-whitelist` và có thể bỏ `--drop-low-value-edges`.

Tham số khác:

```text
--relation-whitelist rel1,rel2,...   # cộng thêm hoặc chỉ định (kết hợp được với default nếu cần logic riêng: chạy không flag default và chỉ dùng whitelist tùy bạn)
```

## Một lệnh full pipeline + resume trích LLM

Sau khi đã có `chunk_graph.jsonl` một phần, chạy lệnh sau sẽ **resume** trích chunk thiếu rồi build lại toàn bộ nodes/relations/edges:

```bash
python src/graph_extraction/extract_chunk_graph.py --input data/chunks/all_chunks.jsonl --output data/graph/chunk_graph.jsonl --summary-out data/graph/graph_summary.json --resume && python src/graph_extraction/build_graph_nodes_edges.py --input data/graph/chunk_graph.jsonl --nodes-out data/graph/nodes.jsonl --edges-out data/graph/edges.jsonl --relations-out data/graph/relations.jsonl --nodes-csv-out data/graph/nodes.csv --edges-csv-out data/graph/edges.csv --relations-csv-out data/graph/relations.csv --summary-out data/graph/graph_nodes_edges_summary.json --add-equivalent-edges --drop-low-value-edges --use-default-relation-whitelist
```

## Step 3 - Convert CSV graph to tensor graph for GFM/GNN

The CSV graph can be converted to a PyTorch Geometric `Data` object. This keeps
the original directed edges as `target_edge_index` / `target_edge_type`, and also
adds inverse edges to `edge_index` / `edge_type` for message passing.

```bash
python src/graph_extraction/build_graph_tensor.py \
  --graph-dir data/graph \
  --output-dir data/graph_tensor
```

Output:

```text
data/graph_tensor/graph.pt
data/graph_tensor/node2id.json
data/graph_tensor/id2node.json
data/graph_tensor/rel2id.json
data/graph_tensor/tensor_summary.json
```

By default this creates structural tensors only (`x=None`, `rel_attr=None`,
`feat_dim=0`). To add BGE node/relation features:

```bash
python src/graph_extraction/build_graph_tensor.py \
  --graph-dir data/graph \
  --output-dir data/graph_tensor_bge \
  --embed-features \
  --embedding-model BAAI/bge-base-en-v1.5
```

## Step 4 - Train GFM-RAG graph reasoning

This project keeps the financial graph in `data/graph`, while the GFM-RAG code
is used as the model/trainer implementation. The adapter below converts the
local graph and benchmark queries into the GFM-RAG stage1 format:

```bash
python src/graph_reasoning/prepare_gfmrag_dataset.py \
  --graph-dir data/graph \
  --queries data/benchmark_report/queries_tagged.jsonl \
  --output-root data/gfmrag_reasoning \
  --data-name multifin_graph
```

It creates:

```text
data/gfmrag_reasoning/multifin_graph/raw/documents.json
data/gfmrag_reasoning/multifin_graph/processed/stage1/nodes.csv
data/gfmrag_reasoning/multifin_graph/processed/stage1/relations.csv
data/gfmrag_reasoning/multifin_graph/processed/stage1/edges.csv
data/gfmrag_reasoning/multifin_graph/processed/stage1/train.json
data/gfmrag_reasoning/multifin_graph/processed/stage1/test.json
```

Train the reasoning model by pointing to the local `gfm-rag` repo:

```bash
python src/graph_reasoning/train_gfmrag_reasoner.py \
  --config configs/gfmrag_reasoning/sft_training.yaml \
  --gfmrag-path d:/Project/gfm-rag
```

The config uses `gfmrag.models.gfm_rag_v1.GNNRetriever` with
`QueryNBFNet`. It reasons over entity nodes and learns to rank evidence chunk
nodes through the `entity -> chunk` `is_mentioned_in` edges. The main supervised
target node type is `chunk`.

The local config sets `disable_custom_rspmm: true`, so `QueryNBFNet` uses the
standard PyG message-passing fallback and runs on the current Windows/CPU
environment without compiling Ultra's `rspmm` C++ extension. This is slower but
portable. Set `disable_custom_rspmm: false` only when running in an environment
that can compile PyTorch extensions, such as Linux/CUDA or Windows with Visual
Studio C++ Build Tools (`cl.exe`).

---

# Rebuild Pipeline

## 1. Convert HTML → PDF

```bash
python src/ingestion/html_to_pdf.py
```

---

## 2. Text Chunking

```bash
python src/chunking/rebuild_text_chunks_bge.py
```

---

## 3. Image Chunking

Download ảnh:

```bash
python src/vlm/download_sec_image_assets.py
```

Summary bằng VLM:

```bash
python src/vlm/enrich_image_chunks_vlm.py --replace-current
```

---

## 4. Table Chunking

Render bảng:

```bash
python src/chunking/render_table_assets.py
```

Summary:

```bash
python src/vlm/enrich_table_chunks_vlm.py --replace-current
```



* Số liệu bảng lấy từ HTML (không dùng VLM)
* VLM chỉ dùng để tạo semantic summary

---

## 5. Build FAISS Index

```bash
python src/indexing/build_bge_index.py --index all --batch-size 64
```

Hoặc build riêng:

```bash
python src/indexing/build_bge_index.py --index table --batch-size 64
python src/indexing/combine_bge_indexes.py
```

---

## 6. Retrieval

```bash
python src/retrieval/retrieve_bge.py "restricted cash" --index table --top-k 3

python src/retrieval/retrieve_bge.py "Daily 95% One-Day Total Management VaR" --index image --top-k 3

python src/retrieval/retrieve_bge.py "net revenue increased" --index all --top-k 5
```

---

## 7. FinanceBench Retrieval Evaluation

FinanceBench khong co ground-truth `chunk_id`. File benchmark chi co `question`, `answer`, `evidence_text`, `evidence_page_num`, `evidence_doc_name/doc_name` va PDF goc. Vi vay evaluation script se tu map evidence vao chunk do pipeline cua minh tao ra:

1. exact match `evidence_text` trong `chunk.text`
2. fuzzy/token-overlap match neu exact match khong duoc
3. fallback theo page neu van khong match duoc

Chay sau khi da chunk va build FAISS index cho PDF FinanceBench:

```bash
python src/evaluation/financebench_retrieval_eval.py ^
  --benchmark benchmark/financebench/data/financebench_open_source.jsonl ^
  --chunks data/chunks/all_chunks.jsonl ^
  --index data/index_bge/all.faiss ^
  --ids data/index_bge/all_chunk_ids.json ^
  --k-values 1,3,5,10 ^
  --out-dir data/eval/financebench_retrieval
```

Output:

```text
data/eval/financebench_retrieval/qrels.jsonl
data/eval/financebench_retrieval/qrel_misses.jsonl
data/eval/financebench_retrieval/runs.jsonl
data/eval/financebench_retrieval/per_question_metrics.jsonl
data/eval/financebench_retrieval/aggregate_metrics.csv
data/eval/financebench_retrieval/report.md
```

Metrics gom `Precision@k`, `Recall@k`, `Hit@k`, `MRR@k`, `Page Recall@k`, va `Evidence Recall@k`.

---

## 7.1 Retrieval Benchmark cho RAG_SEM (tagged vs untagged)

Muc nay danh gia retrieval-only cho pipeline RAG_SEM trong project:

- So sanh query **khong tagged** (`data/qa/test_qa.jsonl`) va query **co tagged** (`data/qa/test_qa_tagged.jsonl`)
- So sanh cac method: BM25, Dense BGE, tag-aware variants, va Hybrid `RRF(BM25 + Dense)`
- Co them buoc rerank top-N sau fusion (hien tai thu nghiem top-20)

Luu y quan trong:

- Khong retag lai chunks
- Khong retag lai queries
- Khong re-embed chunks
- Chi retrieve va tinh metric retrieval

### Du lieu dau vao

```text
data/chunks/all_chunks.jsonl
data/chunks/all_chunks_tagged_paper_schema.jsonl
data/index_bge/all.faiss
data/index_bge/all_chunk_ids.json
data/index_tagged_chunks/chunks.faiss
data/index_tagged_chunks/chunk_ids.json
data/qa/test_qa.jsonl
data/qa/test_qa_tagged.jsonl
```

### Lenh chay benchmark (query khong tagged)

```powershell
python -m src.evaluation.evaluate_rag_sem_retrieval `
  --queries data/qa/test_qa.jsonl `
  --tagged-index-dir data/index_tagged_chunks `
  --baseline-index data/index_bge/all.faiss `
  --baseline-ids data/index_bge/all_chunk_ids.json `
  --baseline-chunks data/chunks/all_chunks.jsonl `
  --out-dir outputs/retrieval_benchmark_report_untagged_queries_hybrid_rrf_top20 `
  --k-values 5 10 `
  --candidate-k 50 `
  --include-bm25 `
  --include-tag-aware `
  --include-hybrid-rrf `
  --rrf-k 60 `
  --hybrid-rerank-top-n 20 `
  --hybrid-rerank-weight 1.0 `
  --hybrid-rerank-batch-size 32
```

### Lenh chay benchmark (query co tagged)

```powershell
python -m src.evaluation.evaluate_rag_sem_retrieval `
  --queries data/qa/test_qa_tagged.jsonl `
  --tagged-index-dir data/index_tagged_chunks `
  --baseline-index data/index_bge/all.faiss `
  --baseline-ids data/index_bge/all_chunk_ids.json `
  --baseline-chunks data/chunks/all_chunks.jsonl `
  --out-dir outputs/retrieval_benchmark_report_tagged_queries_hybrid_rrf_top20 `
  --k-values 5 10 `
  --candidate-k 50 `
  --include-bm25 `
  --include-tag-aware `
  --include-hybrid-rrf `
  --rrf-k 60 `
  --hybrid-rerank-top-n 20 `
  --hybrid-rerank-weight 1.0 `
  --hybrid-rerank-batch-size 32
```

### File output chinh

```text
outputs/retrieval_benchmark_report_untagged_queries_hybrid_rrf_top20/metrics_summary.json
outputs/retrieval_benchmark_report_untagged_queries_hybrid_rrf_top20/metrics_by_question.jsonl
outputs/retrieval_benchmark_report_untagged_queries_hybrid_rrf_top20/retrieval_results_all_modes.jsonl
outputs/retrieval_benchmark_report_tagged_queries_hybrid_rrf_top20/metrics_summary.json
outputs/retrieval_benchmark_report_tagged_queries_hybrid_rrf_top20/metrics_by_question.jsonl
outputs/retrieval_benchmark_report_tagged_queries_hybrid_rrf_top20/retrieval_results_all_modes.jsonl
outputs/retrieval_benchmark_report_summary.md
```

### Cach doc ket qua nhanh

- Dung `metrics_summary.json` de so method theo `Hit@k`, `Recall@k`, `MRR@k`, `NDCG@k`
- Dung `metrics_by_question.jsonl` de xem query nao huong loi/bi giam khi bat tag-aware hoac hybrid
- Dung `retrieval_results_all_modes.jsonl` de debug chi tiet top chunks theo tung method
- `tag_boost` va `tag_filter` la bien the implementation-specific cua project (paper khong cho cong thuc scoring/filtering cu the)

---

## 8. FinanceBench Semantic BGE Retrieval Result

Da tao benchmark retrieval tu FinanceBench open-source bang semantic text chunking BGE.

Input:

```text
benchmark/financebench/data/financebench_open_source.jsonl
benchmark/financebench/pdfs/
```

Chunking:

```text
PDF -> sentence split -> BGE sentence embedding -> semantic breakpoints -> text chunks
```

Do may hien tai khong co CUDA, da chay CPU voi:

```text
BAAI/bge-small-en-v1.5
```

Output chunks:

```text
outputs/financebench_eval_bge/chunks.jsonl
```

Chay semantic chunking:

```powershell
python -m rag_eval.src.build_financebench_bge_chunks `
  --financebench benchmark/financebench/data/financebench_open_source.jsonl `
  --pdf-dir benchmark/financebench/pdfs `
  --output outputs/financebench_eval_bge/chunks.jsonl `
  --model BAAI/bge-small-en-v1.5 `
  --batch-size 32
```

Tao retrieval benchmark tu chunks:

```powershell
python -m rag_eval.src.create_qrels `
  --financebench benchmark/financebench/data/financebench_open_source.jsonl `
  --chunks outputs/financebench_eval_bge/chunks.jsonl `
  --out-dir outputs/financebench_eval_bge/qrels
```

File danh gia duoc tao:

```text
outputs/financebench_eval_bge/qrels/queries.jsonl
outputs/financebench_eval_bge/qrels/qrels.jsonl
outputs/financebench_eval_bge/qrels/unmatched_evidence.jsonl
outputs/financebench_eval_bge/qrels/mapping_report.json
```

Thong ke chunks/qrels:

```text
semantic chunks:      9927
docs_with_chunks:     84
questions:            150
evidence_rows:        189
qrels raw:            258
qrels active:         248
unmatched_evidence:   0
match_method_counts:  fuzzy_text=182, exact_text=66
```

Da audit va loc bot qrels dang ngo:

```text
Tieu chi loc: fuzzy_text co token_recall < 0.60
So qrels bi loai: 10
So query bi anh huong: 7
So query con lai trong benchmark: 150/150
remaining low-token fuzzy qrels: 0
```

Khong loai tat ca `page_match=False`, vi semantic chunk co the gom nhieu page hoac page index cua parser PDF lech voi FinanceBench. Neu `token_recall` van du cao thi qrel duoc giu lai.

File qrels:

```text
outputs/financebench_eval_bge/qrels/qrels.jsonl                  # active, da loc
outputs/financebench_eval_bge/qrels/qrels_unfiltered.jsonl       # backup ban raw
outputs/financebench_eval_bge/qrels/removed_qrels_suspicious.jsonl
outputs/financebench_eval_bge/qrels/mapping_report.json
```

Phan bo query -> relevant chunks:

```text
1 chunk relevant : 103 queries
2 chunks relevant: 30 queries
3 chunks relevant: 11 queries
4 chunks relevant: 5 queries
5 chunks relevant: 1 query
```

Evaluation output:

```text
outputs/financebench_eval_bge/retrieval_eval/
```

Chay retrieval evaluation trong notebook:

```text
notebooks/financebench_current_retrieval_hybrid_eval.ipynb
notebooks/financebench_fast_hybrid_retrieval_eval.ipynb
```

Audit qua trinh tao benchmark:

```text
notebooks/financebench_retrieval_benchmark_audit.ipynb
```

File ket qua evaluation:

```text
outputs/financebench_eval_bge/retrieval_eval_filtered/bm25_retrieval_results.jsonl
outputs/financebench_eval_bge/retrieval_eval_filtered/bge_dense_retrieval_results.jsonl
outputs/financebench_eval_bge/retrieval_eval_filtered/hybrid_retrieval_results.jsonl
outputs/financebench_eval_bge/retrieval_eval_filtered/metadata_aware_retrieval_results.jsonl
outputs/financebench_eval_bge/retrieval_eval_filtered/retrieval_results.jsonl
outputs/financebench_eval_bge/retrieval_eval_filtered/metrics_summary.json
outputs/financebench_eval_bge/retrieval_eval_filtered/metrics_by_question.jsonl
outputs/financebench_eval_bge/retrieval_eval_filtered/retrieval_eval_report.md
```

Chay lai evaluation filtered:

```powershell
python -m rag_eval.src.run_financebench_filtered_retrieval_eval `
  --base-dir outputs/financebench_eval_bge `
  --out-dir outputs/financebench_eval_bge/retrieval_eval_filtered `
  --top-k 10 `
  --candidate-k 50 `
  --hybrid-alpha 0.60 `
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --reranker-batch-size 8
```

Ket qua retrieval:

| Method | Hit@1 | Recall@1 | MRR@1 | Hit@10 | Recall@10 | MRR@10 | Precision@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bge_dense | 0.1067 | 0.0650 | 0.1067 | 0.3400 | 0.2694 | 0.1695 | 0.0387 |
| bm25 | 0.0600 | 0.0500 | 0.0600 | 0.1733 | 0.1539 | 0.0947 | 0.0213 |
| bm25_bge_hybrid_alpha_0.60 | 0.1000 | 0.0706 | 0.1000 | 0.3267 | 0.2639 | 0.1563 | 0.0387 |
| bm25_bge_hybrid_alpha_0.60_reranker_cross_encoder_ms_marco_minilm_l_6_v2 | 0.1067 | 0.0722 | 0.1067 | 0.3133 | 0.2522 | 0.1605 | 0.0373 |
| metadata_doc_filter_bm25_bge_hybrid_alpha_0.60 | 0.2067 | 0.1617 | 0.2067 | 0.5333 | 0.4596 | 0.3018 | 0.0680 |

Nhan xet:

```text
bge_dense dang tot nhat trong open-corpus retrieval theo Hit@10/Recall@10.
BM25 thuan thap hon dense.
Hybrid alpha=0.60 gan bang dense nhung chua vuot dense.
Reranker nhe cross-encoder cai thien Recall@1 so voi dense/hybrid, nhung giam Hit@10/Recall@10 vi rerank lai top-50.
metadata_doc_filter la cau hinh oracle biet truoc doc_name, khong so sanh truc tiep voi open-corpus retrieval.
FinanceBench BGE chunks hien tai chi la text chunks, nen chua co table-aware retrieval rieng.
```

File metric chinh:

```text
outputs/financebench_eval_bge/retrieval_eval_filtered/metrics_summary.json
```


