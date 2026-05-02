# Retrieval Benchmark Summary

## Run Configuration

- Query không tagged: `data/qa/test_qa.jsonl`
- Query có tagged: `data/qa/test_qa_tagged.jsonl`
- Chunk không tagged (baseline): `data/chunks/all_chunks.jsonl`
- Chunk có semantic tags: `data/chunks/all_chunks_tagged_paper_schema.jsonl` (được index trong `data/index_tagged_chunks`)
- Chế độ benchmark: retrieval-only (không generate answer), open-corpus, top-k = 5/10
- Hybrid mới: `RRF(BM25 + Dense BGE)` với `rrf_k=60`, sau đó rerank top-20 (lightweight heuristic).

## A) Baseline With Untagged Queries

| Method | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.342 | 0.193 | 0.211 | 0.558 | 0.340 | 0.242 | 0.218 |
| Dense BGE | 0.292 | 0.149 | 0.172 | 0.425 | 0.235 | 0.190 | 0.155 |
| Hybrid RRF (BM25 + Dense) + top-20 rerank | **0.692** | **0.392** | **0.627** | **0.742** | **0.476** | **0.634** | **0.459** |
| RAG_SEM (vector-only, tagged index) | 0.292 | 0.149 | 0.172 | 0.425 | 0.235 | 0.190 | 0.155 |

Ghi chú: khi query không có semantic tags, các biến thể `tag_boost`/`tag_filter` không tạo khác biệt so với bản gốc.

## B) Tagged Queries + Tagged Index

| Method | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.342 | 0.193 | 0.211 | 0.558 | 0.340 | 0.242 | 0.218 |
| BM25 + Tag Boost | 0.350 | 0.197 | 0.213 | 0.558 | 0.340 | 0.242 | 0.218 |
| BM25 + Tag Filter | 0.333 | 0.192 | 0.203 | 0.475 | 0.279 | 0.223 | 0.192 |
| Dense BGE | 0.292 | 0.149 | 0.172 | 0.425 | 0.235 | 0.190 | 0.155 |
| Dense BGE + Tag Boost | 0.325 | 0.186 | 0.170 | 0.575 | 0.341 | 0.203 | 0.201 |
| Dense BGE + Tag Filter | 0.300 | 0.174 | 0.165 | 0.533 | 0.312 | 0.194 | 0.185 |
| Hybrid RRF (BM25 + Dense) + top-20 rerank | **0.692** | **0.392** | **0.627** | **0.742** | **0.476** | **0.634** | **0.459** |
| Hybrid RRF + Tag Boost + top-20 rerank | 0.667 | 0.383 | 0.570 | 0.742 | 0.467 | 0.581 | 0.431 |
| Hybrid RRF + Tag Filter + top-20 rerank | 0.567 | 0.307 | 0.507 | 0.617 | 0.383 | 0.515 | 0.365 |
| RAG_SEM (vector-only, tagged index) | 0.292 | 0.149 | 0.172 | 0.425 | 0.235 | 0.190 | 0.155 |

## Conclusion

- Hybrid `RRF(BM25 + Dense)` + rerank top-20 vẫn cải thiện rất mạnh so với từng retriever đơn lẻ (ví dụ trên tagged: Recall@10 từ `0.340`/`0.235` tăng lên `0.476`, MRR@10 lên `0.634`).
- Với cấu hình top-20 hiện tại, `Tag Boost` không vượt hybrid base ở nhóm metric chính (đặc biệt MRR/Recall), và có trade-off rõ ở top-5.
- `Tag Filter` vẫn có xu hướng giảm recall nếu lọc quá cứng.
- Kết quả này là benchmark mở rộng ở tầng retrieval của project.
- Theo paper: semantic tags giúp chọn đúng vùng index, nhưng paper không mô tả công thức scoring/filtering cụ thể; vì vậy `tag_boost`, `tag_filter`, RRF fusion và bước rerank top-N trong benchmark này đều là implementation-specific.
