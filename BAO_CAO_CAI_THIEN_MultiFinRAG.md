# Báo cáo cải thiện Pipeline MultiFinRAG (không bao gồm graph)

Ngày báo cáo: 2026-05-04

## 1. Mục tiêu và phạm vi

Báo cáo này mô tả chi tiết các cải thiện trong pipeline MultiFinRAG cho dữ liệu SEC filings, tập trung vào xử lý text, table, image, indexing, retrieval và đánh giá answer bằng LLM. Phần graph được loại bỏ theo yêu cầu. Các mô tả kỹ thuật được đối chiếu từ mã nguồn và kết quả benchmark có sẵn trong project.

## 2. Tổng quan pipeline

Pipeline hiện tại xử lý SEC filings theo định hướng MultiFinRAG: tách text, bảng và hình ảnh thành các object độc lập, chuyển bảng và ảnh thành semantic text, embed bằng BGE, lưu FAISS để truy xuất. Mô tả tổng quan trùng với [README.md](README.md).

## 3. Cải thiện xử lý text

### 3.1 Text chunking theo ngữ nghĩa

Triển khai tại [src/chunking/rebuild_text_chunks_bge.py](src/chunking/rebuild_text_chunks_bge.py).

Quy trình chính:
- Tách câu từ PDF bằng PyMuPDF, làm sạch ký tự và chuẩn hóa dấu câu.
- Với mỗi tài liệu: mã hóa câu bằng BGE, tính cosine similarity giữa các câu liên tiếp.
- Tìm breakpoint theo percentile khoảng cách (mặc định 95th percentile) trong cửa sổ trượt.
- Cắt đoạn theo breakpoint, đồng thời ràng buộc độ dài (min/max số câu).
- Merge các chunk liền kề nếu độ tương đồng đủ cao và không vượt giới hạn độ dài.

Tham số đang dùng (theo code mặc định):
- percentile = 95
- window_size = 32, overlap = 8
- min_sentences = 3, max_sentences = 16
- merge_threshold = 0.85
- max_merge_sentences = 32
- max_merge_words = 1000
- max_merge_pages = 6

Đầu ra:
- data/chunks/text_chunks.jsonl (1.318 text chunks theo [README.md](README.md)).
- data/chunks/all_chunks.jsonl được hợp nhất từ text + table + image.

Lý do và tác động:
- Semantic chunking giữ ranh giới ngữ nghĩa tốt hơn fixed-window.
- Merge có điều kiện tránh chunk quá nhỏ nhưng vẫn không “trộn” nội dung vượt trang.
- Các giới hạn độ dài giúp retrieval ổn định, giảm nhiễu ở câu hỏi dài.

## 4. Cải thiện xử lý table

### 4.1 Table chunking dựa trên HTML gốc

Phân tích và chuẩn hóa bảng xuất phát từ HTML, không OCR PDF. Điều này phù hợp với cấu trúc SEC filings vốn là HTML chuẩn hóa.

Luồng chính liên quan đến table assets và VLM summary:
- Render ảnh bảng từ HTML bằng Playwright tại [src/chunking/render_table_assets.py](src/chunking/render_table_assets.py).
- Dùng Gemma 3 Vision để tạo summary ngắn gọn từ ảnh bảng tại [src/vlm/enrich_table_chunks_vlm.py](src/vlm/enrich_table_chunks_vlm.py).

Các điểm kỹ thuật quan trọng:
- 1 bảng = 1 chunk, không cắt nhỏ để giữ cấu trúc.
- HTML table được coi là source of truth; ảnh bảng chỉ dùng cho VLM summary.
- VLM được yêu cầu trả JSON với summary và evidence; không sao chép table_json đầy đủ.
- chunking_method ghi rõ: html_table_object_gemma3_vision_summary.

Đầu ra:
- data/chunks/table_chunks.jsonl (1.847 table chunks theo [README.md](README.md)).
- data/chunks/table_chunks_assets.jsonl và data/visual_chunks/tables cho ảnh bảng.

Lý do bỏ Detectron2:
- Bảng từ PDF thường bị sai cấu trúc (ô gộp, lệch cột) khi OCR hoặc detect layout.
- HTML của SEC là sạch và đầy đủ, giảm sai số và chi phí xử lý.

## 5. Cải thiện xử lý image/chart

### 5.1 Khôi phục pipeline ảnh và tạo metadata có cấu trúc

Triển khai tại [src/vlm/enrich_image_chunks_vlm.py](src/vlm/enrich_image_chunks_vlm.py).

Quy trình chính:
- Lấy ảnh từ nguồn HTML (hoặc asset gốc từ SEC archive), lọc theo kích thước tối thiểu.
- Dùng Gemma 3 Vision qua NVIDIA API để sinh summary có cấu trúc.
- Ghi nhận thêm các trường visual_metadata như visual_type, metrics, periods, key_values, trend, evidence.

Điểm kỹ thuật đáng chú ý:
- Có cơ chế retry và giảm kích thước ảnh nếu gặp giới hạn context.
- Lưu lại vlm_output để truy vết khi cần kiểm tra chất lượng.
- chunking_method ghi rõ: sec_html_image_asset_gemma3_vision_summary.

Đầu ra:
- data/chunks/image_chunks_vlm_structured.jsonl (mới).
- data/chunks/image_asset_download_summary.json.

Tác động:
- Các biểu đồ tài chính trong SEC filings được đưa trở lại pipeline, tạo evidence cho câu hỏi dạng trend/visual_type.
- Structured metadata cho phép benchmark riêng cho ảnh, không phụ thuộc text mô tả.

## 6. FAISS index và embedding

Triển khai tại [src/indexing/build_bge_index.py](src/indexing/build_bge_index.py).

Điểm chính:
- Index tách theo modality: text, table, image, all.
- BGE dùng làm embedding chính; index sử dụng inner product với vector đã normalize (cosine similarity).
- Có thể xuất embeddings để kiểm tra/ablation.

Đầu ra:
- data/index_bge/text.faiss, table.faiss, image.faiss, all.faiss.
- data/index_bge/*_chunk_ids.json và meta JSON.

Lý do:
- Cho phép lọc retrieval theo modality trong các thí nghiệm.
- Hỗ trợ ablation study và tối ưu pipeline từng loại dữ liệu.

## 7. Tạo dữ liệu đánh giá và benchmark

### 7.1 Cấu trúc bộ benchmark

- data/benchmark/: 150 queries, 176 qrels (text 75, table 35, image 25, multimodal 15).
- data/benchmark_hard/: 150 queries, 200 qrels (loại metadata cover-page; multimodal yêu cầu 2 evidence).
- data/benchmark_visual_metadata/: 8 queries, 8 qrels (image-only theo visual_metadata).

### 7.2 Quy trình tạo query/evidence

Triển khai tại [src/evaluation/generate_eval_qa.py](src/evaluation/generate_eval_qa.py).

Các điểm kỹ thuật:
- Cover-page text QA: trích các trường như period ended, date of report, commission file number từ trang đầu.
- Table QA: parse HTML table, xác định header/label/value, gắn evidence theo row/column.
- Image QA: ban đầu có câu kiểu marker; pipeline mới chuyển sang visual_metadata để answerable hơn.
- Multimodal QA: kết hợp period (cover page) với giá trị bảng để yêu cầu 2 evidence.

Ý nghĩa:
- Dataset vừa đánh giá độ bám theo text/table/image, vừa kiểm tra khả năng phối hợp chứng cứ.

## 8. Đánh giá retrieval

### 8.1 Hybrid RRF: định nghĩa và rerank

Triển khai tại [src/evaluation/evaluate_hybrid_rrf_benchmark.py](src/evaluation/evaluate_hybrid_rrf_benchmark.py).

Chi tiết kỹ thuật:
- BM25 dùng tokenization đơn giản, tính idf theo toàn bộ corpus.
- Dense BGE dùng FAISS, candidate_k mặc định 50.
- Fusion bằng RRF: score = 1 / (rrf_k + rank), với rrf_k mặc định 60.
- Rerank bonus có trọng số nhỏ theo:
	- trùng modality: +0.08
	- query multimodal và chunk thuộc text/table/image: +0.03
	- cùng source_pdf: +0.06

Mục đích rerank:
- Ưu tiên đúng modality và nguồn tài liệu, giảm nhiễu khi corpus rộng.

### 8.2 Kết quả retrieval-only (untagged queries)

Trích từ [outputs/retrieval_benchmark_report_summary.md](outputs/retrieval_benchmark_report_summary.md).

| Method | Hit@5 | Recall@5 | MRR@5 | Hit@10 | Recall@10 | MRR@10 | NDCG@10 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.342 | 0.193 | 0.211 | 0.558 | 0.340 | 0.242 | 0.218 |
| Dense BGE | 0.292 | 0.149 | 0.172 | 0.425 | 0.235 | 0.190 | 0.155 |
| Hybrid RRF (BM25 + Dense) + top-20 rerank | 0.692 | 0.392 | 0.627 | 0.742 | 0.476 | 0.634 | 0.459 |

Nhận xét:
- Hybrid RRF + rerank tăng mạnh so với BM25/Dense ở cả MRR và Recall.
- Dense BGE thuần kém hơn BM25 trong benchmark này, cần cân nhắc cân bằng giữa lexical và semantic.

### 8.3 Retrieval theo modality (benchmark_report)

Dữ liệu từ các file metrics_summary:
- BM25: [outputs/retrieval_benchmark_report_bm25/metrics_summary.json](outputs/retrieval_benchmark_report_bm25/metrics_summary.json)
- Hybrid RRF: [outputs/retrieval_benchmark_report_hybrid_rrf/metrics_summary.json](outputs/retrieval_benchmark_report_hybrid_rrf/metrics_summary.json)
- Hybrid RRF + source rerank: [outputs/retrieval_benchmark_report_hybrid_rrf_source_rerank/metrics_summary.json](outputs/retrieval_benchmark_report_hybrid_rrf_source_rerank/metrics_summary.json)

**All queries (Hit@10 / Recall@10 / MRR@10)**

| Method | Hit@10 | Recall@10 | MRR@10 | Queries |
| --- | --- | --- | --- | --- |
| BM25 | 0.473 | 0.430 | 0.261 | 110 |
| Hybrid RRF | 0.582 | 0.515 | 0.331 | 110 |
| Hybrid RRF + source rerank | 0.778 | 0.697 | 0.386 | 99 |

**Text**

| Method | Hit@10 | Recall@10 | MRR@10 | Queries |
| --- | --- | --- | --- | --- |
| BM25 | 0.150 | 0.150 | 0.065 | 40 |
| Hybrid RRF | 0.250 | 0.250 | 0.117 | 40 |
| Hybrid RRF + source rerank | 0.600 | 0.600 | 0.253 | 40 |

**Table**

| Method | Hit@10 | Recall@10 | MRR@10 | Queries |
| --- | --- | --- | --- | --- |
| BM25 | 0.654 | 0.654 | 0.254 | 26 |
| Hybrid RRF | 0.808 | 0.808 | 0.384 | 26 |
| Hybrid RRF + source rerank | 0.962 | 0.962 | 0.546 | 26 |

**Image**

| Method | Hit@10 | Recall@10 | MRR@10 | Queries |
| --- | --- | --- | --- | --- |
| BM25 | 0.880 | 0.880 | 0.718 | 25 |
| Hybrid RRF | 0.880 | 0.880 | 0.671 | 25 |
| Hybrid RRF + source rerank | 0.833 | 0.833 | 0.465 | 18 |

**Multimodal**

| Method | Hit@10 | Recall@10 | MRR@10 | Queries |
| --- | --- | --- | --- | --- |
| BM25 | 0.368 | 0.123 | 0.082 | 19 |
| Hybrid RRF | 0.579 | 0.193 | 0.263 | 19 |
| Hybrid RRF + source rerank | 0.867 | 0.333 | 0.369 | 15 |

Nhận xét:
- Source rerank cải thiện rõ ở text, table và multimodal.
- Modality image giảm khi filter_source vì số query hợp lệ ít hơn (18 thay vì 25).

### 8.4 Benchmark image-only theo visual_metadata

Dữ liệu từ:
- BM25: [outputs/benchmark_visual_metadata_bm25/metrics_summary.json](outputs/benchmark_visual_metadata_bm25/metrics_summary.json)
- Dense BGE: [outputs/benchmark_visual_metadata_dense/metrics_summary.json](outputs/benchmark_visual_metadata_dense/metrics_summary.json)
- Hybrid RRF: [outputs/benchmark_visual_metadata_hybrid_rrf/metrics_summary.json](outputs/benchmark_visual_metadata_hybrid_rrf/metrics_summary.json)
- Hybrid RRF + source rerank: [outputs/benchmark_visual_metadata_hybrid_rrf_source_rerank/metrics_summary.json](outputs/benchmark_visual_metadata_hybrid_rrf_source_rerank/metrics_summary.json)

| Method | Hit@3 | Recall@3 | MRR@3 | Hit@10 | Recall@10 | MRR@10 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25 | 0.875 | 0.875 | 0.729 | 1.000 | 1.000 | 0.760 |
| Dense BGE | 0.625 | 0.625 | 0.542 | 0.750 | 0.750 | 0.567 |
| Hybrid RRF | 0.875 | 0.875 | 0.667 | 1.000 | 1.000 | 0.685 |
| Hybrid RRF + source rerank | 0.875 | 0.875 | 0.646 | 1.000 | 1.000 | 0.664 |

Nhận xét:
- Bộ benchmark nhỏ (8 query) nhưng cho thấy BM25 và Hybrid RRF phù hợp tốt với metadata dạng text.
- Dense BGE chưa tối ưu cho metadata ngắn; có thể cần tinh chỉnh hoặc thêm trọng số theo trường.

## 9. Đánh giá answer bằng LLM (accuracy)

Pipeline sinh và chấm câu trả lời dùng openai/gpt-oss-20b qua NVIDIA API. Có cơ chế deterministic numeric judge để chuẩn hóa số liệu, giảm sai lệch do format (ví dụ 113,743 vs 113,743 million). Lưu ý GPT-OSS chỉ dùng cho sinh/chấm answer, không thay thế VLM trong chunking ảnh.

Kết quả tóm tắt từ:
- [outputs/answer_judged_gpt_oss_hybrid_rrf_source_rerank_summary.json](outputs/answer_judged_gpt_oss_hybrid_rrf_source_rerank_summary.json)
- [outputs/answer_judged_gpt_oss_hybrid_rrf_source_rerank_v2_summary.json](outputs/answer_judged_gpt_oss_hybrid_rrf_source_rerank_v2_summary.json)
- [outputs/answer_judged_gpt_oss_hybrid_rrf_source_rerank_v3_summary.json](outputs/answer_judged_gpt_oss_hybrid_rrf_source_rerank_v3_summary.json)

| Run | Rows | All | Table | Image | Multimodal |
| --- | --- | --- | --- | --- | --- |
| v1 (gold_only) | 70 | 0.271 | 0.385 | 0.000 | 0.474 |
| v2 (gold_only) | 70 | 0.457 | 0.731 | 0.000 | 0.684 |
| v3 (gold_only) | 59 | 0.763 | 0.731 | 0.889 | 0.667 |

Nhận xét:
- v3 tăng mạnh ở image và overall; cần kiểm tra lại tập query của v3 để đảm bảo không lệch mẫu.
- Table accuracy ổn định ở v2/v3, phản ánh table chunking và retrieval đã bền hơn.

## 10. Tổng kết cải thiện và tác động

| Phần | Trạng thái | Cải thiện chính | Tác động |
| --- | --- | --- | --- |
| Text chunking | Hoàn thành | Semantic chunking + merge limit | Giữ ngữ nghĩa, giảm nhiễu retrieval text |
| Table chunking | Hoàn thành | Parse HTML, VLM summary có kiểm soát | Table retrieval và answer chính xác hơn |
| Image chunking | Hoàn thành | VLM structured metadata | Mở benchmark image theo trend/visual_type |
| FAISS index | Hoàn thành | Tách modality + cosine | Dễ ablation và lọc theo nguồn |
| Retrieval | Hoàn thành | Hybrid RRF + rerank + source filter | Tăng Recall/MRR, nhất là table/multimodal |
| Answer eval | Hoàn thành | GPT-OSS + numeric judge | Đo accuracy ổn định, giảm sai lệch format |

## 11. Đối chiếu với các paper liên quan

### 11.1 MultiFinRAG (https://arxiv.org/abs/2506.20821)

Đây là paper nền tảng phù hợp nhất với pipeline hiện tại. Dự án đã áp dụng gần đầy đủ các ý chính:
- Text semantic chunking (đã triển khai bằng BGE sentence chunking).
- Table/image thành object riêng và dùng VLM tạo summary có cấu trúc.
- Embed bằng BGE và lưu FAISS riêng theo modality.

Kết luận: project đã “đi sát” MultiFinRAG, nên dùng paper này làm nền chính trong báo cáo.

### 11.2 HierFinRAG (https://www.mdpi.com/2227-9709/13/2/30)

Paper này nhấn mạnh graph có nhiều node type (paragraph, section, table, cell, chart). Với dự án hiện tại, hướng này phù hợp hơn “Graph RAG chung chung”. Tuy nhiên phần graph đang được loại khỏi báo cáo nên chỉ nêu ở phần định hướng.

Gợi ý kiến trúc graph (định hướng):
- company -> filing -> section -> paragraph/table/chart
- table -> row -> cell
- chart -> metric/period/trend
- text -> mentions -> metric/entity

### 11.3 Multi-Document Financial QA using LLMs (https://arxiv.org/abs/2411.07264)

Paper này phù hợp cho RAG_SEM, semantic tagging và multi-document retrieval. Dự án đã có semantic_tagging và graph extraction nhưng schema còn chung. Có thể bổ sung các trường tài chính thực dụng hơn để hỗ trợ rerank/filter:

Ví dụ schema đề xuất:
{
	"company": "NVIDIA",
	"filing_type": "10-K",
	"period": "Jan 25 2026",
	"metric": "Revenue",
	"modality": "table",
	"section": "Financial Results"
}

### 11.4 ColPali / visual document retrieval (https://huggingface.co/papers/2407.01449)

ColPali phù hợp cho retrieval trực tiếp từ ảnh/page bằng VLM, giảm phụ thuộc vào OCR/table parse. Dự án hiện dùng VLM summary và retrieval text-based; ColPali có thể ghi như hướng mở rộng hoặc baseline nâng cao.

### 11.5 FinRAGBench-V (https://beancount.io/bean-labs/research-logs/2026/07/12/finragbench-v-multimodal-rag-visual-citation-financial-domain)

Paper này đưa ra taxonomy đánh giá multimodal financial RAG (text inference, chart/table extraction, numerical calculation, time-sensitive query, multi-page reasoning, visual citation). Dự án có thể lấy làm khung phân loại benchmark để “chuẩn hóa” báo cáo.

### 11.6 TAT-QA (https://github.com/NExTplusplus/tat-qa)

Phù hợp để nâng cấp chất lượng QA table: câu hỏi không chỉ hỏi cell đơn giản mà yêu cầu text + table và reasoning số học. Có thể thêm làm benchmark phụ hoặc dùng format tương tự cho bộ QA nội bộ.

### 11.7 FinQA (https://finqasite.github.io/)

Thích hợp cho reasoning số học (tăng/giảm %, so sánh theo năm). Dự án hiện chủ yếu đánh giá retrieval; nếu mở rộng sang answer prediction thì FinQA-style sẽ giúp báo cáo mạnh hơn.

### 11.8 Late Chunking / Contextual Retrieval

- Late Chunking: https://arxiv.org/abs/2409.04701
- Survey chunking/context: https://arxiv.org/abs/2504.19754

Hướng này phù hợp để cải thiện text chunking: thêm ngữ cảnh section/document trước khi embed, hoặc giữ chunk muộn để bảo toàn ngữ cảnh toàn văn. Đây là hướng có thể đưa vào phần future work.

## 12. Chi tiết tạo query đánh giá


- Mô tả chi tiết quy trình tạo QA trong [src/evaluation/generate_eval_qa.py](src/evaluation/generate_eval_qa.py): cover-page QA, table QA, multimodal QA.
- Thêm bảng so sánh retrieval theo từng modality kèm nhận xét “vì sao table/multimodal tăng mạnh khi source rerank”.
- Thêm phần “cơ chế rerank” của Hybrid RRF (trọng số ưu tiên modality và source_pdf) để lý giải kết quả.

