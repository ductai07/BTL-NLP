from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from common.bge_embedder import DEFAULT_BGE_MODEL, load_bge_model


DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def require_faiss():
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError("Thiếu faiss-cpu. Hãy cài đặt bằng lệnh: python -m pip install faiss-cpu") from exc
    return faiss


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Đọc file JSONL và trả về danh sách các dòng đã được phân tích."""
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_text(value: Any) -> str:
    """Chuẩn hóa khoảng trắng và chuyển về chuỗi."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def chunk_text(chunk: dict[str, Any]) -> str:
    """Lấy nội dung văn bản chính từ một đoạn dữ liệu (chunk)."""
    for key in ("embed_text", "text", "summary"):
        value = clean_text(chunk.get(key))
        if value:
            return value
    return ""


def tokenize(text: Any) -> list[str]:
    """Tách từ đơn giản dành cho BM25."""
    return re.findall(r"[a-z0-9][a-z0-9._%$-]*", str(text or "").lower())


class BM25Index:
    """Chỉ mục tìm kiếm BM25 (Best Match 25) dành cho truy xuất thưa."""

    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.chunk_ids = [chunk["id"] for chunk in chunks]
        self.docs = [tokenize(chunk_text(chunk)) for chunk in chunks]
        self.doc_lens = np.array([len(doc) for doc in self.docs], dtype="float32")
        self.avgdl = float(np.mean(self.doc_lens)) if len(self.doc_lens) else 1.0
        self.k1 = k1
        self.b = b
        self.term_freqs = [Counter(doc) for doc in self.docs]

        # Tính tần suất xuất hiện tài liệu (document frequency)
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))

        n_docs = len(self.docs)
        # Tính IDF (Inverse Document Frequency) theo công thức BM25
        self.idf = {
            term: np.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, top_k: int, source_pdf: str | None, filter_source: bool) -> list[str]:
        """Tìm kiếm các đoạn văn bản liên quan nhất theo điểm BM25."""
        query_terms = tokenize(query)
        scores = np.zeros(len(self.docs), dtype="float32")

        for term in query_terms:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for idx, tf in enumerate(self.term_freqs):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * self.doc_lens[idx] / self.avgdl)
                scores[idx] += idf * (freq * (self.k1 + 1.0) / denom)

        out: list[str] = []
        for idx in np.argsort(-scores):
            if scores[idx] <= 0:
                break
            chunk = self.chunks[int(idx)]
            # Lọc theo nguồn PDF nếu được yêu cầu
            if filter_source and source_pdf and chunk.get("source_pdf") != source_pdf:
                continue
            out.append(chunk["id"])
            if len(out) >= top_k:
                break
        return out


def rrf_score(rank: int, rrf_k: int) -> float:
    """Tính điểm RRF (Reciprocal Rank Fusion) cho một thứ hạng nhất định."""
    return 1.0 / (rrf_k + rank)


def rerank_bonus(query: dict[str, Any], chunk: dict[str, Any]) -> float:
    """Tính điểm thưởng khi xếp hạng lại dựa trên loại truy vấn và nguồn dữ liệu."""
    score = 0.0
    # Thưởng khi loại truy vấn khớp với phương thức của đoạn văn bản
    if query.get("type") == chunk.get("modality"):
        score += 0.08
    # Thưởng nhỏ hơn cho truy vấn đa phương thức
    if query.get("type") == "multimodal" and chunk.get("modality") in {"text", "table", "image"}:
        score += 0.03
    # Thưởng khi nguồn PDF khớp nhau
    if query.get("source_pdf") and query.get("source_pdf") == chunk.get("source_pdf"):
        score += 0.06
    return score


def build_prompt(question: str, contexts: list[dict[str, Any]], context_chars: int) -> str:
    """Xây dựng prompt gửi cho LLM với các ngữ cảnh đã truy xuất."""
    blocks = []
    for idx, item in enumerate(contexts, start=1):
        chunk = item["chunk"]
        text = chunk_text(chunk)
        if len(text) > context_chars:
            text = text[:context_chars] + "..."
        blocks.append(
            "\n".join(
                [
                    f"[Ngữ cảnh {idx}]",
                    f"chunk_id: {chunk.get('id')}",
                    f"modality: {chunk.get('modality')}",
                    f"source_pdf: {chunk.get('source_pdf')}",
                    f"page: {chunk.get('page') or chunk.get('page_start')}",
                    text,
                ]
            )
        )

    return (
        "Bạn đang trả lời một câu hỏi tài chính chỉ dựa trên các ngữ cảnh đã được truy xuất.\n"
        "Quy tắc:\n"
        "- Trả lời ngắn gọn và chính xác nhất có thể.\n"
        "- Ưu tiên số liệu, ngày tháng, phần trăm, tên gọi hoặc cụm từ ngắn.\n"
        "- Nếu câu trả lời là số, hãy giữ nguyên giá trị như trong ngữ cảnh.\n"
        "- Chỉ bao gồm đơn vị khi câu hỏi yêu cầu hoặc ngữ cảnh bắt buộc phải có.\n"
        "- Không giải thích lý do.\n"
        "- Nếu ngữ cảnh không chứa câu trả lời, đặt answer là \"Không tìm thấy\".\n"
        "- Chỉ trả về JSON theo đúng cấu trúc: {\"answer\": \"câu trả lời ngắn gọn\"}\n\n"
        f"Câu hỏi: {clean_text(question)}\n\n"
        "Các ngữ cảnh đã truy xuất:\n"
        + "\n".join(blocks)
    )


def parse_answer(content: str) -> str:
    """Phân tích và trích xuất câu trả lời từ phản hồi JSON của LLM."""
    content = clean_text(content)
    if not content:
        return ""
    # Xử lý trường hợp phản hồi bị bọc trong markdown code block
    if content.startswith("```"):
        content = content.strip("`")
        content = re.sub(r"^json\s*", "", content, flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return content
        return clean_text(parsed.get("answer"))
    return content


def answer_with_llm(prompt: str, model: str, base_url: str, api_key: str, temperature: float, max_tokens: int) -> str:
    """Gửi prompt tới LLM và trả về câu trả lời đã được phân tích."""
    client = OpenAI(base_url=base_url, api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Chỉ trả về đối tượng JSON cuối cùng. Không bao gồm lý do giải thích hay markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=1,
        max_tokens=max_tokens,
        stream=False,
    )
    content = completion.choices[0].message.content or ""
    return parse_answer(content)


def resolve_benchmark_query(benchmark_dir: Path, query_id: str) -> dict[str, Any]:
    """Tìm và trả về một truy vấn cụ thể từ tập dữ liệu benchmark."""
    queries = load_jsonl(benchmark_dir / "queries.jsonl")
    for row in queries:
        if str(row.get("query_id") or row.get("id")) == query_id:
            return row
    raise ValueError(f"Không tìm thấy query_id {query_id} trong {benchmark_dir}/queries.jsonl")


def retrieve_hybrid_rrf(
    *,
    question: str,
    query_meta: dict[str, Any],
    chunks: list[dict[str, Any]],
    index_path: Path,
    ids_path: Path,
    top_k: int,
    candidate_k: int,
    rrf_k: int,
    model_name: str,
    batch_size: int,
    device: str | None,
    filter_source: bool,
    rerank: bool,
) -> list[dict[str, Any]]:
    """
    Truy xuất kết hợp (Hybrid Retrieval) sử dụng RRF để hợp nhất:
      - Tìm kiếm dày đặc (Dense Search) qua FAISS + BGE embeddings
      - Tìm kiếm thưa (Sparse Search) qua BM25
    """
    faiss = require_faiss()

    # Tải chỉ mục FAISS và danh sách ID tương ứng
    index = faiss.read_index(str(index_path))
    dense_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    chunks_by_id = {row["id"]: row for row in chunks}

    # Mã hóa câu truy vấn thành vector embedding
    embedder = load_bge_model(model_name=model_name, batch_size=batch_size, device=device)
    query_vec = embedder.encode_queries([question])

    # Tìm kiếm các ứng viên gần nhất trong không gian vector
    dense_scores, dense_indices = index.search(query_vec, min(candidate_k, len(dense_ids)))

    # Tìm kiếm BM25 song song
    bm25 = BM25Index(chunks)
    source_pdf = query_meta.get("source_pdf")
    bm25_ids = bm25.search(question, candidate_k, source_pdf=source_pdf, filter_source=filter_source)
    bm25_rank = {cid: rank for rank, cid in enumerate(bm25_ids, start=1)}

    # Xây dựng bảng xếp hạng từ kết quả tìm kiếm dày đặc
    dense_rank: dict[str, int] = {}
    for rank, raw_idx in enumerate(dense_indices[0], start=1):
        if raw_idx < 0:
            continue
        cid = dense_ids[int(raw_idx)]
        chunk = chunks_by_id.get(cid)
        if not chunk:
            continue
        if filter_source and source_pdf and chunk.get("source_pdf") != source_pdf:
            continue
        dense_rank[cid] = rank

    # Hợp nhất điểm RRF từ cả hai phương pháp tìm kiếm
    fused: dict[str, float] = {}
    for cid, rank in bm25_rank.items():
        fused[cid] = fused.get(cid, 0.0) + rrf_score(rank, rrf_k)
    for cid, rank in dense_rank.items():
        fused[cid] = fused.get(cid, 0.0) + rrf_score(rank, rrf_k)

    # Tính điểm cuối cùng (có thể cộng thêm điểm thưởng khi xếp hạng lại)
    candidates = []
    for cid, score in fused.items():
        chunk = chunks_by_id.get(cid, {})
        final_score = score + (rerank_bonus(query_meta, chunk) if rerank else 0.0)
        candidates.append((final_score, score, cid, chunk))
    candidates.sort(key=lambda row: row[0], reverse=True)

    # Trả về top_k kết quả tốt nhất
    results = []
    for rank, (final_score, fused_score, cid, chunk) in enumerate(candidates[:top_k], start=1):
        results.append(
            {
                "rank": rank,
                "score": float(final_score),
                "rrf_score": float(fused_score),
                "chunk": chunk,
            }
        )
    return results


def retrieve_bm25(
    *,
    question: str,
    query_meta: dict[str, Any],
    chunks: list[dict[str, Any]],
    top_k: int,
    filter_source: bool,
) -> list[dict[str, Any]]:
    """Truy xuất chỉ dùng BM25 (không kết hợp với tìm kiếm dày đặc)."""
    bm25 = BM25Index(chunks)
    source_pdf = query_meta.get("source_pdf")
    bm25_ids = bm25.search(question, top_k, source_pdf=source_pdf, filter_source=filter_source)
    chunks_by_id = {row["id"]: row for row in chunks}
    results = []
    for rank, cid in enumerate(bm25_ids, start=1):
        chunk = chunks_by_id.get(cid, {})
        results.append(
            {
                "rank": rank,
                "score": None,
                "rrf_score": None,
                "chunk": chunk,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo truy xuất Hybrid RRF kết hợp trả lời bằng LLM.")
    parser.add_argument("--question", help="Câu hỏi của người dùng để demo")
    parser.add_argument("--benchmark-id", help="Dùng một query_id từ file data/benchmark_report/queries.jsonl")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark_report"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks/all_chunks.jsonl"))
    parser.add_argument("--index", type=Path, default=Path("data/index_bge/all.faiss"))
    parser.add_argument("--ids", type=Path, default=Path("data/index_bge/all_chunk_ids.json"))
    parser.add_argument("--top-k", type=int, default=5, help="Số lượng kết quả trả về")
    parser.add_argument("--candidate-k", type=int, default=50, help="Số lượng ứng viên xét ban đầu")
    parser.add_argument("--rrf-k", type=int, default=60, help="Hằng số k trong công thức RRF")
    parser.add_argument("--model", default=DEFAULT_BGE_MODEL, help="Tên mô hình BGE để tạo embedding")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", help="Thiết bị tính toán (cpu, cuda, ...)")
    parser.add_argument("--type", choices=["text", "table", "image", "multimodal"], help="Loại truy vấn (tùy chọn)")
    parser.add_argument("--source-pdf", help="Lọc theo tên file PDF nguồn (tùy chọn)")
    parser.add_argument("--filter-source", action="store_true", help="Bật lọc theo nguồn PDF")
    parser.add_argument("--rerank", action="store_true", help="Bật xếp hạng lại với điểm thưởng")
    parser.add_argument("--no-llm", action="store_true", help="Chỉ truy xuất, không gọi LLM")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Tên mô hình LLM dùng để trả lời")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY", help="Tên biến môi trường chứa API key")
    parser.add_argument("--temperature", type=float, default=0.0, help="Nhiệt độ sinh văn bản của LLM")
    parser.add_argument("--max-tokens", type=int, default=512, help="Số token tối đa trong phản hồi LLM")
    parser.add_argument("--context-chars", type=int, default=1200, help="Số ký tự tối đa mỗi đoạn ngữ cảnh")
    args = parser.parse_args()

    if not args.question and not args.benchmark_id:
        raise SystemExit("Vui lòng cung cấp --question hoặc --benchmark-id")

    query_meta: dict[str, Any] = {}
    question = args.question

    # Nếu dùng benchmark, tải câu hỏi và metadata từ file
    if args.benchmark_id:
        q = resolve_benchmark_query(args.benchmark_dir, args.benchmark_id)
        question = q.get("question")
        query_meta["type"] = q.get("type")
        query_meta["source_pdf"] = q.get("source_pdf")

    if not question:
        raise SystemExit("Câu hỏi bị trống")

    # Ghi đè metadata nếu người dùng truyền thủ công
    if args.type:
        query_meta["type"] = args.type
    if args.source_pdf:
        query_meta["source_pdf"] = args.source_pdf

    # Tải toàn bộ chunks và thực hiện truy xuất
    chunks = load_jsonl(args.chunks)
    results = retrieve_hybrid_rrf(
        question=question,
        query_meta=query_meta,
        chunks=chunks,
        index_path=args.index,
        ids_path=args.ids,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        filter_source=args.filter_source,
        rerank=args.rerank,
    )

    print("\n=== CÂU HỎI ===")
    print(question)
    if query_meta:
        print(f"metadata: {query_meta}")

    print("\n=== TOP-K NGỮ CẢNH ===")
    for item in results:
        chunk = item["chunk"]
        preview = chunk_text(chunk)
        preview = preview[:400] + ("..." if len(preview) > 400 else "")
        print(
            json.dumps(
                {
                    "thứ_hạng": item["rank"],
                    "điểm_số": round(item["score"], 4),
                    "chunk_id": chunk.get("id"),
                    "phương_thức": chunk.get("modality"),
                    "nguồn_pdf": chunk.get("source_pdf"),
                    "trang": chunk.get("page") or chunk.get("page_start"),
                    "xem_trước": preview,
                },
                ensure_ascii=False,
            )
        )

    if args.no_llm:
        return

    # Lấy API key từ biến môi trường
    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"Thiếu API key. Hãy đặt biến môi trường {args.api_key_env} hoặc OPENAI_API_KEY.")

    prompt = build_prompt(question, results, args.context_chars)
    answer = answer_with_llm(
        prompt,
        model=args.llm_model,
        base_url=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print("\n=== CÂU TRẢ LỜI TỪ LLM ===")
    print(answer)


if __name__ == "__main__":
    main()