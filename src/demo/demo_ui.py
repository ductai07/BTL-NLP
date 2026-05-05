from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

DEMO_DIR = Path(__file__).resolve().parent
if str(DEMO_DIR) not in sys.path:
    sys.path.append(str(DEMO_DIR))

from demo_rag import (
    answer_with_llm,
    build_prompt,
    retrieve_bm25,
    retrieve_hybrid_rrf,
    resolve_benchmark_query,
)
from demo_rag import load_jsonl as load_jsonl_rows
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BENCHMARK_DIR = Path("data/benchmark_report")
DEFAULT_CHUNKS = Path("data/chunks/all_chunks.jsonl")
DEFAULT_INDEX = Path("data/index_bge/all.faiss")
DEFAULT_IDS = Path("data/index_bge/all_chunk_ids.json")


@st.cache_data
def load_queries(benchmark_dir: Path) -> list[dict[str, Any]]:
    """Tải danh sách truy vấn từ file benchmark (có cache)."""
    return load_jsonl_rows(benchmark_dir / "queries.jsonl")


@st.cache_data
def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Tải toàn bộ các đoạn văn bản (chunks) từ file JSONL (có cache)."""
    return load_jsonl_rows(path)


def main() -> None:
    st.set_page_config(page_title="Demo MultiFinRAG", layout="wide")
    st.title("Demo MultiFinRAG (không dùng graph)")

    with st.sidebar:
        st.header("Cấu hình")

        benchmark_dir = Path(st.text_input("Thư mục Benchmark", str(DEFAULT_BENCHMARK_DIR)))
        chunks_path = Path(st.text_input("Chunks", str(DEFAULT_CHUNKS)))
        index_path = Path(st.text_input("Chỉ mục FAISS", str(DEFAULT_INDEX)))
        ids_path = Path(st.text_input("Chunk IDs", str(DEFAULT_IDS)))

        st.divider()
        st.subheader("Tham số truy xuất")
        top_k = st.slider("Top-k (số kết quả trả về)", min_value=3, max_value=10, value=5)
        candidate_k = st.slider("Candidate-k (số ứng viên xét)", min_value=20, max_value=100, value=50, step=5)
        rrf_k = st.slider("RRF k (hằng số hợp nhất)", min_value=10, max_value=100, value=60, step=5)
        rerank = st.checkbox("Reranker (theo nguồn/phương thức)", value=True)
        filter_source = st.checkbox("Lọc theo source_pdf", value=False)

        st.divider()
        st.subheader("Tùy chọn hiển thị & LLM")
        compare_mode = st.checkbox("So sánh BM25 vs Hybrid", value=True)
        enable_llm = st.checkbox("Bật LLM trả lời", value=True)
        llm_for_both = st.checkbox("Dùng LLM cho cả hai phương pháp", value=False)
        llm_model = st.text_input("Mô hình LLM", os.environ.get("OPENAI_MODEL", "openai/gpt-oss-20b"))
        base_url = st.text_input("Base URL", os.environ.get("OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1"))
        api_key = st.text_input("API key (NVIDIA_API_KEY)", type="password")
        context_chars = st.slider("Số ký tự ngữ cảnh tối đa", min_value=400, max_value=2000, value=1200, step=100)

    st.subheader("Chọn câu hỏi")
    mode = st.radio("Nguồn câu hỏi", ["Truy vấn Benchmark", "Câu hỏi tùy chỉnh"], horizontal=True)

    query_meta: dict[str, Any] = {}
    evidence_ids: set[str] = set()
    question = ""

    if mode == "Truy vấn Benchmark":
        queries = load_queries(benchmark_dir)
        query_ids = [str(row.get("query_id") or row.get("id")) for row in queries]
        selected_qid = st.selectbox("Query ID", query_ids)
        query = resolve_benchmark_query(benchmark_dir, selected_qid)
        question = query.get("question") or ""
        query_meta["type"] = query.get("type")
        query_meta["source_pdf"] = query.get("source_pdf")
        evidence_ids = set(query.get("evidence_chunk_ids") or [])
        with st.expander("Chi tiết truy vấn", expanded=True):
            st.write(query)
    else:
        question = st.text_area("Nhập câu hỏi", "", height=80)
        col1, col2 = st.columns(2)
        with col1:
            query_meta["type"] = st.selectbox("Loại truy vấn", ["text", "table", "image", "multimodal"], index=0)
        with col2:
            query_meta["source_pdf"] = st.text_input("source_pdf (tùy chọn)")

    if st.button("🚀 Chạy demo", type="primary"):
        if not question.strip():
            st.error("❌ Câu hỏi đang trống, vui lòng nhập câu hỏi trước khi chạy.")
            return

        chunks = load_chunks(chunks_path)

        with st.spinner("Đang truy xuất ngữ cảnh..."):
            hybrid_results = retrieve_hybrid_rrf(
                question=question,
                query_meta=query_meta,
                chunks=chunks,
                index_path=index_path,
                ids_path=ids_path,
                top_k=top_k,
                candidate_k=candidate_k,
                rrf_k=rrf_k,
                model_name=os.environ.get("BGE_MODEL") or "BAAI/bge-base-en-v1.5",
                batch_size=32,
                device=None,
                filter_source=filter_source,
                rerank=rerank,
            )
            bm25_results = retrieve_bm25(
                question=question,
                query_meta=query_meta,
                chunks=chunks,
                top_k=top_k,
                filter_source=filter_source,
            )

        def render_results(title: str, results: list[dict[str, Any]], score_label: str) -> None:
            """Hiển thị danh sách kết quả truy xuất dưới dạng các thẻ có thể mở rộng."""
            st.markdown(f"**{title}**")
            for item in results:
                chunk = item["chunk"]
                preview = (chunk.get("embed_text") or chunk.get("text") or chunk.get("summary") or "")
                if len(preview) > 400:
                    preview = preview[:400] + "..."
                is_source_match = bool(query_meta.get("source_pdf")) and chunk.get("source_pdf") == query_meta.get("source_pdf")
                is_modality_match = bool(query_meta.get("type")) and chunk.get("modality") == query_meta.get("type")
                is_evidence_match = chunk.get("id") in evidence_ids
                badge_parts = []
                if is_source_match:
                    badge_parts.append("✅ đúng nguồn")
                if is_modality_match:
                    badge_parts.append("✅ đúng modality")
                if is_evidence_match:
                    badge_parts.append("⭐ evidence")
                badge = " | ".join(badge_parts) if badge_parts else ""
                label = (
                    f"#{item['rank']} | {chunk.get('modality')} "
                    f"| trang {chunk.get('page') or chunk.get('page_start')}"
                )
                score_value = item.get("score")
                score_display = "-" if score_value is None else round(float(score_value), 4)
                with st.expander(label, expanded=item["rank"] == 1):
                    if badge:
                        st.caption(badge)
                    st.json(
                        {
                            "thứ_hạng": item["rank"],
                            score_label: score_display,
                            "chunk_id": chunk.get("id"),
                            "phương_thức": chunk.get("modality"),
                            "nguồn_pdf": chunk.get("source_pdf"),
                            "trang": chunk.get("page") or chunk.get("page_start"),
                        }
                    )
                    st.text(preview)

        st.subheader("Top-k ngữ cảnh truy xuất")
        if compare_mode:
            col_a, col_b = st.columns(2)
            with col_a:
                render_results("BM25", bm25_results, "điểm_bm25")
            with col_b:
                render_results("Hybrid RRF + reranker", hybrid_results, "điểm_hybrid")
        else:
            render_results("Hybrid RRF + reranker", hybrid_results, "điểm_hybrid")

        if enable_llm:
            if not api_key:
                api_key = os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENAI_API_KEY")

            if not api_key:
                st.error("❌ Thiếu API key. Hãy nhập API key hoặc đặt biến môi trường NVIDIA_API_KEY / OPENAI_API_KEY.")
                return

            def llm_answer_for(title: str, results: list[dict[str, Any]]) -> None:
                """Gọi LLM với ngữ cảnh đã truy xuất và hiển thị câu trả lời."""
                prompt = build_prompt(question, results, context_chars)
                with st.spinner(f"LLM đang xử lý ({title})..."):
                    answer = answer_with_llm(
                        prompt,
                        model=llm_model,
                        base_url=base_url,
                        api_key=api_key,
                        temperature=0.0,
                        max_tokens=512,
                    )
                st.markdown(f"**{title}**")
                st.success(answer)

            st.subheader("Câu trả lời từ LLM")
            if compare_mode and llm_for_both:
                col_a, col_b = st.columns(2)
                with col_a:
                    llm_answer_for("BM25", bm25_results)
                with col_b:
                    llm_answer_for("Hybrid RRF + reranker", hybrid_results)
            else:
                llm_answer_for("Hybrid RRF + reranker", hybrid_results)


if __name__ == "__main__":
    main()