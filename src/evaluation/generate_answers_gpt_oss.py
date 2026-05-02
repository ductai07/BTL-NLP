from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_MODEL = "openai/gpt-oss-20b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def context_text(chunk: dict[str, Any], limit: int) -> str:
    for key in ("embed_text", "text", "summary"):
        value = clean_text(chunk.get(key))
        if value:
            break
    else:
        value = ""
    if len(value) > limit:
        value = value[:limit] + "..."
    return value


def load_retrieval_by_query(path: Path, top_k: int) -> dict[str, list[dict[str, Any]]]:
    rows = load_jsonl(path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["query_id"]].append(row)
    out: dict[str, list[dict[str, Any]]] = {}
    for qid, items in grouped.items():
        items.sort(key=lambda row: int(row.get("rank") or 9999))
        out[qid] = items[:top_k]
    return out


def build_prompt(query: dict[str, Any], contexts: list[dict[str, Any]], chunks_by_id: dict[str, dict[str, Any]], context_chars: int) -> str:
    blocks = []
    for item in contexts:
        cid = item["chunk_id"]
        chunk = chunks_by_id.get(cid, {})
        blocks.append(
            "\n".join(
                [
                    f"[Context {len(blocks) + 1}]",
                    f"chunk_id: {cid}",
                    f"modality: {chunk.get('modality')}",
                    f"source_pdf: {chunk.get('source_pdf')}",
                    f"page: {chunk.get('page') or chunk.get('page_start')}",
                    context_text(chunk, context_chars),
                ]
            )
        )

    answer_type = clean_text(query.get("answer_type") or query.get("type"))

    return f"""You are answering a financial QA benchmark question using only the retrieved contexts.

Rules:
- Answer with the shortest exact answer possible.
- Prefer numbers, dates, percentages, names, or short phrases.
- If the answer is a number, preserve the value exactly as shown in the context.
- Include units only when the question asks for them or the context makes the unit necessary.
- Do not explain your reasoning.
- If the contexts do not contain the answer, set answer to "Not found".
- Return JSON only with this exact schema: {{"answer": "short final answer"}}

Question: {clean_text(query.get("question"))}
Answer type: {answer_type}

Retrieved contexts:
{chr(10).join(blocks)}
"""


class AnswerGenerator:
    def __init__(self, *, model: str, base_url: str, api_key: str, temperature: float, max_tokens: int) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def answer(self, prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Return only the final JSON object. Do not include reasoning or markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=1,
            max_tokens=self.max_tokens,
            stream=False,
        )
        message = completion.choices[0].message
        content = clean_text(message.content or "")
        answer = parse_answer(content)
        if answer:
            return answer

        retry = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You must output only JSON like {\"answer\":\"...\"}. No reasoning.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            top_p=1,
            max_tokens=max(self.max_tokens, 1024),
            stream=False,
        )
        retry_content = clean_text(retry.choices[0].message.content or "")
        return parse_answer(retry_content) or retry_content


def parse_answer(content: str) -> str:
    content = clean_text(content)
    if not content:
        return ""
    if content.startswith("```"):
        content = content.strip("`")
        content = re.sub(r"^json\s*", "", content, flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
        else:
            return clean_text(parsed.get("answer"))
    return content


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final answers from retrieval contexts with gpt-oss.")
    parser.add_argument("--benchmark-dir", type=Path, default=Path("data/benchmark_report"))
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks/all_chunks.jsonl"))
    parser.add_argument("--retrieval-results", type=Path, default=Path("outputs/retrieval_benchmark_report_hybrid_rrf_source_rerank/retrieval_results.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/answer_generation_gpt_oss.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-chars", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_dotenv(dotenv_path=args.dotenv)
    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing API key. Set {args.api_key_env} or OPENAI_API_KEY.")

    queries = load_jsonl(args.benchmark_dir / "queries.jsonl")
    if args.limit is not None:
        queries = queries[: args.limit]
    chunks = load_jsonl(args.chunks)
    chunks_by_id = {row["id"]: row for row in chunks}
    retrieval_by_query = load_retrieval_by_query(args.retrieval_results, args.top_k)

    done: set[str] = set()
    if args.resume and args.output.exists():
        done = {row["query_id"] for row in load_jsonl(args.output)}
    elif args.output.exists():
        args.output.unlink()

    generator = AnswerGenerator(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    for idx, query in enumerate(queries, start=1):
        qid = query["query_id"]
        if qid in done:
            continue
        contexts = retrieval_by_query.get(qid, [])
        print(f"[{idx}/{len(queries)}] answering {qid} contexts={len(contexts)}")
        prompt = build_prompt(query, contexts, chunks_by_id, args.context_chars)
        try:
            pred = generator.answer(prompt)
        except Exception as exc:
            pred = ""
            error = str(exc)
        else:
            error = None
        append_jsonl(
            args.output,
            {
                "query_id": qid,
                "id": qid,
                "type": query.get("type"),
                "question": query.get("question"),
                "gold_answer": query.get("answer"),
                "answer_type": query.get("answer_type"),
                "source_pdf": query.get("source_pdf"),
                "evidence_chunk_ids": query.get("evidence_chunk_ids"),
                "retrieved_chunk_ids": [row["chunk_id"] for row in contexts],
                "pred_answer": pred,
                "generation_error": error,
            },
        )
        if args.sleep:
            time.sleep(args.sleep)


if __name__ == "__main__":
    main()
