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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def row_id(row: dict[str, Any], idx: int) -> str:
    for key in ("id", "query_id", "question_id", "qid"):
        value = row.get(key)
        if value is not None and clean_text(value):
            return str(value)
    return f"row_{idx:05d}"


def get_first(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def parse_jsonish(text: str) -> dict[str, Any]:
    text = clean_text(text)
    if text.startswith("```"):
        text = text.strip("`")
        text = re.sub(r"^json\s*", "", text, flags=re.IGNORECASE).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


def normalize_judgment(value: Any) -> str:
    text = clean_text(value).lower()
    if text in {"correct", "true", "yes"}:
        return "correct"
    if text in {"partial", "partially_correct", "partially correct"}:
        return "partial"
    if text in {"unanswerable", "no_prediction", "missing"}:
        return "unanswerable"
    return "incorrect"


def normalize_numeric_text(value: Any) -> str:
    text = clean_text(value).lower()
    text = text.replace("$", "")
    text = text.replace(",", "")
    text = text.replace("%", " percent")
    return re.sub(r"\s+", " ", text)


def first_number(value: Any) -> float | None:
    text = normalize_numeric_text(value)
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def deterministic_judge(row: dict[str, Any]) -> dict[str, Any] | None:
    gold = clean_text(get_first(row, ("gold_answer", "answer", "expected_answer", "reference_answer")))
    pred = clean_text(get_first(row, ("pred_answer", "prediction", "predicted_answer", "model_answer", "generated_answer")))
    answer_type = clean_text(row.get("answer_type") or "").lower()

    if not pred:
        return {
            "judgment": "unanswerable",
            "is_correct": False,
            "confidence": 1.0,
            "rationale": "Prediction is empty.",
        }

    if clean_text(pred).lower() in {"not found", "not_found", "n/a", "unknown", "cannot answer"}:
        return {
            "judgment": "unanswerable",
            "is_correct": False,
            "confidence": 1.0,
            "rationale": "Prediction says the answer was not found.",
        }

    if answer_type in {"number", "percentage"}:
        gold_num = first_number(gold)
        pred_num = first_number(pred)
        if gold_num is not None and pred_num is not None:
            tolerance = 0.001 * max(1.0, abs(gold_num))
            if abs(gold_num - pred_num) <= tolerance:
                return {
                    "judgment": "correct",
                    "is_correct": True,
                    "confidence": 1.0,
                    "rationale": "Numeric value matches after normalization.",
                }

    gold_norm = normalize_numeric_text(gold)
    pred_norm = normalize_numeric_text(pred)
    if gold_norm and (gold_norm in pred_norm or pred_norm in gold_norm):
        return {
            "judgment": "correct",
            "is_correct": True,
            "confidence": 0.95,
            "rationale": "Normalized short answer matches.",
        }

    return None


def answer_judge_prompt(row: dict[str, Any]) -> str:
    question = clean_text(get_first(row, ("question", "query")))
    gold = clean_text(get_first(row, ("gold_answer", "answer", "expected_answer", "reference_answer")))
    pred = clean_text(get_first(row, ("pred_answer", "prediction", "predicted_answer", "model_answer", "generated_answer")))
    answer_type = clean_text(row.get("answer_type") or row.get("type") or "")
    unit = clean_text(row.get("unit"))
    evidence = clean_text(row.get("evidence") or row.get("gold_evidence") or row.get("context"))

    return f"""You are judging answers for a financial QA benchmark.

Return JSON only with this exact schema:
{{
  "judgment": "correct" | "partial" | "incorrect" | "unanswerable",
  "is_correct": true | false,
  "confidence": 0.0,
  "rationale": "short explanation"
}}

Judging rules:
- Mark "correct" when the predicted answer is semantically equivalent to the gold answer.
- For numbers, ignore commas, currency symbols, and harmless words like "million" if the expected unit is already provided.
- Treat 12% and 12 percent as equivalent.
- Treat "29,904" and "29,904 million" as equivalent if the unit says million.
- Mark "partial" only when the answer contains the right entity/metric but misses a required value, period, unit, or qualifier.
- Mark "incorrect" when the value, date, company, period, or direction is wrong.
- Mark "unanswerable" when the prediction is empty, says it cannot answer, or refuses.
- Do not require exact wording for short text answers if meaning is the same.

Question: {question}
Gold answer: {gold}
Predicted answer: {pred}
Answer type: {answer_type}
Expected unit: {unit}
Gold evidence/context: {evidence[:1500]}
"""


class GptOssJudge:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        temperature: float,
        max_tokens: int,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def judge(self, row: dict[str, Any]) -> dict[str, Any]:
        deterministic = deterministic_judge(row)
        if deterministic is not None:
            return deterministic

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": answer_judge_prompt(row)}],
            temperature=self.temperature,
            top_p=1,
            max_tokens=self.max_tokens,
            stream=False,
        )
        content = completion.choices[0].message.content or ""
        parsed = parse_jsonish(content)
        judgment = normalize_judgment(parsed.get("judgment"))
        return {
            "judgment": judgment,
            "is_correct": bool(parsed.get("is_correct")) and judgment == "correct",
            "confidence": float(parsed.get("confidence") or 0.0),
            "rationale": clean_text(parsed.get("rationale")),
            "raw_judge_output": content,
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        qtype = clean_text(row.get("type") or row.get("question_type") or "unknown")
        grouped[qtype].append(row)
        grouped["all"].append(row)

    metrics: dict[str, Any] = {}
    for qtype, items in grouped.items():
        total = len(items)
        correct = sum(1 for item in items if item.get("is_correct"))
        partial = sum(1 for item in items if item.get("judgment") == "partial")
        incorrect = sum(1 for item in items if item.get("judgment") == "incorrect")
        unanswerable = sum(1 for item in items if item.get("judgment") == "unanswerable")
        metrics[qtype] = {
            "total": total,
            "correct": correct,
            "partial": partial,
            "incorrect": incorrect,
            "unanswerable": unanswerable,
            "accuracy": correct / total if total else 0.0,
            "partial_rate": partial / total if total else 0.0,
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge final RAG answers with NVIDIA openai/gpt-oss-20b.")
    parser.add_argument("--input", type=Path, required=True, help="JSONL with question, gold_answer/answer, pred_answer.")
    parser.add_argument("--output", type=Path, default=Path("outputs/answer_judged_gpt_oss.jsonl"))
    parser.add_argument("--summary-out", type=Path, default=Path("outputs/answer_judged_gpt_oss_summary.json"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dotenv", type=Path, default=Path(".env"))
    args = parser.parse_args()

    load_dotenv(dotenv_path=args.dotenv)
    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(f"Missing API key. Set {args.api_key_env} or OPENAI_API_KEY.")

    rows = load_jsonl(args.input)
    if args.limit is not None:
        rows = rows[: args.limit]

    done_ids: set[str] = set()
    judged_rows: list[dict[str, Any]] = []
    if args.resume and args.output.exists():
        judged_rows = load_jsonl(args.output)
        done_ids = {str(row.get("id") or row.get("query_id")) for row in judged_rows}
    elif args.output.exists():
        args.output.unlink()

    judge = GptOssJudge(
        model=args.model,
        base_url=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    for idx, row in enumerate(rows, start=1):
        rid = row_id(row, idx)
        if rid in done_ids:
            continue
        print(f"[{idx}/{len(rows)}] judging {rid}")
        try:
            result = judge.judge(row)
        except Exception as exc:
            result = {
                "judgment": "judge_error",
                "is_correct": False,
                "confidence": 0.0,
                "rationale": str(exc),
            }
        out = dict(row)
        out["id"] = rid
        out.update(result)
        append_jsonl(args.output, out)
        judged_rows.append(out)
        if args.sleep:
            time.sleep(args.sleep)

    summary = {
        "input": str(args.input).replace("\\", "/"),
        "output": str(args.output).replace("\\", "/"),
        "model": args.model,
        "base_url": args.base_url,
        "rows": len(judged_rows),
        "metrics": summarize(judged_rows),
    }
    write_json(args.summary_out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
