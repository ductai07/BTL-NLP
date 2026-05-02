from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from PIL import Image


NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
DEFAULT_MODEL = "google/gemma-3-27b-it"


load_dotenv()


def image_data_uri(image_path: Path, max_side: int = 1024, jpeg_quality: int = 85) -> str:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def parse_sse_response(response: requests.Response) -> str:
    parts: list[str] = []
    for raw_line in response.iter_lines():
        if not raw_line:
            continue

        line = raw_line.decode("utf-8")
        if not line.startswith("data:"):
            continue

        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break

        payload = json.loads(data)
        delta = payload.get("choices", [{}])[0].get("delta", {})
        content = delta.get("content")
        if content:
            parts.append(content)

    return "".join(parts)


def call_gemma_vision(
    prompt: str,
    image_path: Path,
    *,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    max_tokens: int = 800,
    temperature: float = 0.2,
    top_p: float = 0.7,
    stream: bool = False,
    timeout: int = 120,
    max_image_side: int = 1024,
) -> str:
    api_key = api_key or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing NVIDIA_API_KEY. Set it first, for example:\n"
            "  $env:NVIDIA_API_KEY='your_key_here'"
        )

    content = f"{prompt}\n\n<img src=\"{image_data_uri(image_path, max_side=max_image_side)}\" />"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "text/event-stream" if stream else "application/json",
        "Content-Type": "application/json",
    }
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": stream,
    }

    response = requests.post(
        NVIDIA_CHAT_URL,
        headers=headers,
        json=payload,
        stream=stream,
        timeout=timeout,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        detail = response.text[:2000]
        raise RuntimeError(f"NVIDIA API request failed: {response.status_code} {detail}") from exc

    if stream:
        return parse_sse_response(response)

    result = response.json()
    return result["choices"][0]["message"]["content"]


TABLE_PROMPT = """You are extracting a financial table from an SEC filing image.
Return JSON only with these keys:
- summary: one short sentence describing what the table reports
- table_json: a structured JSON object preserving row labels, column labels, values, and units when visible
- table_metadata: object with title, unit, periods, metrics, row_labels, column_labels
- key_facts: list of objects with row, column, value, unit, period, metric
- evidence: short note about visible title/header/page context

Rules:
- Keep numeric values exactly as shown.
- Do not invent missing values.
- If image text is blurry but a parsed table is provided in the prompt, use the parsed table for summary.
- Prefer a precise summary naming visible row labels, column labels, periods, units, and financial metrics.
- key_facts must contain only explicit table cells, not calculations.
- If the image is not a table, set table_json to null and explain in summary."""


IMAGE_PROMPT = """You are summarizing a chart, figure, or image from an SEC filing.
Return JSON only with these keys:
- visual_type: one of line_chart, bar_chart, pie_chart, table_like_chart, diagram, logo, photo, decorative, unreadable, other
- title: visible title, or null
- summary: 3 to 6 concise sentences describing the visual
- x_axis: visible x-axis label or null
- y_axis: visible y-axis label or null
- legend: list of visible legend labels
- metrics: list of financial/business metrics visible or clearly implied by labels
- periods: list of visible years, quarters, dates, or ranges
- key_values: short list of visible labels, ranges, values, or categories
- trend: one short phrase such as upward, downward, mixed, stable, not_applicable, or unreadable
- evidence: short note about visible title/header/page context

Rules:
- Keep numeric values exactly as shown.
- Do not guess the business domain or chart meaning if labels are unreadable.
- If text, axes, or legend are not readable, say they are not readable.
- Describe only what is visually present: chart type, visible labels, axes, legend, trend direction, colors, and readable numbers.
- If labels do not identify a metric, keep metrics empty.
- If the image is a logo, icon, signature, decorative banner, or unreadable crop, say so clearly and keep key_values empty."""


def main() -> None:
    parser = argparse.ArgumentParser(description="Call NVIDIA Gemma 3 27B vision on a local image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--kind", choices=["table", "image"], default="table")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--max-image-side", type=int, default=1024)
    args = parser.parse_args()

    prompt = TABLE_PROMPT if args.kind == "table" else IMAGE_PROMPT
    output = call_gemma_vision(
        prompt,
        args.image,
        model=args.model,
        stream=args.stream,
        max_tokens=args.max_tokens,
        max_image_side=args.max_image_side,
    )
    print(output)


if __name__ == "__main__":
    main()
