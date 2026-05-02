# Semantic Tagging

Semantic tags follow only the categories from "Multi-Document Financial Question Answering using LLMs".

## API key

Set an OpenAI-compatible API key and model:

```powershell
$env:OPENAI_API_KEY="your_key"
$env:OPENAI_MODEL="your_model"
```

For NVIDIA's OpenAI-compatible endpoint:

```powershell
$env:OPENAI_API_KEY="your_nvidia_key"
$env:OPENAI_BASE_URL="https://integrate.api.nvidia.com/v1"
$env:OPENAI_MODEL="openai/gpt-oss-20b"
```

## Run From Scratch

```bash
python src/semantic_tagging/tag_pipeline.py --input data/chunks.jsonl --output data/chunks_tagged_paper_schema.jsonl --model $OPENAI_MODEL --overwrite
```

## Resume

```bash
python src/semantic_tagging/tag_pipeline.py --input data/chunks.jsonl --output data/chunks_tagged_paper_schema.jsonl --model $OPENAI_MODEL --resume
```

## Dry Run

Dry-run writes the empty paper schema without calling an LLM:

```bash
python src/semantic_tagging/tag_pipeline.py --input data/chunks.jsonl --output data/chunks_tagged_paper_schema.jsonl --dry-run --overwrite
```
