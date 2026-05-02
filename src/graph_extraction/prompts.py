from __future__ import annotations

NER_PROMPT_TEMPLATE = """You extract named entities from financial-report text.

Return strict JSON only:
{{
  "named_entities": ["..."]
}}

Rules:
- Include only entities explicitly present in the text.
- Keep concise canonical names.
- Prioritize financial entities: company names, ticker symbols, report forms, periods, products, business segments, geographies, regulators.
- Keep important numeric entities when they are part of financial facts (percentages, currency values, share counts, filing IDs).
- No extra keys.
- If none, return an empty list.

Text:
{text}
"""


OPENIE_PROMPT_TEMPLATE = """You extract relationship triples from financial-report text.

Return strict JSON only:
{{
  "triples": [
    ["subject", "relation", "object"]
  ]
}}

Known entities:
{entities_json}

Rules:
- Use only facts explicitly present in the text.
- Prefer triples grounded in the known entities list.
- Keep relation short and normalized (verb phrase).
- No duplicate triples.
- No extra keys.
- Do not infer who/where unless the sentence states it explicitly.
- Subject and object should be exact spans from the text when possible.
- Focus on financially meaningful relations and events.
- Capture change-direction language explicitly: increased, decreased, reduced, declined, grew, rose, fell, improved, worsened.
- Preserve period context when possible (e.g., fiscal year, quarter, as of date) in the object.
- Keep units in object values (%, $, million, billion, shares) when available.
- Prefer relations from this style list when applicable:
  - increased, decreased, reduced, declined, grew, rose, fell
  - reported, recorded, generated, incurred
  - guidance_for, expects, projects
  - has_ticker, listed_on, filed_form, filed_with
  - acquired, merged_with, partnered_with, invested_in
- If none, return an empty triples list.

Text:
{text}
"""
