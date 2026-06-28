# utils.py — Shared helpers used across the pipeline.
# Centralises JSON parsing so jd_parser and rerank don't duplicate it.

import json
import re


def try_json(text: str):
    """Return a parsed dict/list if text is valid JSON, else None."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def safe_parse_json(raw: str, fallback: dict) -> dict:
    """
    Parse LLM output to a dict, with multiple fallback levels:
      1. Direct JSON parse of the cleaned string.
      2. Extract the last {...} block via regex (handles preamble text).
      3. Extract the first {...} block.
      4. Parse key: value lines (for models that skip JSON).
      5. Return `fallback` so the pipeline never crashes on bad output.
    Strips markdown code fences before any attempt.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Attempt 1: direct parse
    data = try_json(cleaned)
    if isinstance(data, dict):
        return data

    # Attempt 2: last JSON object (model output often has JSON at the end)
    matches = list(re.finditer(r"\{[^{}]*\}", cleaned, re.DOTALL))
    if not matches:
        matches = list(re.finditer(r"\{.*?\}", cleaned, re.DOTALL))
    for m in reversed(matches):
        data = try_json(m.group())
        if isinstance(data, dict):
            return data

    # Attempt 3: greedy match for nested JSON
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        data = try_json(match.group())
        if isinstance(data, dict):
            return data

    # Attempt 4: parse "key: value" lines for models that skip JSON format
    kv = {}
    for line in cleaned.splitlines():
        line = line.strip().strip(",")
        m = re.match(r'^["\']?(\w+)["\']?\s*[=:]\s*["\']?([^"\']+)["\']?$', line)
        if m:
            kv[m.group(1).lower()] = m.group(2).strip()
    if kv and any(k in kv for k in ("llm_score", "score", "reason", "confidence")):
        if "score" in kv and "llm_score" not in kv:
            kv["llm_score"] = kv["score"]
        return kv

    return fallback if isinstance(fallback, dict) else {}


def as_list(value) -> list:
    """Coerce a value to a list — returns [] for None or unexpected types."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value]
    return []
