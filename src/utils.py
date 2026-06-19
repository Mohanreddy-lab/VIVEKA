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
    Parse LLM output to a dict, with three fallback levels:
      1. Direct JSON parse of the cleaned string.
      2. Extract the first {...} block via regex.
      3. Return `fallback` so the pipeline never crashes on bad output.
    Strips markdown code fences before any attempt.
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    data = try_json(cleaned)

    if data is None:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = try_json(match.group())

    return data if isinstance(data, dict) else fallback


def as_list(value) -> list:
    """Coerce a value to a list — returns [] for None or unexpected types."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value]
    return []
