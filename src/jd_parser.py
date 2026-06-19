"""
jd_parser.py — Stage 1: JD Intelligence

Takes a raw job description string and uses the local LLM (via llm.py)
to return structured data:
  - required_skills : skills explicitly stated in the JD
  - implied_skills  : skills not stated but clearly needed for the role
  - seniority       : "junior" | "mid" | "senior"
  - latent_needs    : what the role truly tests beyond the listed skills

Runs fully offline — no API key, no internet required.
"""

import json
import re
import sys
import os

# Allow running as `python src/jd_parser.py` from the project root
sys.path.insert(0, os.path.dirname(__file__))

from llm import get_llm
from langchain.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Prompts
# Local models need shorter, more explicit instructions than GPT-4o.
# We end the user turn with "JSON:" to steer the model straight into output.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior technical recruiter. Analyze job descriptions and extract structured data.

You MUST return ONLY a JSON object — nothing else. No markdown, no explanation, no code fences.
The JSON must have exactly these four keys:
  "required_skills"  - array of skills explicitly mentioned in the job description
  "implied_skills"   - array of skills not stated but obviously needed (e.g. Git for any dev role)
  "seniority"        - exactly one string: "junior", "mid", or "senior"
  "latent_needs"     - array of short phrases describing what this role truly tests
                       (e.g. "works under ambiguity", "owns outcomes end-to-end")

Start your response with { and end with }. Nothing before or after the JSON."""

USER_TEMPLATE = """Job Description:
{jd_text}

JSON:"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_jd(jd_text: str) -> dict:
    """
    Parse a job description and return a structured dict.

    Args:
        jd_text: Raw job description as plain text.

    Returns:
        dict with keys: required_skills, implied_skills, seniority, latent_needs.
        All keys are always present — empty lists / "unknown" on failure.
    """
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])

    chain = prompt | llm
    response = chain.invoke({"jd_text": jd_text.strip()})

    return _safe_parse(response.content)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_parse(raw: str) -> dict:
    """
    Parse LLM output to a dict, with multiple fallback strategies.

    Local models sometimes wrap JSON in prose or markdown fences.
    We try progressively looser extraction until something parses.
    """
    # 1. Strip markdown code fences if the model added them
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # 2. Try parsing the whole response directly
    data = _try_json(cleaned)

    # 3. If that fails, find the first {...} block in the text
    if data is None:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            data = _try_json(match.group())

    # 4. Still nothing — return safe defaults so the pipeline keeps running
    if data is None:
        print(f"[jd_parser] Warning: could not parse LLM output. Raw:\n{raw[:300]}")
        data = {}

    return {
        "required_skills": _as_list(data.get("required_skills")),
        "implied_skills":  _as_list(data.get("implied_skills")),
        "seniority":       str(data.get("seniority", "unknown")).lower(),
        "latent_needs":    _as_list(data.get("latent_needs")),
    }


def _try_json(text: str):
    """Return parsed dict if text is valid JSON, else None."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _as_list(value) -> list:
    """Coerce a value to a list — returns [] for None or unexpected types."""
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        return [value]
    return []


# ---------------------------------------------------------------------------
# Smoke test — run with:  python src/jd_parser.py
# Requires Ollama running: ollama serve
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_JD = """
    We are looking for a Senior Data Engineer to join our platform team.

    You will design, build, and maintain large-scale data pipelines using
    Apache Spark and Apache Airflow. Strong Python skills are required.
    Experience with AWS (S3, Glue, Redshift) is a must.

    You will work closely with data scientists and product managers to turn
    vague, evolving requirements into reliable, production-grade infrastructure.
    Experience with dbt and dimensional data modeling is highly valued.

    We move fast. Ambiguity is the norm, not the exception.
    We expect you to own your pipelines end to end and speak up when something
    is broken — even if it is not yours.
    """

    print("Parsing sample JD with local Ollama model...\n")
    result = parse_jd(SAMPLE_JD)
    print(json.dumps(result, indent=2))
