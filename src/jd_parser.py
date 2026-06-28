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
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from llm import get_llm

log = logging.getLogger("viveka.jd_parser")
from utils import safe_parse_json, as_list
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Prompts
# Local models need shorter, more explicit instructions than cloud models.
# Ending the user turn with "JSON:" steers the model straight into output.
# Double braces {{ }} are LangChain template escapes that render as { }.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a senior technical recruiter. Analyze job descriptions and extract structured data.

You MUST return ONLY a JSON object — nothing else. No markdown, no explanation, no code fences.
The JSON must have exactly these four keys:
  "required_skills"  - array of skills explicitly mentioned in the job description
  "implied_skills"   - array of skills not stated but obviously needed (e.g. Git for any dev role)
  "seniority"        - exactly one string: "junior", "mid", or "senior"
  "latent_needs"     - array of short phrases describing what this role truly tests

Example output format (use this exact structure):
{{"required_skills": ["Python", "Apache Spark", "SQL"], "implied_skills": ["Git", "Linux", "Docker"], "seniority": "senior", "latent_needs": ["owns outcomes end-to-end", "works under ambiguity"]}}

Start your response with {{ and end with }}. Nothing before or after the JSON."""

USER_TEMPLATE = """Job Description:
{jd_text}

JSON:"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_jd(jd_text: str) -> dict:
    """
    Parse a job description and return a structured dict.

    Returns dict with keys: required_skills, implied_skills, seniority, latent_needs.
    All keys are always present — empty lists / "unknown" on parse failure.
    """
    llm = get_llm(json_mode=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  USER_TEMPLATE),
    ])
    chain = prompt | llm
    response = chain.invoke({"jd_text": jd_text.strip()})

    fallback = {"required_skills": [], "implied_skills": [], "seniority": "unknown", "latent_needs": []}
    data = safe_parse_json(response.content, fallback)

    if data is fallback:
        log.warning("Could not parse LLM output. Raw:\n%s", response.content[:300])

    return {
        "required_skills": as_list(data.get("required_skills")),
        "implied_skills":  as_list(data.get("implied_skills")),
        "seniority":       str(data.get("seniority", "unknown")).lower(),
        "latent_needs":    as_list(data.get("latent_needs")),
    }


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
