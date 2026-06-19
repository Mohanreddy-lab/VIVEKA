"""
rerank.py — Stage 4: Honest LLM Rerank

For the top candidates from Stage 3, asks the local LLM to:
  - Think step by step (chain-of-thought) before scoring
  - Score fit 1-10 with evidence from the ACTUAL profile text
  - State confidence and admit thin evidence honestly

Confidence weighting: a low-confidence LLM score moves toward the
Stage 3 composite (safe default) rather than overriding it.

CONFIDENCE_WEIGHT: high=1.0, medium=0.7, low=0.3
Final = composite * blend + effective_llm * (1-blend)
where effective_llm = conf_w * llm_norm + (1-conf_w) * composite

Provides both a batch function and a streaming generator so the
Streamlit demo can show results live as each candidate is scored.
"""

import os
import sys
import time
from typing import List, Dict, Generator

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from llm   import get_llm
from utils import safe_parse_json
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Config (read at call time so env changes take effect without restart)
# ---------------------------------------------------------------------------

def _cfg() -> dict:
    return {
        "max_chars":       int(os.getenv("MANTHAN_PROFILE_CHARS",  900)),
        "rerank_n":        int(os.getenv("MANTHAN_RERANK_N",        50)),
        "blend_composite": float(os.getenv("MANTHAN_BLEND_COMPOSITE", 0.50)),
        "blend_llm":       float(os.getenv("MANTHAN_BLEND_LLM",       0.50)),
        "max_retries":     int(os.getenv("MANTHAN_LLM_RETRIES",        2)),
    }

CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.70, "low": 0.30}


# ---------------------------------------------------------------------------
# Prompt — chain-of-thought forces the model to reason before scoring
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an honest technical recruiter. Score candidate fit.

Return ONLY a JSON object with exactly these three keys:
  "llm_score"  : integer 1 to 10
  "reason"     : one sentence using REAL words from the profile as evidence.
                 If evidence is thin, start with: "Limited evidence:"
                 NEVER invent skills or facts not in the profile.
  "confidence" : "high", "medium", or "low"

high   = clear direct evidence for the score
medium = some signals, some gaps
low    = profile lacks information

Start your reply with {{ and end with }}. No markdown, no explanation."""

USER_TEMPLATE = """Job summary:
{jd_summary}

Candidate profile:
{profile_text}

JSON:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jd_summary(parsed_jd: dict) -> str:
    req      = ", ".join(parsed_jd.get("required_skills", []))
    implied  = ", ".join(parsed_jd.get("implied_skills",  []))
    seniority = parsed_jd.get("seniority", "unknown")
    needs    = "; ".join(parsed_jd.get("latent_needs",    []))
    return (
        f"Seniority: {seniority}. "
        f"Must-have: {req}. "
        f"Implied: {implied}. "
        f"Key traits: {needs}."
    )


def _make_profile_text(profile: dict, max_chars: int) -> str:
    """Build a readable profile string, stopping cleanly on a line boundary."""
    priority = [
        "title", "headline", "current_role",
        "summary", "bio", "about",
        "skills", "tech_skills", "tools",
        "experience", "work_history",
        "education", "certifications",
        "github_repos", "projects", "endorsements",
    ]
    lines, total = [], 0
    for field in priority:
        val = profile.get(field)
        if not val:
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        line = f"{field}: {str(val).strip()}"
        if total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _parse_result(raw: str) -> dict:
    fallback = {
        "analysis":   "",
        "llm_score":  5,
        "reason":     "Model output could not be parsed.",
        "confidence": "low",
    }
    data = safe_parse_json(raw, fallback)

    score = data.get("llm_score", 5)
    try:
        score = max(1, min(10, int(float(score))))
    except (TypeError, ValueError):
        score = 5

    return {
        "analysis":   str(data.get("analysis",   "")).strip(),
        "llm_score":  score,
        "reason":     str(data.get("reason",     "No reason provided.")).strip(),
        "confidence": str(data.get("confidence", "low")).lower(),
    }


def _final_score(composite: float, llm_score: int, confidence: str, cfg: dict) -> float:
    """
    Blend composite + LLM score, but downweight the LLM score when confidence is low.
    Low-confidence LLM scores drift toward the composite (the safer signal).
    """
    llm_norm = llm_score / 10.0
    conf_w   = CONFIDENCE_WEIGHT.get(confidence, 0.5)
    # effective_llm: weighted between LLM's own score and the composite fallback
    effective_llm = conf_w * llm_norm + (1.0 - conf_w) * composite
    return round(
        cfg["blend_composite"] * composite + cfg["blend_llm"] * effective_llm, 4
    )


def _score_one(candidate: dict, chain, jd_summary: str, cfg: dict) -> dict:
    """Score a single candidate with retry on failure."""
    profile_text = _make_profile_text(candidate, cfg["max_chars"])
    last_exc = None

    for attempt in range(cfg["max_retries"] + 1):
        try:
            response   = chain.invoke({"jd_summary": jd_summary, "profile_text": profile_text})
            llm_result = _parse_result(response.content)
            break
        except Exception as exc:
            last_exc = exc
            if attempt < cfg["max_retries"]:
                time.sleep(2 ** attempt)   # exponential back-off: 1s, 2s
    else:
        cid = candidate.get("id", "?")
        print(f"\n[rerank] All retries failed for {cid}: {last_exc}")
        llm_result = {
            "analysis":   "",
            "llm_score":  0,
            "reason":     "Model call failed — scored 0 to avoid false ranking.",
            "confidence": "low",
        }

    result = dict(candidate)
    result.update(llm_result)
    result["final_score"] = _final_score(
        result.get("composite_score", 0.0),
        result["llm_score"],
        result["confidence"],
        cfg,
    )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank_candidates(
    candidates: List[Dict],
    parsed_jd:  dict,
    top_n:      int = None,
) -> List[Dict]:
    """
    LLM-score top_n candidates and return them sorted by final_score descending.
    Blocking — waits for all results before returning.
    For live streaming use rerank_stream() instead.
    """
    results = list(rerank_stream(candidates, parsed_jd, top_n))
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


def rerank_stream(
    candidates: List[Dict],
    parsed_jd:  dict,
    top_n:      int = None,
) -> Generator[Dict, None, None]:
    """
    Generator: yield each candidate result as soon as the LLM finishes it.
    Use in the Streamlit demo to show live updates.
    """
    cfg = _cfg()
    if top_n is None:
        top_n = cfg["rerank_n"]

    batch = candidates[:top_n]

    llm   = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  USER_TEMPLATE),
    ])
    chain = prompt | llm

    jd_summary = _make_jd_summary(parsed_jd)
    print(f"[rerank] Scoring {len(batch)} candidates  (MANTHAN_RERANK_N={top_n})")
    print(f"         Confidence weights: high=1.0, medium=0.70, low=0.30\n")

    for candidate in tqdm(batch, desc="Reranking", unit="candidate"):
        yield _score_one(candidate, chain, jd_summary, cfg)


# ---------------------------------------------------------------------------
# Smoke test — python src/rerank.py   (set MANTHAN_RERANK_N=3 for speed)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    CANDIDATES = [
        {
            "id": "C001", "title": "Senior Data Engineer",
            "skills": ["Python", "PySpark", "Airflow", "AWS", "dbt", "SQL"],
            "summary": "8 years building data pipelines on AWS. Led Hadoop → S3/Glue migration.",
            "embedding_score": 0.91, "skill_score": 0.80,
            "seniority_score": 1.0,  "activity_score": 0.50, "composite_score": 0.742,
            "hidden_gem": False,
        },
        {
            "id": "C004", "title": "Data Engineer",
            "skills": ["Spark", "Airflow", "Python"],
            "summary": "Built batch pipelines on AWS. Self-taught. No degree. Ships fast.",
            "embedding_score": 0.78, "skill_score": 0.55,
            "seniority_score": 0.55, "activity_score": 0.50, "composite_score": 0.598,
            "hidden_gem": True,
        },
        {
            "id": "C003", "title": "Data Analyst",
            "skills": ["SQL", "Python", "Tableau"],
            "summary": "Strong SQL and storytelling. No pipeline experience.",
            "embedding_score": 0.62, "skill_score": 0.20,
            "seniority_score": 0.55, "activity_score": 0.50, "composite_score": 0.382,
            "hidden_gem": False,
        },
    ]

    PARSED_JD = {
        "required_skills": ["Python", "Apache Spark", "Apache Airflow", "AWS"],
        "implied_skills":  ["Git", "SQL", "Linux", "dbt"],
        "seniority":       "senior",
        "latent_needs":    ["owns pipelines end-to-end", "works under ambiguity"],
    }

    print("Reranking with chain-of-thought...\n")
    results = rerank_candidates(CANDIDATES, PARSED_JD, top_n=3)

    for i, c in enumerate(results, 1):
        gem = " ★ HIDDEN GEM" if c.get("hidden_gem") else ""
        print(f"{i}. [{c['id']}] {c['title']}{gem}")
        print(f"   composite={c['composite_score']}  llm={c['llm_score']}/10  "
              f"conf={c['confidence']}  final={c['final_score']}")
        if c.get("analysis"):
            print(f"   Analysis: {c['analysis'][:120]}")
        print(f"   Reason: {c['reason']}")
        print()
