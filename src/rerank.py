"""
rerank.py — Stage 4: Honest LLM Rerank

For the top ~50 candidates from Stage 3, asks the local LLM to:
  - Score fit on a 1–10 scale
  - Write a 1-2 sentence reason grounded ONLY in real profile text
  - State confidence: high / medium / low
  - Explicitly admit when evidence is thin — never invent facts

Final score = 50% composite_score (Stage 3) + 50% llm_score (normalised).
Blending keeps the fast signals from Stage 3 while adding LLM judgment.

Note: with a local model, each candidate takes 5–15 s.
50 candidates ≈ 5–12 minutes. Reduce MANTHAN_RERANK_N to speed up tests.
"""

import sys
import os
from typing import List, Dict
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from llm import get_llm
from utils import safe_parse_json
from langchain_core.prompts import ChatPromptTemplate


def _cfg() -> dict:
    """Read all tuneable config from env at call time — not at import time."""
    return {
        "max_chars":       int(float(os.getenv("MANTHAN_PROFILE_CHARS", 900))),
        "rerank_n":        int(os.getenv("MANTHAN_RERANK_N",        50)),
        "blend_composite": float(os.getenv("MANTHAN_BLEND_COMPOSITE", 0.5)),
        "blend_llm":       float(os.getenv("MANTHAN_BLEND_LLM",       0.5)),
    }


# ---------------------------------------------------------------------------
# Prompt
# Ends with "JSON:" to push the local model straight into output mode.
# Rules are explicit because local models cut corners when instructions are vague.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an honest, careful technical recruiter scoring candidate fit.

Given a job summary and a candidate profile, return ONLY a JSON object with three keys:
  "llm_score"  : integer 1 to 10 (10 = exceptional fit, 1 = no fit)
  "reason"     : 1 or 2 sentences. Cite specific text FROM the profile as evidence.
                 If evidence is thin, start with: "Limited evidence:"
                 NEVER invent skills, experience, or facts not in the profile.
  "confidence" : "high", "medium", or "low"
                 high   = profile has clear, direct evidence for the score
                 medium = some relevant signals, some gaps
                 low    = profile lacks information to score reliably

Return ONLY the JSON. No markdown, no preamble, no explanation. Start with {{"""

USER_TEMPLATE = """Job summary:
{jd_summary}

Candidate profile:
{profile_text}

JSON:"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jd_summary(parsed_jd: dict) -> str:
    """Compact one-liner the LLM uses as the job spec."""
    req      = ", ".join(parsed_jd.get("required_skills", []))
    implied  = ", ".join(parsed_jd.get("implied_skills",  []))
    seniority = parsed_jd.get("seniority", "unknown")
    needs    = "; ".join(parsed_jd.get("latent_needs",    []))
    return (
        f"Seniority: {seniority}. "
        f"Must-have skills: {req}. "
        f"Implied skills: {implied}. "
        f"Key traits needed: {needs}."
    )


def _make_profile_text(profile: dict, max_chars: int) -> str:
    """
    Extract key-value lines from the profile, stopping on a clean line boundary
    so the LLM never sees a broken mid-value truncation.
    """
    priority_fields = [
        "title", "headline", "current_role",
        "summary", "bio", "about",
        "skills", "tech_skills", "tools",
        "experience", "work_history",
        "education", "certifications",
        "github_repos", "projects", "endorsements",
    ]
    lines, total = [], 0
    for field in priority_fields:
        val = profile.get(field)
        if not val:
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        line = f"{field}: {str(val).strip()}"
        if total + len(line) > max_chars:
            break                       # stop at a clean line boundary
        lines.append(line)
        total += len(line) + 1          # +1 for the newline

    return "\n".join(lines)


def _parse_llm_result(raw: str) -> dict:
    """Parse LLM output — delegates to shared utils, then clamps the score."""
    fallback = {"llm_score": 5, "reason": "Model output could not be parsed.", "confidence": "low"}
    data = safe_parse_json(raw, fallback)

    score = data.get("llm_score", 5)
    try:
        score = max(1, min(10, int(score)))
    except (TypeError, ValueError):
        score = 5

    return {
        "llm_score":  score,
        "reason":     str(data.get("reason", "No reason provided.")).strip(),
        "confidence": str(data.get("confidence", "low")).lower(),
    }


def _final_score(composite: float, llm_score: int, cfg: dict) -> float:
    return round(cfg["blend_composite"] * composite + cfg["blend_llm"] * (llm_score / 10.0), 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank_candidates(
    candidates: List[Dict],
    parsed_jd: dict,
    top_n: int = None,
) -> List[Dict]:
    """
    LLM-score the top_n candidates and return them sorted by final_score.

    Args:
        candidates:  Stage 3 output (sorted by composite_score descending).
        parsed_jd:   Structured JD dict from Stage 1.
        top_n:       How many candidates to send to the LLM.
                     Defaults to MANTHAN_RERANK_N env var (50).

    Returns:
        List sorted by final_score descending, each with added keys:
        llm_score, reason, confidence, final_score.
    """
    cfg = _cfg()
    if top_n is None:
        top_n = cfg["rerank_n"]

    candidates = candidates[:top_n]

    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  USER_TEMPLATE),
    ])
    chain = prompt | llm

    jd_summary = _make_jd_summary(parsed_jd)

    print(f"[rerank] Scoring {len(candidates)} candidates with local model…")
    print(f"         Tip: set MANTHAN_RERANK_N=5 for a quick test run.\n")

    results = []
    for candidate in tqdm(candidates, desc="Reranking", unit="candidate"):
        profile_text = _make_profile_text(candidate, cfg["max_chars"])

        try:
            response   = chain.invoke({"jd_summary": jd_summary, "profile_text": profile_text})
            llm_result = _parse_llm_result(response.content)
        except Exception as exc:
            cid = candidate.get("id", "?")
            print(f"\n[rerank] Warning: model call failed for {cid}: {exc}")
            llm_result = {
                "llm_score":  0,
                "reason":     "Model call failed — scored 0 to avoid false ranking.",
                "confidence": "low",
            }

        result = dict(candidate)
        result.update(llm_result)
        result["final_score"] = _final_score(result.get("composite_score", 0.0), result["llm_score"], cfg)
        results.append(result)

    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Smoke test — run with: python src/rerank.py
# Set MANTHAN_RERANK_N=3 to only score 3 candidates for a quick check.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate Stage 3 output — candidates already have composite_score
    CANDIDATES = [
        {
            "id": "C001", "title": "Senior Data Engineer",
            "skills": ["Python", "Apache Spark", "Airflow", "AWS", "dbt", "SQL"],
            "summary": "8 years building large-scale data pipelines on AWS. Led platform migration from on-prem Hadoop to S3+Glue+Redshift.",
            "embedding_score": 0.91, "skill_score": 0.75,
            "activity_score": 0.50, "composite_score": 0.727,
            "hidden_gem": False,
        },
        {
            "id": "C004", "title": "Data Engineer",
            "skills": ["Spark", "Airflow", "Python"],
            "summary": "Built batch pipelines on AWS. Self-taught. No degree. Ships fast.",
            "embedding_score": 0.78, "skill_score": 0.50,
            "activity_score": 0.50, "composite_score": 0.614,
            "hidden_gem": True,
        },
        {
            "id": "C003", "title": "Data Analyst",
            "skills": ["SQL", "Python", "Tableau"],
            "summary": "Strong SQL and data storytelling at a fintech. No pipeline experience.",
            "embedding_score": 0.62, "skill_score": 0.25,
            "activity_score": 0.50, "composite_score": 0.435,
            "hidden_gem": False,
        },
    ]

    PARSED_JD = {
        "required_skills": ["Python", "Apache Spark", "Airflow", "AWS"],
        "implied_skills":  ["Git", "SQL", "Linux", "dbt"],
        "seniority":       "senior",
        "latent_needs":    ["owns pipelines end-to-end", "works under ambiguity"],
    }

    results = rerank_candidates(CANDIDATES, PARSED_JD, top_n=3)

    print("\n--- Final Reranked Shortlist ---\n")
    for i, c in enumerate(results, 1):
        gem = " ★ HIDDEN GEM" if c.get("hidden_gem") else ""
        print(f"{i}. [{c['id']}] {c['title']}{gem}")
        print(f"   composite={c['composite_score']}  llm={c['llm_score']}/10  "
              f"confidence={c['confidence']}  final={c['final_score']}")
        print(f"   Reason: {c['reason']}")
        print()
