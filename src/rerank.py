"""
rerank.py — Stage 4: Honest LLM Rerank

God-level features added on top of the original:

  Citation grounding     — LLM must provide verbatim snippets from the profile.
                           A deterministic verifier checks each snippet against
                           the actual profile text. Hallucinated citations are
                           flagged and confidence is downgraded.

  Calibrated confidence  — high/medium/low becomes a real 0–1 score derived from
                           evidence coverage + score decisiveness. Now "confidence"
                           means something you can sort and filter on.

  Graceful degradation   — If the LLM is unreachable (Ollama down), falls back
                           to composite scores from Stage 3 instead of crashing.

Original design preserved:
  Parallel mode (default): all LLM calls fire simultaneously via ThreadPoolExecutor.
  Sequential fallback: rerank_stream() for single-threaded environments.
  Confidence weighting: low-confidence LLM score drifts toward composite (safe default).

CONFIDENCE_WEIGHT: high=1.0, medium=0.7, low=0.3
Final = composite * blend + effective_llm * (1-blend)
where effective_llm = conf_w * llm_norm + (1-conf_w) * composite
"""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Generator, Tuple

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from llm   import get_llm
from utils import safe_parse_json
from langchain_core.prompts import ChatPromptTemplate

log = logging.getLogger("viveka.rerank")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _cfg() -> dict:
    return {
        "max_chars":        int(os.getenv("VIVEKA_PROFILE_CHARS",    900)),
        "rerank_n":         int(os.getenv("VIVEKA_RERANK_N",          50)),
        "blend_composite":  float(os.getenv("VIVEKA_BLEND_COMPOSITE", 0.50)),
        "blend_llm":        float(os.getenv("VIVEKA_BLEND_LLM",       0.50)),
        "max_retries":      int(os.getenv("VIVEKA_LLM_RETRIES",         2)),
        "parallel_workers": int(os.getenv("VIVEKA_PARALLEL_WORKERS",    5)),
    }

CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.70, "low": 0.30}


# ---------------------------------------------------------------------------
# Prompts — updated to request verbatim evidence citations
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an honest technical recruiter. Score candidate fit.

Return ONLY a JSON object with exactly these four keys:
  "llm_score"  : integer 1 to 10
  "reason"     : one sentence using REAL words from the profile as evidence.
                 If evidence is thin, start with: "Limited evidence:"
                 NEVER invent skills or facts not in the profile.
  "confidence" : "high", "medium", or "low"
  "evidence"   : array of 1-3 SHORT phrases copied VERBATIM from the profile
                 that directly justify the score. Copy exact words — do not
                 paraphrase. If you cannot find supporting text, use [].

high   = clear direct evidence in the profile for the score
medium = some signals, some gaps
low    = profile lacks sufficient information

Start your reply with {{ and end with }}. No markdown, no explanation."""

USER_TEMPLATE = """Job summary:
{jd_summary}

Candidate profile:
{profile_text}

JSON:"""

INTERVIEW_SYSTEM = """You are a senior technical interviewer preparing for a candidate interview.
Return ONLY a JSON array of exactly 5 question strings. No other text."""

INTERVIEW_USER = """Job requirements: {jd_summary}

Candidate profile: {profile_text}
Recruiter assessment: {reason}
Skill gaps: {gaps}

Generate 5 tailored interview questions:
- 2 technical (test their claimed skills, probe depth)
- 2 behavioral (probe uncertainty areas and growth mindset)
- 1 motivation/fit question

JSON array of 5 strings:"""

IDEAL_SYSTEM = """You are an expert talent strategist. Return ONLY valid JSON."""

IDEAL_USER = """Job requirements: {jd_summary}

Describe the ideal candidate for this role. Return JSON with these exact keys:
{{
  "summary": "2-sentence profile of the ideal hire",
  "must_have": ["skill1", "skill2", "skill3"],
  "differentiators": ["thing that separates good from great"],
  "hidden_gem_signal": "one non-obvious indicator of excellence",
  "red_flag": "one warning sign to watch for",
  "estimated_market_scarcity": "rare | uncommon | available"
}}"""

OUTREACH_SYSTEM = """You are a recruiter writing a professional outreach message. Be concise and genuine."""

OUTREACH_USER = """Candidate: {name} — {title}
Role: {jd_summary}
Why they're a fit: {reason}

Write a 3-sentence LinkedIn outreach message. Mention one specific thing from their profile.
Return only the message text, no subject line."""


# ---------------------------------------------------------------------------
# Citation verifier (deterministic — zero LLM calls)
# ---------------------------------------------------------------------------

def verify_evidence(evidence: list, profile_text: str) -> Tuple[list, list]:
    """
    Check each claimed evidence snippet against the actual profile text.
    Uses case-insensitive substring match — fully deterministic, no LLM.

    Returns (verified, unsupported).
    Verified  = snippet actually appears in profile_text.
    Unsupported = snippet is NOT in profile_text → likely hallucinated.
    """
    text_lower = profile_text.lower()
    verified, unsupported = [], []
    for snippet in evidence:
        if not isinstance(snippet, str) or not snippet.strip():
            continue
        if snippet.strip().lower() in text_lower:
            verified.append(snippet)
        else:
            unsupported.append(snippet)
    return verified, unsupported


# ---------------------------------------------------------------------------
# Calibrated confidence (0–1 numeric score)
# ---------------------------------------------------------------------------

def calibrate_confidence(
    confidence_label: str,
    evidence_verified: list,
    evidence_unsupported: list,
    llm_score: int,
) -> float:
    """
    Convert high/medium/low + evidence coverage into a real 0–1 confidence score.

    Formula:
      base  = label anchor (high=0.85, medium=0.55, low=0.25)
      ev    = verified / (verified + unsupported) if any evidence given
      margin = abs(llm_score - 5) / 5   (1.0 when decisive, 0 when neutral)
      calibrated = 0.55 * base + 0.35 * ev + 0.10 * margin
    """
    base_map = {"high": 0.85, "medium": 0.55, "low": 0.25}
    base = base_map.get(str(confidence_label).lower(), 0.55)

    total_ev = len(evidence_verified) + len(evidence_unsupported)
    ev_score = len(evidence_verified) / total_ev if total_ev > 0 else base

    margin = abs(int(llm_score) - 5) / 5.0

    return round(0.55 * base + 0.35 * ev_score + 0.10 * margin, 4)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jd_summary(parsed_jd: dict) -> str:
    req       = ", ".join(parsed_jd.get("required_skills", []))
    implied   = ", ".join(parsed_jd.get("implied_skills",  []))
    seniority = parsed_jd.get("seniority", "unknown")
    needs     = "; ".join(parsed_jd.get("latent_needs",    []))
    return (
        f"Seniority: {seniority}. "
        f"Must-have: {req}. "
        f"Implied: {implied}. "
        f"Key traits: {needs}."
    )


def _make_profile_text(profile: dict, max_chars: int) -> str:
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


def _downgrade_confidence(confidence: str) -> str:
    """Downgrade confidence one level: high→medium→low."""
    return {"high": "medium", "medium": "low", "low": "low"}.get(
        confidence.lower(), "low"
    )


def _parse_result(raw: str) -> dict:
    fallback = {
        "analysis":   "",
        "llm_score":  5,
        "reason":     "Model output could not be parsed.",
        "confidence": "low",
        "evidence":   [],
    }
    data = safe_parse_json(raw, fallback)

    score = data.get("llm_score", 5)
    try:
        score = max(1, min(10, int(float(score))))
    except (TypeError, ValueError):
        score = 5

    evidence = data.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []

    return {
        "analysis":   str(data.get("analysis",   "")).strip(),
        "llm_score":  score,
        "reason":     str(data.get("reason",     "No reason provided.")).strip(),
        "confidence": str(data.get("confidence", "low")).lower(),
        "evidence":   evidence,
    }


def _final_score(composite: float, llm_score: int, confidence: str, cfg: dict) -> float:
    """
    Blend composite + LLM score, downweighting LLM when confidence is low.
    """
    llm_norm      = llm_score / 10.0
    conf_w        = CONFIDENCE_WEIGHT.get(confidence, 0.5)
    effective_llm = conf_w * llm_norm + (1.0 - conf_w) * composite
    return round(
        cfg["blend_composite"] * composite + cfg["blend_llm"] * effective_llm, 4
    )


def _score_one(candidate: dict, chain, jd_summary: str, cfg: dict) -> dict:
    """Score a single candidate with citation verification and retry on failure."""
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
                time.sleep(2 ** attempt)
    else:
        cid = candidate.get("id", "?")
        log.error("All retries failed for %s: %s", cid, last_exc)
        llm_result = {
            "analysis":   "",
            "llm_score":  0,
            "reason":     "Model call failed — scored 0 to avoid false ranking.",
            "confidence": "low",
            "evidence":   [],
        }

    # ── Citation verification (deterministic) ─────────────────────────────
    ev_verified, ev_unsupported = verify_evidence(
        llm_result.get("evidence", []), profile_text
    )

    # Downgrade confidence if the model cited text that isn't in the profile
    confidence = llm_result["confidence"]
    if ev_unsupported:
        confidence = _downgrade_confidence(confidence)
        log.debug(
            "Confidence downgraded for %s — unsupported citations: %s",
            candidate.get("id", "?"), ev_unsupported,
        )

    # ── Calibrated confidence (0–1 numeric) ───────────────────────────────
    cal_conf = calibrate_confidence(
        confidence, ev_verified, ev_unsupported, llm_result["llm_score"]
    )

    result = dict(candidate)
    result.update(llm_result)
    result["confidence"]           = confidence
    result["evidence_verified"]    = ev_verified
    result["evidence_unsupported"] = ev_unsupported
    result["calibrated_confidence"] = cal_conf
    result["final_score"] = _final_score(
        result.get("composite_score", 0.0),
        result["llm_score"],
        confidence,
        cfg,
    )
    return result


def _make_chain():
    """Create a fresh LangChain chain per thread."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  USER_TEMPLATE),
    ])
    return prompt | get_llm(json_mode=True)


# ---------------------------------------------------------------------------
# Public Reranking API
# ---------------------------------------------------------------------------

def rerank_stream_parallel(
    candidates:  List[Dict],
    parsed_jd:   dict,
    top_n:       int = None,
    max_workers: int = None,
) -> Generator[Dict, None, None]:
    """Parallel rerank — all LLM calls fire simultaneously."""
    cfg = _cfg()
    if top_n is None:
        top_n = cfg["rerank_n"]

    batch      = candidates[:top_n]
    workers    = max_workers or min(len(batch), cfg["parallel_workers"])
    jd_summary = _make_jd_summary(parsed_jd)

    log.info("Parallel scoring %d candidates  workers=%d", len(batch), workers)

    def _worker(candidate: dict) -> dict:
        chain = _make_chain()
        return _score_one(candidate, chain, jd_summary, cfg)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, c): c for c in batch}
        for future in as_completed(futures):
            try:
                yield future.result()
            except Exception as exc:
                c = futures[future]
                result = dict(c)
                result.update({
                    "analysis": "", "llm_score": 0,
                    "reason": f"LLM call failed: {exc}", "confidence": "low",
                    "evidence": [], "evidence_verified": [], "evidence_unsupported": [],
                    "calibrated_confidence": 0.25,
                })
                result["final_score"] = _final_score(
                    result.get("composite_score", 0.0), 0, "low", cfg
                )
                yield result


def rerank_stream(
    candidates: List[Dict],
    parsed_jd:  dict,
    top_n:      int = None,
) -> Generator[Dict, None, None]:
    """Sequential fallback — one candidate at a time."""
    cfg        = _cfg()
    if top_n is None:
        top_n = cfg["rerank_n"]

    batch      = candidates[:top_n]
    chain      = _make_chain()
    jd_summary = _make_jd_summary(parsed_jd)

    log.info("Sequential scoring %d candidates", len(batch))

    for candidate in tqdm(batch, desc="Reranking", unit="candidate"):
        yield _score_one(candidate, chain, jd_summary, cfg)


def rerank_candidates(
    candidates: List[Dict],
    parsed_jd:  dict,
    top_n:      int  = None,
    parallel:   bool = True,
) -> List[Dict]:
    """
    LLM-score top_n candidates and return sorted by final_score descending.

    Graceful degradation: if the LLM is completely unreachable, falls back
    to composite-score ordering from Stage 3 instead of crashing. The result
    is flagged with llm_unavailable=True so the caller can warn the user.
    """
    try:
        stream  = rerank_stream_parallel if parallel else rerank_stream
        results = list(stream(candidates, parsed_jd, top_n))
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
    except Exception as exc:
        log.error("LLM rerank failed entirely — falling back to composite scores: %s", exc)
        # Graceful degradation: use Stage 3 composite as final_score
        fallback = []
        for c in candidates[: (top_n or 50)]:
            r = dict(c)
            r["final_score"]           = r.get("composite_score", 0.0)
            r["llm_score"]             = None
            r["reason"]                = "LLM unavailable — ranked by composite score only."
            r["confidence"]            = "low"
            r["calibrated_confidence"] = 0.25
            r["evidence"]              = []
            r["evidence_verified"]     = []
            r["evidence_unsupported"]  = []
            r["llm_unavailable"]       = True
            fallback.append(r)
        fallback.sort(key=lambda x: x["final_score"], reverse=True)
        return fallback


# ---------------------------------------------------------------------------
# Interview Questions Generator
# ---------------------------------------------------------------------------

def generate_interview_questions(candidate: dict, parsed_jd: dict) -> list:
    cfg          = _cfg()
    llm          = get_llm(json_mode=True)
    prompt       = ChatPromptTemplate.from_messages([
        ("system", INTERVIEW_SYSTEM),
        ("human",  INTERVIEW_USER),
    ])
    chain = prompt | llm

    jd_summary   = _make_jd_summary(parsed_jd)
    profile_text = _make_profile_text(candidate, cfg["max_chars"])
    reason       = candidate.get("reason", "")
    ev           = candidate.get("skill_evidence", {})
    gaps         = ", ".join(ev.get("required_missing", [])) or "none identified"

    try:
        response  = chain.invoke({
            "jd_summary":   jd_summary,
            "profile_text": profile_text,
            "reason":       reason,
            "gaps":         gaps,
        })
        questions = safe_parse_json(response.content, [])
        if isinstance(questions, list) and questions:
            return [str(q) for q in questions[:7]]
    except Exception as exc:
        log.warning("interview_questions failed: %s", exc)

    req = parsed_jd.get("required_skills", ["required skills"])
    return [
        f"Walk me through your hands-on experience with {req[0] if req else 'the core technologies'}.",
        "Describe the most complex data pipeline you've owned end-to-end.",
        "Tell me about a time you delivered under significant ambiguity.",
        "How do you stay current with fast-moving data engineering tools?",
        "What would you build or improve in the first 90 days in this role?",
    ]


# ---------------------------------------------------------------------------
# Ideal Candidate Blueprint Generator
# ---------------------------------------------------------------------------

def generate_ideal_candidate(parsed_jd: dict) -> dict:
    llm    = get_llm(json_mode=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system", IDEAL_SYSTEM),
        ("human",  IDEAL_USER),
    ])
    chain      = prompt | llm
    jd_summary = _make_jd_summary(parsed_jd)

    fallback = {
        "summary":                   "Experienced professional with strong technical skills.",
        "must_have":                 parsed_jd.get("required_skills", [])[:3],
        "differentiators":           ["Strong ownership mentality", "Clear communicator"],
        "hidden_gem_signal":         "Open-source contributions or side projects in the domain",
        "red_flag":                  "Job-hopping pattern with no evidence of deep work",
        "estimated_market_scarcity": "uncommon",
    }

    try:
        response = chain.invoke({"jd_summary": jd_summary})
        data     = safe_parse_json(response.content, fallback)
        if isinstance(data, dict) and "summary" in data:
            return data
    except Exception as exc:
        log.warning("ideal_candidate failed: %s", exc)

    return fallback


# ---------------------------------------------------------------------------
# Outreach Email Generator
# ---------------------------------------------------------------------------

def generate_outreach_message(candidate: dict, parsed_jd: dict) -> str:
    cfg    = _cfg()
    llm    = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", OUTREACH_SYSTEM),
        ("human",  OUTREACH_USER),
    ])
    chain = prompt | llm

    name       = candidate.get("name") or candidate.get("id", "there")
    title      = candidate.get("title", "")
    jd_summary = _make_jd_summary(parsed_jd)
    reason     = candidate.get("reason", "")

    try:
        response = chain.invoke({
            "name":       name,
            "title":      title,
            "jd_summary": jd_summary,
            "reason":     reason,
        })
        return response.content.strip()
    except Exception as exc:
        log.warning("outreach failed: %s", exc)
        return (
            f"Hi {name}, I came across your profile and was impressed by your background "
            f"in {title}. We're hiring for a role that seems like a great match — "
            f"would love to connect and share more details!"
        )


# ---------------------------------------------------------------------------
# Smoke test — python src/rerank.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
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
    ]

    PARSED_JD = {
        "required_skills": ["Python", "Apache Spark", "Apache Airflow", "AWS"],
        "implied_skills":  ["Git", "SQL", "Linux", "dbt"],
        "seniority":       "senior",
        "latent_needs":    ["owns pipelines end-to-end", "works under ambiguity"],
    }

    print("=== Parallel rerank test ===\n")
    results = rerank_candidates(CANDIDATES, PARSED_JD, top_n=2, parallel=True)
    for i, c in enumerate(results, 1):
        gem = " ★ HIDDEN GEM" if c.get("hidden_gem") else ""
        print(f"{i}. [{c['id']}] {c['title']}{gem}")
        print(f"   composite={c['composite_score']}  llm={c['llm_score']}/10  "
              f"conf={c['confidence']}  cal_conf={c['calibrated_confidence']}  final={c['final_score']}")
        print(f"   Reason: {c['reason']}")
        print(f"   Evidence verified: {c['evidence_verified']}")
        print(f"   Evidence unsupported: {c['evidence_unsupported']}\n")
