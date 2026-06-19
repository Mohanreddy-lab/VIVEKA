"""
scoring.py — Stage 3: Multi-signal Scoring

Blends four signals into a composite score:
  1. embedding_score  — semantic similarity (Stage 2, normalised per batch)
  2. skill_score      — synonym-aware weighted skill overlap
  3. seniority_score  — does candidate level match JD seniority?
  4. activity_score   — behavioral signals (GitHub, projects, etc.)

Weights (tune via env vars, always auto-normalised):
  MANTHAN_W_EMBED      default 0.30
  MANTHAN_W_SKILL      default 0.40
  MANTHAN_W_SENIORITY  default 0.15
  MANTHAN_W_ACTIVITY   default 0.15

Hidden gem detection: candidates whose composite rank is significantly
higher than their raw embedding rank — i.e., the skill/seniority signals
surfaced them from below the semantic horizon.
"""

import os
from typing import List, Dict

from skills import skill_matches as _skill_match, matched_skills
from config import (
    get_weights,
    MAX_GITHUB_REPOS, MAX_PROJECTS, MAX_ENDORSEMENTS,
    HIDDEN_GEM_MIN_COMPOSITE, HIDDEN_GEM_MIN_RANK_JUMP, HIDDEN_GEM_MIN_SKILL,
    SENIORITY_RANK, SENIOR_WORDS, LEAD_WORDS, MID_WORDS, JUNIOR_WORDS,
    SKILL_FIELDS, SENIORITY_TEXT_FIELDS,
)


# ---------------------------------------------------------------------------
# Signal 2: Skill overlap (synonym-aware)
# ---------------------------------------------------------------------------

def _build_candidate_text(profile: dict) -> str:
    """Collect all text from skill-relevant fields."""
    parts = []
    for f in SKILL_FIELDS:
        val = profile.get(f)
        if not val:
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        parts.append(str(val))
    return " ".join(parts)


def _skill_score(candidate: dict, parsed_jd: dict) -> float:
    """
    Return score 0–1 for how well candidate covers JD skills.
    Required skills worth 1.0 each; implied 0.5. Synonym-aware.
    """
    score, _ = _skill_score_with_evidence(candidate, parsed_jd)
    return score


def _skill_score_with_evidence(candidate: dict, parsed_jd: dict) -> tuple[float, dict]:
    """Like _skill_score but also returns matched/missing evidence dict."""
    required = parsed_jd.get("required_skills", [])
    implied  = parsed_jd.get("implied_skills",  [])

    if not required and not implied:
        return 0.0, {}

    text = _build_candidate_text(candidate)
    req_hit,  req_miss  = matched_skills(required, text)
    impl_hit, impl_miss = matched_skills(implied,  text)

    max_score = len(required) * 1.0 + len(implied) * 0.5
    if max_score == 0:
        return 0.0, {}

    earned = len(req_hit) * 1.0 + len(impl_hit) * 0.5
    score  = min(earned / max_score, 1.0)

    evidence = {
        "required_matched": req_hit,
        "required_missing": req_miss,
        "implied_matched":  impl_hit,
        "implied_missing":  impl_miss,
    }
    return score, evidence


# ---------------------------------------------------------------------------
# Signal 3: Seniority match
# ---------------------------------------------------------------------------

def _infer_seniority(profile: dict) -> int:
    """Infer candidate seniority level (1–4) from title / experience fields."""
    text = " ".join(
        str(profile.get(f, "")) for f in SENIORITY_TEXT_FIELDS
    ).lower()

    if any(w in text for w in LEAD_WORDS):
        return 4
    if any(w in text for w in SENIOR_WORDS):
        return 3
    if any(w in text for w in JUNIOR_WORDS):
        return 1
    if any(w in text for w in MID_WORDS):
        return 2
    return 2   # default: mid


def _seniority_score(candidate: dict, jd_seniority: str) -> float:
    """1.0 if candidate matches or exceeds JD level, lower if below."""
    jd_level   = SENIORITY_RANK.get(jd_seniority.lower(), 2)
    cand_level = _infer_seniority(candidate)
    diff = cand_level - jd_level
    if diff >= 0:
        return 1.0    # at level or above
    if diff == -1:
        return 0.55   # one step below
    return 0.15       # two+ steps below — strong signal against


# ---------------------------------------------------------------------------
# Signal 4: Activity / behavior
# ---------------------------------------------------------------------------

def _activity_score(candidate: dict) -> float:
    """
    Score behavioral signals 0–1.
    Returns 0.0 when no activity data exists (honest: no data = no signal).
    """
    signals = []

    repos = candidate.get("github_repos") or candidate.get("repositories")
    if repos is not None:
        signals.append(min(int(repos) / MAX_GITHUB_REPOS, 1.0))

    projects = candidate.get("projects") or candidate.get("project_count")
    if projects is not None:
        count = len(projects) if isinstance(projects, list) else int(projects)
        signals.append(min(count / MAX_PROJECTS, 1.0))

    endorsements = candidate.get("endorsements") or candidate.get("recommendations")
    if endorsements is not None:
        count = len(endorsements) if isinstance(endorsements, list) else int(endorsements)
        signals.append(min(count / MAX_ENDORSEMENTS, 1.0))

    return sum(signals) / len(signals) if signals else 0.0


# ---------------------------------------------------------------------------
# Hidden gem: exported helper + rank-jump detection
# ---------------------------------------------------------------------------

def _is_hidden_gem(candidate: dict, composite: float, skill: float) -> bool:
    """
    True if this candidate has a strong score but an unassuming profile.
    Checks composite threshold, skill threshold, and title modesty.
    """
    if composite < HIDDEN_GEM_MIN_COMPOSITE:
        return False
    if skill < HIDDEN_GEM_MIN_SKILL:
        return False
    title = " ".join([
        str(candidate.get("title",    "")),
        str(candidate.get("headline", "")),
    ]).lower()
    if any(w in title for w in SENIOR_WORDS | LEAD_WORDS):
        return False
    return True


def _detect_hidden_gems(scored: List[Dict]) -> List[Dict]:
    """
    Mark candidates whose composite rank is much better than embed rank.
    These are people semantic search almost missed, but skill + seniority
    signals surfaced them.
    """
    composite_rank = {c["_id"]: i for i, c in enumerate(scored)}

    by_embed   = sorted(scored, key=lambda c: c["embedding_score"], reverse=True)
    embed_rank = {c["_id"]: i for i, c in enumerate(by_embed)}

    for c in scored:
        cid  = c["_id"]
        jump = embed_rank[cid] - composite_rank[cid]   # positive = rose after scoring
        c["hidden_gem"] = (
            c["composite_score"] >= HIDDEN_GEM_MIN_COMPOSITE
            and jump >= HIDDEN_GEM_MIN_RANK_JUMP
        )
        c["_rank_jump"] = jump

    return scored


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_candidates(
    candidates: List[Dict],
    parsed_jd:  dict,
    top_n:      int = 50,
) -> List[Dict]:
    """
    Score recalled candidates on four signals and return top_n sorted descending.

    Adds keys to each candidate:
      skill_score, seniority_score, activity_score, composite_score,
      skill_evidence, hidden_gem, _rank_jump.
    """
    w_embed, w_skill, w_seniority, w_activity = get_weights()
    jd_seniority = parsed_jd.get("seniority", "mid")

    # Normalise embedding scores across this batch (they cluster in 0.6–0.9)
    raw_embeds = [float(c.get("embedding_score", 0.0)) for c in candidates]
    e_min, e_max = min(raw_embeds), max(raw_embeds)
    e_range = (e_max - e_min) or 1.0

    scored = []
    for c, raw_e in zip(candidates, raw_embeds):
        embed              = (raw_e - e_min) / e_range
        skill, evidence    = _skill_score_with_evidence(c, parsed_jd)
        seniority          = _seniority_score(c, jd_seniority)
        activity           = _activity_score(c)

        composite = (
            w_embed     * embed     +
            w_skill     * skill     +
            w_seniority * seniority +
            w_activity  * activity
        )

        result = dict(c)
        result["_id"]             = c.get("id") or c.get("candidate_id") or id(c)
        result["skill_score"]     = round(skill,     4)
        result["seniority_score"] = round(seniority, 4)
        result["activity_score"]  = round(activity,  4)
        result["composite_score"] = round(composite, 4)
        result["skill_evidence"]  = evidence
        result["hidden_gem"]      = False   # set by _detect_hidden_gems
        scored.append(result)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    scored = _detect_hidden_gems(scored)

    return scored[:top_n]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    CANDIDATES = [
        {
            "id": "C001", "title": "Senior Data Engineer",
            "skills": ["Python", "PySpark", "Airflow", "AWS", "dbt", "SQL"],
            "summary": "8 years building large-scale data pipelines.",
            "github_repos": 42, "embedding_score": 0.91,
        },
        {
            "id": "C002", "title": "Junior Frontend Developer",
            "skills": ["React", "TypeScript", "CSS"],
            "summary": "2 years building web UIs.",
            "github_repos": 8, "embedding_score": 0.31,
        },
        {
            "id": "C003", "title": "Data Analyst",
            "skills": ["SQL", "Python", "Tableau"],
            "summary": "Strong SQL and data storytelling.",
            "embedding_score": 0.62,
        },
        {
            "id": "C004", "title": "Data Engineer",
            "skills": ["Spark", "Airflow", "Python"],
            "summary": "Built batch pipelines on AWS. Self-taught.",
            "embedding_score": 0.78,
        },
    ]

    PARSED_JD = {
        "required_skills": ["Python", "Apache Spark", "Apache Airflow", "AWS"],
        "implied_skills":  ["Git", "SQL", "Linux", "dbt"],
        "seniority":       "senior",
        "latent_needs":    ["owns pipelines end-to-end"],
    }

    results = score_candidates(CANDIDATES, PARSED_JD, top_n=5)

    print(f"{'#':<3} {'ID':<6} {'Title':<26} {'Embed':>6} {'Skill':>6} {'Sen':>5} {'Act':>5} {'Score':>7} {'Gem'}")
    print("-" * 75)
    for i, c in enumerate(results, 1):
        gem = "★" if c["hidden_gem"] else ""
        ev  = c["skill_evidence"]
        print(
            f"{i:<3} {c['id']:<6} {c['title']:<26} "
            f"{c['embedding_score']:>6.3f} {c['skill_score']:>6.3f} "
            f"{c['seniority_score']:>5.2f} {c['activity_score']:>5.2f} "
            f"{c['composite_score']:>7.3f}  {gem}"
        )
        print(f"     Hit: {ev.get('required_matched')}  Miss: {ev.get('required_missing')}")
