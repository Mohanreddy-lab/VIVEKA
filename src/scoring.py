"""
scoring.py — Stage 3: Multi-signal Scoring

Blends four signals into a composite score:
  1. embedding_score  — semantic similarity (Stage 2, normalised per batch)
  2. skill_score      — synonym-aware weighted skill overlap
  3. seniority_score  — does candidate level match JD seniority?
  4. activity_score   — behavioral signals (GitHub, projects, etc.)

God-level additions:
  · Keyword-stuffing detector: skills listed but never mentioned in narrative
    prose get a gentle penalty. Résumé padding doesn't win.
  · Counterfactual explainer: "Would rank #N with evidence of X" — deterministic,
    free, computed from the same scoring math.

Weights (tune via env vars, always auto-normalised):
  MANTHAN_W_EMBED      default 0.30
  MANTHAN_W_SKILL      default 0.40
  MANTHAN_W_SENIORITY  default 0.15
  MANTHAN_W_ACTIVITY   default 0.15
"""

import os
from typing import List, Dict, Tuple

from skills import skill_matches as _skill_match, matched_skills
from config import (
    get_weights,
    MAX_GITHUB_REPOS, MAX_PROJECTS, MAX_ENDORSEMENTS,
    HIDDEN_GEM_MIN_COMPOSITE, HIDDEN_GEM_MIN_RANK_JUMP, HIDDEN_GEM_MIN_SKILL,
    SENIORITY_RANK, SENIOR_WORDS, LEAD_WORDS, MID_WORDS, JUNIOR_WORDS,
    SKILL_FIELDS, SENIORITY_TEXT_FIELDS,
    LISTED_SKILL_FIELDS, NARRATIVE_FIELDS, STUFFING_PENALTY,
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


def _build_narrative_text(profile: dict) -> str:
    """Collect only narrative/prose fields (for stuffing detection)."""
    parts = []
    for f in NARRATIVE_FIELDS:
        val = profile.get(f)
        if not val:
            continue
        if isinstance(val, list):
            val = " ".join(str(v) for v in val)
        parts.append(str(val))
    return " ".join(parts)


def _build_listed_skills(profile: dict) -> list:
    """Extract skills that are explicitly listed in structured fields."""
    skills = []
    for f in LISTED_SKILL_FIELDS:
        val = profile.get(f)
        if not val:
            continue
        if isinstance(val, list):
            skills.extend(str(v) for v in val)
        elif isinstance(val, str):
            skills.extend(s.strip() for s in val.split(",") if s.strip())
    return skills


def _skill_score(candidate: dict, parsed_jd: dict) -> float:
    score, _ = _skill_score_with_evidence(candidate, parsed_jd)
    return score


def _skill_score_with_evidence(candidate: dict, parsed_jd: dict) -> Tuple[float, dict]:
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
    return 2


def _seniority_score(candidate: dict, jd_seniority: str) -> float:
    jd_level   = SENIORITY_RANK.get(jd_seniority.lower(), 2)
    cand_level = _infer_seniority(candidate)
    diff = cand_level - jd_level
    if diff >= 0:
        return 1.0
    if diff == -1:
        return 0.55
    return 0.15


# ---------------------------------------------------------------------------
# Signal 4: Activity / behavior
# ---------------------------------------------------------------------------

def _activity_score(candidate: dict) -> float:
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
# Keyword-stuffing detector
# ---------------------------------------------------------------------------

def detect_stuffing(candidate: dict, parsed_jd: dict) -> dict:
    """
    Detect résumé padding: skills that are listed in structured fields but
    never mentioned in any narrative/prose context.

    A skill that exists ONLY in the skills list with zero narrative support
    is flagged as 'claimed_unsupported'. The stuffing_ratio (0–1) drives a
    gentle penalty on skill_score — not a disqualification, just honest discounting.

    Returns:
        {
            "claimed_unsupported": ["Skill A", "Skill B"],
            "stuffing_ratio": 0.33
        }
    """
    listed    = _build_listed_skills(candidate)
    narrative = _build_narrative_text(candidate)

    if not listed:
        return {"claimed_unsupported": [], "stuffing_ratio": 0.0}

    unsupported = []
    for skill in listed:
        # Skill is "supported" if it appears anywhere in narrative prose
        if not _skill_match(skill, narrative):
            unsupported.append(skill)

    ratio = len(unsupported) / max(len(listed), 1)
    return {
        "claimed_unsupported": unsupported,
        "stuffing_ratio": round(ratio, 4),
    }


# ---------------------------------------------------------------------------
# Counterfactual explainer
# ---------------------------------------------------------------------------

def explain_why_not_higher(
    candidate: dict,
    parsed_jd: dict,
    current_rank: int,
    all_composites: list,  # sorted desc list of composites from the full batch
) -> dict:
    """
    Deterministic counterfactual: what rank would this candidate reach if they
    had the missing required skills? Cheap — uses the same scoring math, no LLM.

    Returns a human-readable ceiling explanation.
    """
    ev      = candidate.get("skill_evidence", {})
    missing = ev.get("required_missing", [])

    if not missing:
        return {
            "ceiling": "At skill ceiling — no required skills missing.",
            "missing_skills": [],
            "estimated_rank_gain": 0,
            "hypothetical_composite": candidate.get("composite_score", 0.0),
        }

    required = parsed_jd.get("required_skills", [])
    implied  = parsed_jd.get("implied_skills",  [])
    max_score = len(required) * 1.0 + len(implied) * 0.5 or 1.0

    req_hit  = len(ev.get("required_matched", []))
    impl_hit = len(ev.get("implied_matched",  []))

    # Hypothetical: ALL required skills present, same implied coverage
    hyp_skill = min((len(required) * 1.0 + impl_hit * 0.5) / max_score, 1.0)

    w_embed, w_skill, w_seniority, w_activity = get_weights()
    hyp_composite = round(
        w_embed     * float(candidate.get("embedding_score",  0)) +
        w_skill     * hyp_skill +
        w_seniority * float(candidate.get("seniority_score", 0)) +
        w_activity  * float(candidate.get("activity_score",  0)),
        4,
    )

    current_composite = float(candidate.get("composite_score", 0))
    rank_gain = sum(
        1 for s in all_composites
        if current_composite < s <= hyp_composite
    )
    estimated_rank = max(1, current_rank - rank_gain)

    skills_str = ", ".join(missing[:3])
    more       = f" (+{len(missing) - 3} more)" if len(missing) > 3 else ""
    ceiling    = (
        f"Would reach ~#{estimated_rank} with evidence of: {skills_str}{more}."
    )

    return {
        "ceiling":               ceiling,
        "missing_skills":        missing,
        "estimated_rank_gain":   rank_gain,
        "hypothetical_composite": hyp_composite,
    }


# ---------------------------------------------------------------------------
# Hidden gem: exported helper + rank-jump detection
# ---------------------------------------------------------------------------

def _is_hidden_gem(candidate: dict, composite: float, skill: float) -> bool:
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
    composite_rank = {c["_id"]: i for i, c in enumerate(scored)}

    by_embed   = sorted(scored, key=lambda c: c["embedding_score"], reverse=True)
    embed_rank = {c["_id"]: i for i, c in enumerate(by_embed)}

    for c in scored:
        cid  = c["_id"]
        jump = embed_rank[cid] - composite_rank[cid]
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

    Adds to each candidate:
      skill_score, seniority_score, activity_score, composite_score,
      skill_evidence, hidden_gem, _rank_jump,
      stuffing (keyword-stuffing analysis),
      counterfactual (why-not-higher explainer).
    """
    w_embed, w_skill, w_seniority, w_activity = get_weights()
    jd_seniority = parsed_jd.get("seniority", "mid")

    raw_embeds = [float(c.get("embedding_score", 0.0)) for c in candidates]
    e_min, e_max = min(raw_embeds), max(raw_embeds)
    e_range = (e_max - e_min) or 1.0

    scored = []
    for c, raw_e in zip(candidates, raw_embeds):
        embed              = (raw_e - e_min) / e_range
        skill, evidence    = _skill_score_with_evidence(c, parsed_jd)
        seniority          = _seniority_score(c, jd_seniority)
        activity           = _activity_score(c)

        # Stuffing detector: listed skills with no narrative support are penalised
        stuffing     = detect_stuffing(c, parsed_jd)
        skill_adj    = skill * (1.0 - STUFFING_PENALTY * stuffing["stuffing_ratio"])

        composite = (
            w_embed     * embed     +
            w_skill     * skill_adj +
            w_seniority * seniority +
            w_activity  * activity
        )

        result = dict(c)
        result["_id"]             = c.get("id") or c.get("candidate_id") or id(c)
        result["skill_score"]     = round(skill_adj, 4)
        result["skill_score_raw"] = round(skill,     4)  # pre-penalty, for transparency
        result["seniority_score"] = round(seniority, 4)
        result["activity_score"]  = round(activity,  4)
        result["composite_score"] = round(composite, 4)
        result["skill_evidence"]  = evidence
        result["stuffing"]        = stuffing
        result["hidden_gem"]      = False  # set by _detect_hidden_gems
        scored.append(result)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    scored = _detect_hidden_gems(scored)

    top = scored[:top_n]

    # Counterfactual: computed after ranking so we know the true position
    all_composites = [c["composite_score"] for c in scored]
    for rank, c in enumerate(top, 1):
        c["counterfactual"] = explain_why_not_higher(c, parsed_jd, rank, all_composites)

    return top


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
            "summary": "8 years building large-scale data pipelines using Python and Airflow on AWS.",
            "github_repos": 42, "embedding_score": 0.91,
        },
        {
            "id": "C002", "title": "Junior Frontend Developer",
            "skills": ["React", "TypeScript", "CSS"],
            "summary": "2 years building web UIs.",
            "github_repos": 8, "embedding_score": 0.31,
        },
        # Stuffed profile: claims many skills but summary mentions nothing
        {
            "id": "C003", "title": "Data Analyst",
            "skills": ["SQL", "Python", "Spark", "Airflow", "Kafka", "dbt"],
            "summary": "Experienced professional with various technical skills.",
            "embedding_score": 0.62,
        },
    ]

    PARSED_JD = {
        "required_skills": ["Python", "Apache Spark", "Apache Airflow", "AWS"],
        "implied_skills":  ["Git", "SQL", "Linux", "dbt"],
        "seniority":       "senior",
        "latent_needs":    ["owns pipelines end-to-end"],
    }

    results = score_candidates(CANDIDATES, PARSED_JD, top_n=5)

    print(f"{'#':<3} {'ID':<6} {'Title':<26} {'Score':>7} {'Stuff%':>7} {'Ceiling'}")
    print("-" * 85)
    for i, c in enumerate(results, 1):
        s   = c["stuffing"]
        cf  = c["counterfactual"]
        gem = " ★" if c["hidden_gem"] else ""
        print(
            f"{i:<3} {c['id']:<6} {c['title']:<26} "
            f"{c['composite_score']:>7.3f} "
            f"{s['stuffing_ratio']:>6.0%}  "
            f"{cf['ceiling'][:50]}{gem}"
        )
        if s["claimed_unsupported"]:
            print(f"     Unsupported claims: {s['claimed_unsupported']}")
