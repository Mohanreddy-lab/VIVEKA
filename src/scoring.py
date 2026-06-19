"""
scoring.py — Stage 3: Multi-signal Scoring

Takes the top-200 candidates from Stage 2 (each with an embedding_score)
and blends three signals into a single composite score:

  1. embedding_score  — semantic similarity from FAISS (Stage 2)
  2. skill_score      — weighted skill overlap with the JD
  3. activity_score   — behavioral signals (GitHub, projects, endorsements, etc.)
                        Returns 0.5 as a neutral default until the real dataset
                        schema is known; plug real fields in one place below.

Weights are read from environment variables so you can tune without editing code:
  MANTHAN_W_EMBED     (default 0.40)
  MANTHAN_W_SKILL     (default 0.40)
  MANTHAN_W_ACTIVITY  (default 0.20)

Returns the top-50 candidates sorted by composite_score descending.
"""

import os
import re
from typing import List, Dict


# ---------------------------------------------------------------------------
# Blend weights — tune via env vars, defaults sum to 1.0
# ---------------------------------------------------------------------------

def _get_weights() -> tuple:
    w_embed    = float(os.getenv("MANTHAN_W_EMBED",    0.40))
    w_skill    = float(os.getenv("MANTHAN_W_SKILL",    0.40))
    w_activity = float(os.getenv("MANTHAN_W_ACTIVITY", 0.20))
    total = w_embed + w_skill + w_activity
    # Normalise so they always sum to 1, even if env vars don't
    return w_embed / total, w_skill / total, w_activity / total


# ---------------------------------------------------------------------------
# Signal 2: Skill overlap
# Required skills are worth 1.0 each; implied skills 0.5 each.
# Matching is case-insensitive substring — "Spark" hits "Apache Spark".
# ---------------------------------------------------------------------------

def _skill_score(candidate: dict, parsed_jd: dict) -> float:
    """Return a 0–1 score for how well the candidate's skills cover the JD."""
    required = parsed_jd.get("required_skills", [])
    implied  = parsed_jd.get("implied_skills",  [])

    if not required and not implied:
        return 0.0  # no skill data in JD — no signal

    # Build a single lowercased string of the candidate's skills/text to search
    candidate_text = _candidate_skill_text(candidate).lower()

    max_score = len(required) * 1.0 + len(implied) * 0.5
    if max_score == 0:
        return 0.0

    earned = 0.0
    for skill in required:
        if _skill_match(skill, candidate_text):
            earned += 1.0
    for skill in implied:
        if _skill_match(skill, candidate_text):
            earned += 0.5

    return min(earned / max_score, 1.0)


def _skill_match(skill: str, candidate_text: str) -> bool:
    """
    True if the skill appears as a standalone term in the candidate text.

    Uses negative lookaround — not \b — so it handles special-char skills
    (C++, .NET, R) correctly while still preventing "SQL" from hitting "NoSQL".

      "SQL"  in "NoSQL expert"          → False  ✓
      "SQL"  in "strong SQL background" → True   ✓
      "C++"  in "C++ and Python"        → True   ✓
      ".NET" in ".NET developer"        → True   ✓
    """
    pattern = r"(?<!\w)" + re.escape(skill.lower()) + r"(?!\w)"
    return bool(re.search(pattern, candidate_text))


def _candidate_skill_text(profile: dict) -> str:
    """Collect all skill-related fields from the profile into one string."""
    skill_fields = ["skills", "tech_skills", "tools", "certifications",
                    "title", "headline", "summary", "bio", "about",
                    "experience", "work_history"]
    parts = []
    for field in skill_fields:
        val = profile.get(field)
        if not val:
            continue
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        parts.append(str(val))
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Signal 3: Activity / behavior
# Plug in real fields from the dataset here — one function to change.
# Returns 0.5 (neutral) if no behavioral data is present.
# ---------------------------------------------------------------------------

def _activity_score(candidate: dict) -> float:
    """
    Score behavioral signals 0–1.
    Add real field names from the dataset when schema is known.
    Currently reads: github_repos, projects, project_count, endorsements, recommendations.
    Returns 0.0 when no activity data exists — does not inflate with a fake neutral.
    """
    signals = []

    # GitHub / open-source activity
    repos = candidate.get("github_repos") or candidate.get("repositories")
    if repos is not None:
        signals.append(min(int(repos) / 30, 1.0))  # 30 repos → full score

    # Portfolio projects
    projects = candidate.get("projects") or candidate.get("project_count")
    if projects is not None:
        count = len(projects) if isinstance(projects, list) else int(projects)
        signals.append(min(count / 10, 1.0))  # 10 projects → full score

    # Endorsements / recommendations
    endorsements = candidate.get("endorsements") or candidate.get("recommendations")
    if endorsements is not None:
        count = len(endorsements) if isinstance(endorsements, list) else int(endorsements)
        signals.append(min(count / 20, 1.0))  # 20 endorsements → full score

    # No signals → 0.0. A fake 0.5 neutral inflates everyone equally and adds no signal.
    return round(sum(signals) / len(signals), 4) if signals else 0.0


# ---------------------------------------------------------------------------
# Hidden gem detection
# High composite score but thin/unassuming profile → flag for human review.
# ---------------------------------------------------------------------------

def _is_hidden_gem(candidate: dict, composite: float, skill: float) -> bool:
    """
    Flag when a candidate scores well on substance but looks unassuming on paper.
    Threshold is tunable via MANTHAN_GEM_THRESHOLD env var (default 0.55).
    """
    threshold = float(os.getenv("MANTHAN_GEM_THRESHOLD", 0.55))
    if composite < threshold:
        return False

    # Sparse skills list despite high score
    skills_list = candidate.get("skills") or candidate.get("tech_skills") or []
    sparse_skills = isinstance(skills_list, list) and len(skills_list) < 4

    # Title doesn't scream 'senior' but score is high
    title = (candidate.get("title") or candidate.get("headline") or "").lower()
    modest_title = not any(word in title for word in
                           ["senior", "lead", "principal", "staff", "head", "architect", "manager"])

    return (sparse_skills or modest_title) and skill >= 0.50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score_candidates(
    candidates: List[Dict],
    parsed_jd: dict,
    top_n: int = 50,
) -> List[Dict]:
    """
    Score and rank a list of candidates (output of Stage 2 recall).

    Args:
        candidates:  List of profile dicts, each with an 'embedding_score' key.
        parsed_jd:   Structured JD dict from Stage 1.
        top_n:       How many top candidates to return (default 50 for Stage 4).

    Returns:
        List of dicts sorted by composite_score descending, each with added keys:
          skill_score, activity_score, composite_score, hidden_gem (bool).
    """
    w_embed, w_skill, w_activity = _get_weights()

    # Collect raw embed scores to normalize across this batch.
    # FAISS cosine scores cluster in [0.6, 0.9]; normalizing spreads them
    # so the embed signal doesn't get compressed into a tiny range.
    raw_embeds = [float(c.get("embedding_score", 0.0)) for c in candidates]
    e_min, e_max = min(raw_embeds), max(raw_embeds)
    e_range = e_max - e_min if e_max > e_min else 1.0

    scored = []
    for c, raw_e in zip(candidates, raw_embeds):
        embed    = (raw_e - e_min) / e_range   # stretch to [0, 1]
        skill    = _skill_score(c, parsed_jd)
        activity = _activity_score(c)
        composite = (
            w_embed    * embed +
            w_skill    * skill +
            w_activity * activity
        )

        result = dict(c)
        result["skill_score"]     = round(skill,     4)
        result["activity_score"]  = round(activity,  4)
        result["composite_score"] = round(composite, 4)
        result["hidden_gem"]      = _is_hidden_gem(c, composite, skill)
        scored.append(result)

    scored.sort(key=lambda x: x["composite_score"], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# Smoke test — run with: python src/scoring.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))

    # Simulate Stage 2 output (embedding_score already attached)
    CANDIDATES = [
        {
            "id": "C001", "title": "Senior Data Engineer",
            "skills": ["Python", "Apache Spark", "Airflow", "AWS", "dbt", "SQL"],
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
            "summary": "Strong SQL and data storytelling at a fintech.",
            "embedding_score": 0.62,
        },
        {
            "id": "C004",
            # No senior title, sparse profile — but skills hit hard: hidden gem candidate
            "title": "Data Engineer",
            "skills": ["Spark", "Airflow", "Python"],
            "summary": "Built batch pipelines on AWS. Self-taught, no degree.",
            "embedding_score": 0.78,
        },
        {
            "id": "C005", "title": "ML Engineer",
            "skills": ["Python", "PyTorch", "Spark", "MLflow"],
            "summary": "Deploys ML models at scale.",
            "github_repos": 25, "embedding_score": 0.70,
        },
    ]

    PARSED_JD = {
        "required_skills": ["Python", "Apache Spark", "Airflow", "AWS"],
        "implied_skills":  ["Git", "SQL", "Linux", "dbt"],
        "seniority":       "senior",
        "latent_needs":    ["owns pipelines end-to-end", "works under ambiguity"],
    }

    results = score_candidates(CANDIDATES, PARSED_JD, top_n=5)

    print(f"{'Rank':<5} {'ID':<6} {'Title':<28} {'Embed':>6} {'Skill':>6} {'Act':>5} {'Score':>7} {'Gem'}")
    print("-" * 72)
    for i, c in enumerate(results, 1):
        gem = "★" if c["hidden_gem"] else ""
        print(
            f"{i:<5} {c['id']:<6} {c['title']:<28} "
            f"{c['embedding_score']:>6.3f} {c['skill_score']:>6.3f} "
            f"{c['activity_score']:>5.3f} {c['composite_score']:>7.3f}  {gem}"
        )
