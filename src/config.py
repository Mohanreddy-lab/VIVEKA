"""config.py — All magic numbers and shared field lists for MANTHAN.

Single source of truth — change a constant here and every module picks it up.
"""
import os


# ---------------------------------------------------------------------------
# Scoring weights  (always auto-normalised so they can be tuned freely)
# ---------------------------------------------------------------------------

def get_weights() -> tuple[float, float, float, float]:
    """Return (embed, skill, seniority, activity) weights, normalised to sum=1."""
    w_embed     = float(os.getenv("MANTHAN_W_EMBED",      0.30))
    w_skill     = float(os.getenv("MANTHAN_W_SKILL",      0.40))
    w_seniority = float(os.getenv("MANTHAN_W_SENIORITY",  0.15))
    w_activity  = float(os.getenv("MANTHAN_W_ACTIVITY",   0.15))
    total = w_embed + w_skill + w_seniority + w_activity or 1.0
    return (w_embed/total, w_skill/total, w_seniority/total, w_activity/total)


# ---------------------------------------------------------------------------
# Activity score caps
# ---------------------------------------------------------------------------

MAX_GITHUB_REPOS  = 30
MAX_PROJECTS      = 10
MAX_ENDORSEMENTS  = 20

# ---------------------------------------------------------------------------
# Hidden gem detection
# ---------------------------------------------------------------------------

HIDDEN_GEM_MIN_COMPOSITE = 0.40   # minimum composite to be considered
HIDDEN_GEM_MIN_RANK_JUMP = 2      # must rise ≥ N positions vs raw embed rank
HIDDEN_GEM_MIN_SKILL     = 0.40   # minimum skill score to flag as gem

# ---------------------------------------------------------------------------
# Seniority
# ---------------------------------------------------------------------------

SENIORITY_RANK: dict[str, int] = {"junior": 1, "mid": 2, "senior": 3, "lead": 4}

SENIOR_WORDS = {"senior", "sr", "staff", "principal", "architect", "distinguished"}
LEAD_WORDS   = {"lead", "head", "director", "vp", "chief", "manager"}
MID_WORDS    = {"ii", "iii", "intermediate"}
JUNIOR_WORDS = {"junior", "jr", "entry", "associate", "intern", "trainee", "graduate"}

# ---------------------------------------------------------------------------
# Pipeline defaults  (all overridable via env vars)
# ---------------------------------------------------------------------------

RECALL_TOP_K  = int(os.getenv("MANTHAN_RECALL_K",   200))
SCORE_TOP_N   = int(os.getenv("MANTHAN_SCORE_N",     50))
RERANK_N      = int(os.getenv("MANTHAN_RERANK_N",    50))
PROFILE_CHARS = int(float(os.getenv("MANTHAN_PROFILE_CHARS", 900)))

BLEND_COMPOSITE = float(os.getenv("MANTHAN_BLEND_COMPOSITE", 0.5))
BLEND_LLM       = float(os.getenv("MANTHAN_BLEND_LLM",       0.5))

# ---------------------------------------------------------------------------
# Field name priority lists  (shared by recall, scoring, rerank, output)
# ---------------------------------------------------------------------------

SKILL_FIELDS: list[str] = [
    "skills", "tech_skills", "tools", "certifications",
    "title", "headline", "summary", "bio", "about",
    "experience", "work_history",
]

PROFILE_EMBED_FIELDS: list[str] = [
    "title", "headline", "current_role",
    "skills", "tech_skills", "tools",
    "summary", "bio", "about",
    "experience", "work_history",
    "education", "certifications",
]

RERANK_PROFILE_FIELDS: list[str] = [
    "title", "headline", "current_role",
    "summary", "bio", "about",
    "skills", "tech_skills", "tools",
    "experience", "work_history",
    "education", "certifications",
    "github_repos", "projects", "endorsements",
]

ID_FIELDS: list[str] = [
    "candidate_id", "id", "profile_id", "applicant_id", "email",
]

SENIORITY_TEXT_FIELDS: list[str] = [
    "title", "headline", "current_role", "experience",
]

# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

EMBED_MODEL = os.getenv("MANTHAN_EMBED_MODEL", "all-MiniLM-L6-v2")
