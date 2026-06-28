"""
data_loader.py — Flexible dataset loader for VIVEKA.

Accepts .json or .csv; auto-detects and normalises column names to the
internal profile schema; skips gracefully if any field is missing.

Internal schema (canonical names the pipeline expects):
  id, title, skills, summary, experience, education, certifications,
  github_repos, projects, endorsements

Column-name normalisation:
  headline / current_role / job_title / position / role -> title
  about / bio / description / overview / objective / ...  -> summary
  tech_skills / technologies / tools / technical_skills / ... -> skills
  work_history / work_experience / employment_history / ...   -> experience
  candidate_id / profile_id / applicant_id / user_id  -> id  (only if id absent)

NOT mapped: "name" is kept as "name" — never merged into "id".
"""

import csv
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional

log = logging.getLogger("viveka.data_loader")

# Maps raw column names (lowercase) -> canonical field name.
# Skills fields are handled specially: multiple fields are MERGED into one list.
_FIELD_MAP: dict[str, str] = {
    # title variants
    "headline":             "title",
    "current_role":         "title",
    "job_title":            "title",
    "position":             "title",
    "role":                 "title",
    "designation":          "title",
    "current_designation":  "title",

    # summary variants
    "about":                    "summary",
    "bio":                      "summary",
    "description":              "summary",
    "overview":                 "summary",
    "objective":                "summary",
    "profile_summary":          "summary",
    "professional_summary":     "summary",
    "introduction":             "summary",
    "profile_description":      "summary",
    "candidate_summary":        "summary",

    # skills fields — all merged together
    "tech_skills":          "skills",
    "technologies":         "skills",
    "tools":                "skills",
    "technical_skills":     "skills",
    "core_skills":          "skills",
    "key_skills":           "skills",
    "competencies":         "skills",
    "expertise":            "skills",
    "skill_set":            "skills",

    # experience / work history
    "work_history":             "experience",
    "work_experience":          "experience",
    "employment_history":       "experience",
    "professional_experience":  "experience",
    "career_history":           "experience",
    "job_history":              "experience",

    # id alternates (used only when "id" is missing)
    "candidate_id":     "_id_alt",
    "profile_id":       "_id_alt",
    "applicant_id":     "_id_alt",
    "user_id":          "_id_alt",
    "profile_number":   "_id_alt",

    # education
    "educational_background":   "education",
    "academic_background":      "education",
    "qualifications":           "education",
    "degrees":                  "education",
    "academic_qualification":   "education",

    # activity signals
    "repositories":             "github_repos",
    "github_repositories":      "github_repos",
    "project_count":            "projects",
    "recommendations":          "endorsements",
}

# Skill fields whose values get merged into one list
_SKILL_FIELD_KEYS = {
    "skills", "tech_skills", "technologies", "tools", "technical_skills",
    "core_skills", "key_skills", "competencies", "expertise", "skill_set",
}


def _coerce_skills(val) -> list:
    """Convert a skills value (list or comma-/semicolon-string) to a list."""
    if isinstance(val, list):
        return [str(s).strip() for s in val if str(s).strip()]
    if isinstance(val, str):
        sep = ";" if ";" in val else ","
        return [s.strip() for s in val.split(sep) if s.strip()]
    return []


def _normalise_profile(raw: dict) -> dict:
    """Apply field-name normalisation to a single raw profile dict."""
    result: dict = {}
    id_alt: Optional[str] = None
    skill_parts: list = []

    for raw_key, raw_val in raw.items():
        key = raw_key.strip().lower()
        canonical = _FIELD_MAP.get(key)

        if canonical == "_id_alt":
            if raw_val and not id_alt:
                id_alt = str(raw_val)
        elif key in _SKILL_FIELD_KEYS or canonical == "skills":
            # All skill fields are merged into one list
            skill_parts.extend(_coerce_skills(raw_val))
        elif canonical:
            # First writer wins (primary field takes precedence over synonym)
            if canonical not in result and raw_val not in ("", None, [], {}):
                result[canonical] = raw_val
        else:
            # Unknown field: keep as-is under lowercase original name
            lower_key = raw_key.lower()
            if lower_key not in result:
                result[lower_key] = raw_val

    # Resolve id: prefer explicit "id" field, then id_alt
    if not result.get("id") and id_alt:
        result["id"] = id_alt

    if not result.get("id"):
        # Synthetic id so downstream output never crashes
        result["id"] = f"C{abs(hash(str(raw))) % 100000:05d}"
        log.warning("Profile missing id — assigned synthetic id: %s", result["id"])

    # Merge all skill fragments into a de-duplicated list
    existing = _coerce_skills(result.get("skills", []))
    seen = {s.lower() for s in existing}
    for s in skill_parts:
        if s.lower() not in seen:
            existing.append(s)
            seen.add(s.lower())
    result["skills"] = existing

    return result


def load_candidates(path) -> List[Dict]:
    """
    Load candidate profiles from a .json or .csv file.

    Normalises column names to the internal schema.
    Returns a list of profile dicts ready for the pipeline.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = _load_json(path)
    elif suffix == ".csv":
        raw = _load_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Use .json or .csv.")

    profiles = [_normalise_profile(p) for p in raw]
    log.info("Loaded %d profiles from %s", len(profiles), path.name)
    return profiles


def _load_json(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("candidates", "profiles", "data", "results", "items"):
            if isinstance(data.get(key), list):
                return data[key]
        return [data]  # single profile
    raise ValueError(f"Unexpected JSON structure in {path}")


def _load_csv(path: Path) -> List[Dict]:
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def auto_find_dataset(data_dir: Path) -> Optional[Path]:
    """
    Scan data_dir for a dataset file, skipping known output files.
    Returns the first match (.json before .csv), or None.
    """
    skip_stems = {"ranked_output", "audit", "sample"}
    for suffix in (".json", ".csv"):
        for p in sorted(data_dir.glob(f"*{suffix}")):
            if not any(s in p.stem.lower() for s in skip_stems):
                return p
    # Fall back to sample if that's all we have
    for suffix in (".json", ".csv"):
        for p in sorted(data_dir.glob(f"*{suffix}")):
            if "ranked_output" not in p.stem.lower() and "audit" not in p.stem.lower():
                return p
    return None


# ---------------------------------------------------------------------------
# Smoke test — python src/data_loader.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample_candidates.json")
    profiles = load_candidates(path)
    print(f"Loaded {len(profiles)} profiles.")
    for p in profiles[:3]:
        print(f"  {p['id']} | {p.get('title', '?')} | skills: {p.get('skills', [])[:4]}")
