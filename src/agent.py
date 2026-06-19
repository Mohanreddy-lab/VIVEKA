"""
agent.py — Pillar 4: Agentic Recruiter

Orchestrates the full 5-stage MANTHAN pipeline:
  Stage 1  jd_parser  — parse the job description
  Stage 2  recall     — embed + FAISS top-200
  Stage 3  scoring    — multi-signal composite score
  Stage 4  rerank     — LLM honest re-score + reason
  Stage 5  output     — write CSV + JSON + print summary

Usage:
  python src/agent.py                          # uses data/job_description.txt
  python src/agent.py path/to/jd.txt          # custom JD file
  python src/agent.py path/to/jd.txt 10       # rerank only top 10 (quick test)

Data:
  Place candidate profiles at data/profiles.json or data/profiles.csv
  Place the job description at data/job_description.txt
  (both paths are overridable via command-line arguments)
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

# Allow running as `python src/agent.py` from the project root
sys.path.insert(0, os.path.dirname(__file__))

from jd_parser import parse_jd
from recall    import RecallEngine
from scoring   import score_candidates
from rerank    import rerank_candidates
from output    import write_output, print_summary


# ---------------------------------------------------------------------------
# Data loader — flexible, adapts to JSON or CSV
# ---------------------------------------------------------------------------

def _validate_profiles(profiles: List[Dict]) -> None:
    """Warn if profiles are missing IDs or if IDs are duplicated."""
    id_fields = ["candidate_id", "id", "profile_id", "applicant_id"]
    missing, seen, dupes = 0, set(), set()
    for p in profiles:
        pid = next((str(p[f]) for f in id_fields if p.get(f)), None)
        if pid is None:
            missing += 1
        elif pid in seen:
            dupes.add(pid)
        else:
            seen.add(pid)
    if missing:
        print(f"[agent] Warning: {missing} profiles have no ID field — output will show 'UNKNOWN'.")
    if dupes:
        print(f"[agent] Warning: duplicate IDs found: {dupes} — ranked CSV will have duplicate rows.")


def load_profiles(data_dir: Path) -> List[Dict]:
    """
    Load candidate profiles from data_dir.
    Tries profiles.json first, then profiles.csv.
    Falls back to built-in demo profiles so the pipeline always runs.
    """
    json_path = data_dir / "profiles.json"
    csv_path  = data_dir / "profiles.csv"

    if json_path.exists():
        profiles = json.loads(json_path.read_text(encoding="utf-8"))
        print(f"[agent] Loaded {len(profiles)} profiles from profiles.json")
        _validate_profiles(profiles)
        return profiles

    if csv_path.exists():
        import csv
        with csv_path.open(encoding="utf-8") as f:
            profiles = list(csv.DictReader(f))
        print(f"[agent] Loaded {len(profiles)} profiles from profiles.csv")
        _validate_profiles(profiles)
        return profiles

    print("[agent] No profiles.json or profiles.csv found — using built-in demo profiles.")
    profiles = _demo_profiles()

    _validate_profiles(profiles)
    return profiles


def _demo_profiles() -> List[Dict]:
    return [
        {
            "id": "C001", "title": "Senior Data Engineer",
            "skills": ["Python", "Apache Spark", "Airflow", "AWS", "dbt", "SQL", "Redshift"],
            "summary": "8 years building large-scale data pipelines. Led platform migrations on AWS. Owns pipelines end-to-end.",
            "github_repos": 42,
        },
        {
            "id": "C002", "title": "Junior Frontend Developer",
            "skills": ["React", "TypeScript", "CSS", "Figma"],
            "summary": "2 years building web UIs. Passionate about accessibility and design systems.",
            "github_repos": 8,
        },
        {
            "id": "C003", "title": "Data Analyst",
            "skills": ["SQL", "Python", "Tableau", "Excel"],
            "summary": "Analyst at a fintech. Strong SQL and storytelling with data. No pipeline experience.",
        },
        {
            "id": "C004", "title": "Data Engineer",
            "skills": ["Spark", "Airflow", "Python", "S3"],
            "summary": "Built batch pipelines on AWS. Self-taught, no degree. Ships fast, works well under ambiguity.",
        },
        {
            "id": "C005", "title": "ML Engineer",
            "skills": ["Python", "PyTorch", "Spark", "Kubernetes", "MLflow", "feature stores"],
            "summary": "Deploys ML models at scale. Experience with batch inference and feature stores on AWS.",
            "github_repos": 25,
        },
        {
            "id": "C006", "title": "Backend Engineer",
            "skills": ["Java", "Kafka", "PostgreSQL", "Docker", "AWS"],
            "summary": "5 years in backend services. Built Kafka consumers for real-time data ingestion.",
        },
        {
            "id": "C007", "title": "Analytics Engineer",
            "skills": ["dbt", "SQL", "Snowflake", "Python", "Airflow"],
            "summary": "Owns the analytics layer. Built dbt models and Airflow DAGs for a 100M-row warehouse.",
        },
        {
            "id": "C008", "title": "DevOps Engineer",
            "skills": ["Kubernetes", "Terraform", "AWS", "CI/CD", "Docker"],
            "summary": "Platform engineer focused on infrastructure reliability and deployment automation.",
        },
    ]


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    jd_text:    str,
    profiles:   List[Dict],
    rerank_n:   int = None,
    out_dir:    Path = None,
) -> List[Dict]:
    """
    Run all 5 stages and return the final ranked list.

    Args:
        jd_text:   Raw job description text.
        profiles:  List of candidate profile dicts.
        rerank_n:  How many candidates to send to the LLM (default: env var or 50).
        out_dir:   Where to write the output files.

    Returns:
        Final ranked list of candidate dicts with all scoring fields attached.
    """
    t0 = time.time()

    # Stage 1: Parse JD
    print("\n[Stage 1] Parsing job description...")
    parsed_jd = parse_jd(jd_text)
    print(f"  Seniority : {parsed_jd['seniority']}")
    print(f"  Required  : {', '.join(parsed_jd['required_skills'][:5])}")
    print(f"  Implied   : {', '.join(parsed_jd['implied_skills'][:4])}")

    # Stage 2: Fast recall
    print("\n[Stage 2] Building embedding index and recalling top candidates...")
    engine = RecallEngine()
    engine.index_candidates(profiles)
    recalled = engine.recall(parsed_jd, top_k=min(200, len(profiles)))
    print(f"  Recalled {len(recalled)} candidates.")

    # Stage 3: Multi-signal scoring
    print("\n[Stage 3] Multi-signal scoring...")
    scored = score_candidates(recalled, parsed_jd, top_n=50)
    gems_after_scoring = sum(1 for c in scored if c.get("hidden_gem"))
    print(f"  Scored {len(scored)} candidates. Hidden gems so far: {gems_after_scoring}")

    # Stage 4: LLM rerank
    print("\n[Stage 4] Honest LLM rerank...")
    n = rerank_n or int(os.getenv("MANTHAN_RERANK_N", 50))
    ranked = rerank_candidates(scored, parsed_jd, top_n=n)
    print(f"  Reranked {len(ranked)} candidates.")

    # Stage 5: Write output
    print("\n[Stage 5] Writing output...")
    if out_dir:
        write_output(ranked, out_dir=out_dir)

    elapsed = time.time() - t0
    print(f"\n[agent] Pipeline complete in {elapsed:.1f}s")
    return ranked


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = Path(__file__).parent.parent / "data"

    # JD file (optional positional arg)
    jd_file = Path(sys.argv[1]) if len(sys.argv) > 1 else data_dir / "job_description.txt"

    # Optional rerank cap (quick test: python src/agent.py jd.txt 5)
    rerank_n = int(sys.argv[2]) if len(sys.argv) > 2 else None

    # Write sample JD if none exists
    if not jd_file.exists():
        sample_jd = """Senior Data Engineer — Platform Team

We are looking for a Senior Data Engineer to join our growing data platform team.

Requirements:
- 5+ years of experience in data engineering
- Strong Python and SQL skills
- Experience with Apache Spark and Airflow
- Hands-on with AWS (S3, Glue, Redshift)
- Experience with dbt for data transformation

Nice to have:
- Kafka or other streaming technologies
- MLflow or feature store experience

Responsibilities: mentor junior engineers, own data model design,
work closely with ML and product teams. We move fast — ambiguity is normal."""
        jd_file.write_text(sample_jd, encoding="utf-8")
        print(f"[agent] No JD found — wrote sample to {jd_file}\n")

    jd_text  = jd_file.read_text(encoding="utf-8").strip()
    profiles = load_profiles(data_dir)

    ranked = run_pipeline(
        jd_text  = jd_text,
        profiles = profiles,
        rerank_n = rerank_n,
        out_dir  = data_dir,
    )

    print_summary(ranked)
