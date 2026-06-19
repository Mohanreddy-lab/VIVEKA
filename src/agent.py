"""
agent.py — Pillar 4: Agentic Recruiter + Orchestrator

Runs the full 5-stage pipeline with hash-based stage caching so
expensive stages (embedding, LLM rerank) are not repeated when
only one stage's inputs have changed.

Cache lives in data/.cache/. Delete it or pass --force to re-run everything.

Usage:
  python src/agent.py                           # auto-finds data/job_description.txt
  python src/agent.py path/to/jd.txt           # custom JD
  python src/agent.py path/to/jd.txt 10        # rerank only top 10 (fast test)
  python src/agent.py path/to/jd.txt 0 --force # bypass cache entirely
"""

import csv
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))

from jd_parser import parse_jd
from recall    import RecallEngine
from scoring   import score_candidates
from rerank    import rerank_candidates
from output    import write_output, print_summary, normalize_scores


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _cache_path(cache_dir: Path, key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.pkl"


def _cache_load(cache_dir: Path, key: str):
    p = _cache_path(cache_dir, key)
    if p.exists():
        with p.open("rb") as f:
            return pickle.load(f)
    return None


def _cache_save(cache_dir: Path, key: str, data) -> None:
    with _cache_path(cache_dir, key).open("wb") as f:
        pickle.dump(data, f)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_profiles(data_dir: Path) -> List[Dict]:
    """Load profiles from JSON or CSV; fall back to built-in demo profiles."""
    json_path = data_dir / "profiles.json"
    csv_path  = data_dir / "profiles.csv"

    if json_path.exists():
        profiles = json.loads(json_path.read_text(encoding="utf-8"))
        print(f"[agent] Loaded {len(profiles)} profiles from profiles.json")
        return profiles

    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as f:
            profiles = list(csv.DictReader(f))
        print(f"[agent] Loaded {len(profiles)} profiles from profiles.csv")
        return profiles

    print("[agent] No profiles found in data/ — using built-in demo profiles.")
    return _demo_profiles()


def _demo_profiles() -> List[Dict]:
    return [
        {
            "id": "C001", "title": "Senior Data Engineer",
            "skills": ["Python", "Apache Spark", "Airflow", "AWS", "dbt", "SQL", "Redshift"],
            "summary": "8 years building large-scale data pipelines. Led platform migration from Hadoop to S3+Glue+Redshift. Owns pipelines end-to-end.",
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
# Pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    jd_text:   str,
    profiles:  List[Dict],
    rerank_n:  int  = None,
    out_dir:   Path = None,
    force:     bool = False,
) -> List[Dict]:
    """
    Run all 5 stages with caching. Returns the final ranked list.

    Cache keys:
      Stage 1: hash of jd_text
      Stage 2: hash of jd_text + all profile IDs
      Stage 3: same as Stage 2
      Stage 4: hash of Stage 3 top-50 + rerank_n
    """
    t0        = time.time()
    cache_dir = (out_dir or Path("data")) / ".cache"

    jd_hash       = _md5(jd_text)
    profiles_hash = _md5(json.dumps([p.get("id", i) for i, p in enumerate(profiles)]))
    recall_key    = f"recall_{jd_hash}_{profiles_hash}"
    scored_key    = f"scored_{jd_hash}_{profiles_hash}"

    # ── Stage 1: Parse JD ─────────────────────────────────────────────────────
    s1_key = f"parsed_jd_{jd_hash}"
    parsed_jd = None if force else _cache_load(cache_dir, s1_key)
    if parsed_jd:
        print(f"[Stage 1] Loaded parsed JD from cache.")
    else:
        print(f"\n[Stage 1] Parsing job description...")
        parsed_jd = parse_jd(jd_text)
        _cache_save(cache_dir, s1_key, parsed_jd)

    print(f"  Seniority  : {parsed_jd['seniority']}")
    print(f"  Required   : {', '.join(parsed_jd['required_skills'][:6])}")
    print(f"  Implied    : {', '.join(parsed_jd['implied_skills'][:4])}")

    # ── Stage 2: Recall ───────────────────────────────────────────────────────
    recalled = None if force else _cache_load(cache_dir, recall_key)
    if recalled:
        print(f"\n[Stage 2] Loaded {len(recalled)} recalled candidates from cache.")
    else:
        print(f"\n[Stage 2] Building embedding index and recalling top candidates...")
        engine = RecallEngine()
        engine.index_candidates(profiles)
        recalled = engine.recall(parsed_jd, top_k=min(200, len(profiles)))
        _cache_save(cache_dir, recall_key, recalled)
        print(f"  Recalled {len(recalled)} candidates.")

    # ── Stage 3: Scoring ──────────────────────────────────────────────────────
    scored = None if force else _cache_load(cache_dir, scored_key)
    if scored:
        print(f"\n[Stage 3] Loaded {len(scored)} scored candidates from cache.")
    else:
        print(f"\n[Stage 3] Multi-signal scoring (synonym-aware)...")
        scored = score_candidates(recalled, parsed_jd, top_n=50)
        _cache_save(cache_dir, scored_key, scored)

    gems = sum(1 for c in scored if c.get("hidden_gem"))
    print(f"  Scored {len(scored)} candidates.  Hidden gems found: {gems}")

    # ── Stage 4: LLM Rerank ───────────────────────────────────────────────────
    n = rerank_n or int(os.getenv("MANTHAN_RERANK_N", 50))
    rerank_key = f"ranked_{scored_key}_{n}"
    ranked = None if force else _cache_load(cache_dir, rerank_key)
    if ranked:
        print(f"\n[Stage 4] Loaded {len(ranked)} reranked candidates from cache.")
    else:
        print(f"\n[Stage 4] Honest LLM rerank (chain-of-thought, top {n})...")
        ranked = rerank_candidates(scored, parsed_jd, top_n=n)
        ranked = normalize_scores(ranked)
        _cache_save(cache_dir, rerank_key, ranked)
        print(f"  Reranked {len(ranked)} candidates.")

    # ── Stage 5: Output ───────────────────────────────────────────────────────
    print(f"\n[Stage 5] Writing output...")
    if out_dir:
        write_output(ranked, out_dir=out_dir)

    elapsed = time.time() - t0
    print(f"\n[agent] Done in {elapsed:.1f}s  (cache_dir: {cache_dir})")
    return ranked


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir  = Path(__file__).parent.parent / "data"
    jd_file   = Path(sys.argv[1]) if len(sys.argv) > 1 else data_dir / "job_description.txt"
    rerank_n  = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
    force     = "--force" in sys.argv

    if not jd_file.exists():
        sample = """Senior Data Engineer — Platform Team

We are looking for a Senior Data Engineer to join our data platform team.

Requirements:
- 5+ years data engineering experience
- Strong Python and SQL
- Apache Spark and Airflow
- AWS (S3, Glue, Redshift)
- dbt experience

Nice to have: Kafka, MLflow, feature stores.

Responsibilities: own pipelines end-to-end, mentor junior engineers,
work with ML and product teams. Ambiguity is normal here."""
        jd_file.write_text(sample, encoding="utf-8")
        print(f"[agent] No JD found — wrote sample to {jd_file}\n")

    jd_text  = jd_file.read_text(encoding="utf-8").strip()
    profiles = load_profiles(data_dir)

    ranked = run_pipeline(
        jd_text  = jd_text,
        profiles = profiles,
        rerank_n = rerank_n,
        out_dir  = data_dir,
        force    = force,
    )
    print_summary(ranked)
