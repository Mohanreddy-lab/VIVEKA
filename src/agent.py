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

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))

log = logging.getLogger("viveka.agent")

from jd_parser   import parse_jd
from recall      import RecallEngine
from scoring     import score_candidates
from rerank      import rerank_candidates
from output      import write_output, print_summary, normalize_scores, validate_output
from pii         import redact_profile, is_firewall_on
from audit       import AuditLogger, is_audit_on
from config      import get_weights
from data_loader import load_candidates, auto_find_dataset


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def _cache_path(cache_dir: Path, key: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{key}.json"


def _cache_load(cache_dir: Path, key: str):
    p = _cache_path(cache_dir, key)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Cache corrupt for %s — ignoring. (%s)", key, exc)
    return None


def _cache_save(cache_dir: Path, key: str, data) -> None:
    try:
        _cache_path(cache_dir, key).write_text(
            json.dumps(data, default=str, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        log.warning("Could not save cache for %s: %s", key, exc)


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------

def load_profiles(data_dir: Path) -> List[Dict]:
    """
    Load candidate profiles using data_loader.
    Search order: profiles.json/csv first (explicit upload), then any other
    non-output dataset file, then sample_candidates.json fallback.
    """
    # 1. Explicit upload targets
    for fname in ("profiles.json", "profiles.csv"):
        p = data_dir / fname
        if p.exists():
            profiles = load_candidates(p)
            log.info("Loaded %d profiles from %s", len(profiles), fname)
            return profiles

    # 2. Auto-discover any dataset file in data/
    dataset = auto_find_dataset(data_dir)
    if dataset:
        profiles = load_candidates(dataset)
        log.info("Loaded %d profiles from %s", len(profiles), dataset.name)
        return profiles

    # 3. Fallback to built-in sample
    sample = data_dir / "sample_candidates.json"
    if sample.exists():
        profiles = load_candidates(sample)
        log.info("Loaded %d profiles from sample_candidates.json", len(profiles))
        return profiles

    log.warning("No dataset found in data/ — returning empty list.")
    return []


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

    # ── PII Firewall ─────────────────────────────────────────────────────────
    if is_firewall_on():
        profiles = [redact_profile(p) for p in profiles]
        log.info("PII firewall ON — identity fields stripped from %d profiles.", len(profiles))
    else:
        log.warning("PII firewall OFF — identity fields are visible to the model.")

    jd_hash       = _md5(jd_text)
    profiles_hash = _md5(json.dumps([p.get("id", i) for i, p in enumerate(profiles)]))
    recall_key    = f"recall_{jd_hash}_{profiles_hash}"
    scored_key    = f"scored_{jd_hash}_{profiles_hash}"

    # ── Stage 1: Parse JD ─────────────────────────────────────────────────────
    s1_key = f"parsed_jd_{jd_hash}"
    parsed_jd = None if force else _cache_load(cache_dir, s1_key)
    if parsed_jd:
        log.info("Stage 1: loaded parsed JD from cache.")
    else:
        log.info("Stage 1: parsing job description...")
        parsed_jd = parse_jd(jd_text)
        _cache_save(cache_dir, s1_key, parsed_jd)

    log.info("  Seniority: %s  Required: %s", parsed_jd["seniority"],
             ", ".join(parsed_jd["required_skills"][:6]))

    # ── Stage 2: Recall ───────────────────────────────────────────────────────
    recalled = None if force else _cache_load(cache_dir, recall_key)
    if recalled:
        log.info("Stage 2: loaded %d recalled candidates from cache.", len(recalled))
    else:
        log.info("Stage 2: building embedding index and recalling top candidates...")
        engine = RecallEngine()
        engine.index_candidates(profiles)
        recalled = engine.recall(parsed_jd, top_k=min(200, len(profiles)))
        _cache_save(cache_dir, recall_key, recalled)
        log.info("  Recalled %d candidates.", len(recalled))

    # ── Stage 3: Scoring ──────────────────────────────────────────────────────
    scored = None if force else _cache_load(cache_dir, scored_key)
    if scored:
        log.info("Stage 3: loaded %d scored candidates from cache.", len(scored))
    else:
        log.info("Stage 3: multi-signal scoring (synonym-aware)...")
        scored = score_candidates(recalled, parsed_jd, top_n=50)
        _cache_save(cache_dir, scored_key, scored)

    gems = sum(1 for c in scored if c.get("hidden_gem"))
    log.info("  Scored %d candidates.  Hidden gems found: %d", len(scored), gems)

    # ── Stage 4: LLM Rerank ───────────────────────────────────────────────────
    n = rerank_n or int(os.getenv("VIVEKA_RERANK_N", 50))
    rerank_key = f"ranked_{scored_key}_{n}"
    ranked = None if force else _cache_load(cache_dir, rerank_key)
    if ranked:
        log.info("Stage 4: loaded %d reranked candidates from cache.", len(ranked))
    else:
        log.info("Stage 4: honest LLM rerank (top %d)...", n)
        ranked = rerank_candidates(scored, parsed_jd, top_n=n)
        ranked = normalize_scores(ranked)
        _cache_save(cache_dir, rerank_key, ranked)
        log.info("  Reranked %d candidates.", len(ranked))

    # ── Stage 5: Output ───────────────────────────────────────────────────────
    log.info("Stage 5: writing output...")
    if out_dir:
        write_output(ranked, out_dir=out_dir)

    # ── Audit trail ───────────────────────────────────────────────────────────
    if out_dir and is_audit_on():
        model   = os.getenv("VIVEKA_MODEL", "llama3.2")
        weights = get_weights()
        logger  = AuditLogger(out_dir, jd_text, parsed_jd, model, weights)
        for rank, candidate in enumerate(ranked, 1):
            logger.log_candidate(candidate, rank)
        run_id = logger.flush()
        log.info("Audit trail written  run_id=%s", run_id)

    elapsed = time.time() - t0
    log.info("Done in %.1fs  (cache_dir: %s)", elapsed, cache_dir)
    return ranked


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    )

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
        log.info("No JD found — wrote sample to %s", jd_file)

    jd_text  = jd_file.read_text(encoding="utf-8").strip()
    profiles = load_profiles(data_dir)

    ranked = run_pipeline(
        jd_text  = jd_text,
        profiles = profiles,
        rerank_n = rerank_n,
        out_dir  = data_dir,
        force    = force,
    )
    # Add sequential rank field before validation
    for i, c in enumerate(ranked, 1):
        c["rank"] = i
    print_summary(ranked)
    validate_output(ranked)
