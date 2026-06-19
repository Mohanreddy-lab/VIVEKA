"""
output.py — Stage 5: Output Writer

Writes the final ranked shortlist in two formats:
  - ranked_output.csv   primary deliverable; organizers' format
  - ranked_output.json  full data for the Streamlit demo

Columns written (adapt to exact organizers' spec when dataset arrives):
  rank | candidate_id | final_score | llm_score | confidence |
  hidden_gem | reason | skill_score | composite_score
"""

import csv
import json
from pathlib import Path
from typing import List, Dict


# Fields to write to CSV — reorder / add columns to match organizers' spec
CSV_COLUMNS = [
    "rank",
    "candidate_id",
    "final_score",
    "llm_score",
    "confidence",
    "hidden_gem",
    "reason",
    "skill_score",
    "composite_score",
    "embedding_score",
]

# Field names the dataset might use for the candidate's ID
ID_FIELD_CANDIDATES = ["candidate_id", "id", "profile_id", "applicant_id", "email"]


def _get_id(candidate: dict) -> str:
    """Extract the candidate's unique ID, trying common field names."""
    for field in ID_FIELD_CANDIDATES:
        val = candidate.get(field)
        if val:
            return str(val)
    return "UNKNOWN"


def _to_row(rank: int, candidate: dict) -> dict:
    """Flatten one candidate into a CSV row dict."""
    return {
        "rank":             rank,
        "candidate_id":     _get_id(candidate),
        "final_score":      round(float(candidate.get("final_score",      0.0)), 4),
        "llm_score":        candidate.get("llm_score",      "N/A"),
        "confidence":       candidate.get("confidence",     "N/A"),
        "hidden_gem":       "yes" if candidate.get("hidden_gem") else "no",
        "reason":           candidate.get("reason",         ""),
        "skill_score":      round(float(candidate.get("skill_score",      0.0)), 4),
        "composite_score":  round(float(candidate.get("composite_score",  0.0)), 4),
        "embedding_score":  round(float(candidate.get("embedding_score",  0.0)), 4),
    }


def write_output(
    ranked: List[Dict],
    out_dir: str | Path = "data",
    stem: str = "ranked_output",
) -> tuple[Path, Path]:
    """
    Write the ranked shortlist to CSV and JSON.

    Args:
        ranked:   Sorted list of candidates from Stage 4 (rerank.py).
        out_dir:  Directory to write into (default: data/).
        stem:     Base filename without extension.

    Returns:
        (csv_path, json_path) — paths to the two output files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"

    rows = [_to_row(rank, c) for rank, c in enumerate(ranked, start=1)]

    # CSV
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # JSON (full candidate data, not just the CSV columns)
    enriched = []
    for rank, c in enumerate(ranked, start=1):
        entry = dict(c)
        entry["rank"] = rank
        entry["candidate_id"] = _get_id(c)
        enriched.append(entry)

    json_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"[output] Wrote {len(rows)} candidates -> {csv_path}")
    print(f"[output] Wrote full data   -> {json_path}")
    return csv_path, json_path


def print_summary(ranked: List[Dict], top_n: int = 10) -> None:
    """Print a quick leaderboard to stdout."""
    gems   = [c for c in ranked if c.get("hidden_gem")]
    top    = ranked[:top_n]

    print(f"\n{'='*70}")
    print(f"  MANTHAN — Final Shortlist  ({len(ranked)} candidates ranked)")
    print(f"{'='*70}")
    print(f"{'#':<4} {'ID':<14} {'Score':>6} {'LLM':>4} {'Conf':<8} {'Gem':<4} Reason")
    print(f"{'-'*70}")
    for i, c in enumerate(top, 1):
        gem    = "★" if c.get("hidden_gem") else ""
        cid    = _get_id(c)[:13]
        reason = c.get("reason", "")[:45]
        score  = c.get("final_score", 0.0)
        llm    = c.get("llm_score", "-")
        conf   = c.get("confidence", "-")[:7]
        print(f"{i:<4} {cid:<14} {score:>6.3f} {str(llm):>4} {conf:<8} {gem:<4} {reason}")

    if gems:
        print(f"\n  ★ Hidden Gems ({len(gems)} found):")
        for c in gems:
            print(f"    [{_get_id(c)}] {c.get('title', '')} — {c.get('reason', '')[:60]}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Smoke test — run with: python src/output.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    SAMPLE = [
        {
            "id": "C001", "title": "Senior Data Engineer",
            "final_score": 0.838, "llm_score": 9, "confidence": "high",
            "hidden_gem": False, "reason": "8 years on AWS Spark pipelines; direct match.",
            "skill_score": 0.75, "composite_score": 0.727, "embedding_score": 0.91,
        },
        {
            "id": "C004", "title": "Data Engineer",
            "final_score": 0.682, "llm_score": 7, "confidence": "medium",
            "hidden_gem": True,
            "reason": "Limited evidence: self-taught, no degree, but ships Spark + Airflow on AWS.",
            "skill_score": 0.50, "composite_score": 0.614, "embedding_score": 0.78,
        },
        {
            "id": "C003", "title": "Data Analyst",
            "final_score": 0.418, "llm_score": 4, "confidence": "low",
            "hidden_gem": False, "reason": "Limited evidence: no pipeline experience stated.",
            "skill_score": 0.25, "composite_score": 0.435, "embedding_score": 0.62,
        },
    ]

    data_dir = Path(__file__).parent.parent / "data"
    csv_p, json_p = write_output(SAMPLE, out_dir=data_dir)
    print_summary(SAMPLE)
