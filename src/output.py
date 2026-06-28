"""
output.py — Stage 5: Output Writer

Writes the final ranked shortlist in two formats:
  - ranked_output.csv   primary deliverable; organizers' format
  - ranked_output.json  full data for the Streamlit demo

Also prints a rich-formatted leaderboard to the terminal.

# ── HOW TO CHANGE THE OUTPUT COLUMNS ──────────────────────────────────────
#
# To match an organizer's exact column spec, edit CSV_COLUMNS below.
# The pipeline attaches all signals to each candidate dict; just name
# the keys you need and they'll appear in the CSV in that order.
#
# Available keys per candidate (after all pipeline stages):
#   rank, candidate_id, final_score, llm_score, confidence, hidden_gem,
#   reason, evidence (pipe-joined verified snippets), skill_score,
#   seniority_score, activity_score, composite_score, embedding_score,
#   calibrated_confidence, score_100, stuffing_ratio
#
# Example: organizer wants only rank, candidate_id, score, reason:
#   CSV_COLUMNS = ["rank", "candidate_id", "final_score", "reason"]
#
# The JSON output always includes ALL fields — only the CSV is trimmed.
# ──────────────────────────────────────────────────────────────────────────
"""

import csv
import json
from pathlib import Path
from typing import List, Dict

from config import ID_FIELDS

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box as rich_box
    _RICH = True
except ImportError:
    _RICH = False


CSV_COLUMNS = [
    "rank",
    "candidate_id",
    "final_score",
    "llm_score",
    "confidence",
    "hidden_gem",
    "reason",
    "evidence",
    "skill_score",
    "seniority_score",
    "activity_score",
    "composite_score",
    "embedding_score",
]


def _get_id(candidate: dict) -> str:
    for field in ID_FIELDS:
        val = candidate.get(field)
        if val:
            return str(val)
    return "UNKNOWN"


def _to_row(rank: int, candidate: dict) -> dict:
    # Flatten evidence_verified list into a pipe-separated string for CSV.
    evidence_list = candidate.get("evidence_verified") or candidate.get("evidence") or []
    if isinstance(evidence_list, list):
        evidence_str = " | ".join(str(e) for e in evidence_list)
    else:
        evidence_str = str(evidence_list)

    return {
        "rank":             rank,
        "candidate_id":     _get_id(candidate),
        "final_score":      round(float(candidate.get("final_score",      0.0)), 4),
        "llm_score":        candidate.get("llm_score",      "N/A"),
        "confidence":       candidate.get("confidence",     "N/A"),
        "hidden_gem":       "yes" if candidate.get("hidden_gem") else "no",
        "reason":           candidate.get("reason",         ""),
        "evidence":         evidence_str,
        "skill_score":      round(float(candidate.get("skill_score",      0.0)), 4),
        "seniority_score":  round(float(candidate.get("seniority_score",  0.0)), 4),
        "activity_score":   round(float(candidate.get("activity_score",   0.0)), 4),
        "composite_score":  round(float(candidate.get("composite_score",  0.0)), 4),
        "embedding_score":  round(float(candidate.get("embedding_score",  0.0)), 4),
    }


def write_output(
    ranked: List[Dict],
    out_dir: str | Path = "data",
    stem: str = "ranked_output",
) -> tuple[Path, Path]:
    """
    Write ranked shortlist to CSV and JSON.

    Returns (csv_path, json_path).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path  = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"

    rows = [_to_row(rank, c) for rank, c in enumerate(ranked, start=1)]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    enriched = []
    for rank, c in enumerate(ranked, start=1):
        entry = dict(c)
        entry["rank"]         = rank
        entry["candidate_id"] = _get_id(c)
        entry.pop("_id",        None)
        entry.pop("_rank_jump", None)
        enriched.append(entry)

    json_path.write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    print(f"[output] Wrote {len(rows)} candidates -> {csv_path}")
    print(f"[output] Wrote full data   -> {json_path}")
    return csv_path, json_path


def _conf_color(conf: str) -> str:
    return {"high": "green", "medium": "yellow", "low": "red"}.get(
        str(conf).lower(), "white"
    )


def print_summary(ranked: List[Dict], top_n: int = 10) -> None:
    """Print a leaderboard — uses Rich if available, falls back to plain text."""
    if _RICH:
        _print_summary_rich(ranked, top_n)
    else:
        _print_summary_plain(ranked, top_n)


def _print_summary_rich(ranked: List[Dict], top_n: int) -> None:
    console = Console(legacy_windows=False)
    gems    = [c for c in ranked if c.get("hidden_gem")]
    top     = ranked[:top_n]

    table = Table(
        title=f"MANTHAN — Final Shortlist  ({len(ranked)} candidates ranked)",
        box=rich_box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
        highlight=True,
    )
    table.add_column("#",           style="dim",     width=4,  justify="right")
    table.add_column("ID",                           width=14)
    table.add_column("Title",                        width=26)
    table.add_column("Score",       justify="right", width=7)
    table.add_column("LLM",         justify="right", width=5)
    table.add_column("Conf",        justify="center",width=8)
    table.add_column("Skill",       justify="right", width=6)
    table.add_column("Gem",         justify="center",width=4)
    table.add_column("Reason",                       min_width=40)

    for i, c in enumerate(top, 1):
        is_gem = c.get("hidden_gem", False)
        conf   = str(c.get("confidence", "—")).lower()
        score  = c.get("final_score", 0.0)
        row_style = "bold yellow" if is_gem else ("bold" if i <= 3 else "")

        table.add_row(
            str(i),
            _get_id(c)[:13],
            c.get("title", "")[:25],
            f"{score:.3f}",
            str(c.get("llm_score", "—")),
            f"[{_conf_color(conf)}]{conf[:6]}[/]",
            f"{c.get('skill_score', 0):.0%}",
            "★" if is_gem else "",
            c.get("reason", "")[:80],
            style=row_style,
        )

    console.print()
    console.print(table)
    console.print()
    _print_score_breakdown_rich(console, top[:5])

    if gems:
        gem_lines = "\n".join(
            f"  [bold yellow]★[/] [{_get_id(c)}] {c.get('title', '')} "
            f"— {c.get('reason', '')[:70]}"
            for c in gems
        )
        console.print(
            Panel(
                gem_lines,
                title=f"[bold yellow]★ Hidden Gems  ({len(gems)} found)[/]",
                border_style="yellow",
            )
        )
        console.print()


def _print_score_breakdown_rich(console, top5: List[Dict]) -> None:
    if not top5:
        return
    t = Table(
        title="Score Breakdown — Top 5",
        box=rich_box.SIMPLE,
        header_style="bold",
    )
    t.add_column("ID",          width=14)
    t.add_column("Embed",       justify="right", width=7)
    t.add_column("Skill",       justify="right", width=7)
    t.add_column("Seniority",   justify="right", width=10)
    t.add_column("Activity",    justify="right", width=9)
    t.add_column(">> Composite", justify="right", width=12)

    for c in top5:
        t.add_row(
            _get_id(c)[:13],
            f"{c.get('embedding_score', 0):.3f}",
            f"{c.get('skill_score',     0):.3f}",
            f"{c.get('seniority_score', 0):.3f}",
            f"{c.get('activity_score',  0):.3f}",
            f"[bold]{c.get('composite_score', 0):.3f}[/]",
        )
    console.print(t)


def _print_summary_plain(ranked: List[Dict], top_n: int) -> None:
    gems = [c for c in ranked if c.get("hidden_gem")]
    top  = ranked[:top_n]

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


def normalize_scores(ranked: List[Dict]) -> List[Dict]:
    """Add score_100 = final_score * 100 to each candidate (in-place)."""
    for c in ranked:
        c["score_100"] = round(c.get("final_score", 0.0) * 100, 1)
    return ranked


def validate_output(ranked: List[Dict]) -> bool:
    """
    Validate the ranked output for submission correctness.
    Checks: sequential ranks from 1, no null final_scores, sorted desc, non-empty reasons.
    Prints PASS/FAIL per check and returns True if all pass.
    """
    checks = []

    # 1. Ranks are 1..N with no gaps
    expected = list(range(1, len(ranked) + 1))
    actual   = [int(c.get("rank", c.get("_rank", i + 1))) for i, c in enumerate(ranked)]
    checks.append(("Ranks 1..N sequential", actual == expected))

    # 2. No null final_scores
    null_scores = [c.get("candidate_id") or c.get("id") for c in ranked
                   if c.get("final_score") is None]
    checks.append(("No null final_scores", len(null_scores) == 0))

    # 3. Sorted by final_score descending
    scores = [float(c.get("final_score", 0)) for c in ranked]
    checks.append(("Sorted by final_score desc", scores == sorted(scores, reverse=True)))

    # 4. Every row has a non-empty reason
    empty_reasons = [c.get("candidate_id") or c.get("id") for c in ranked
                     if not str(c.get("reason", "")).strip()]
    checks.append(("All rows have non-empty reason", len(empty_reasons) == 0))

    print(f"\n{'='*50}")
    print("  OUTPUT VALIDATION")
    print(f"{'='*50}")
    all_pass = True
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_pass = False

    if null_scores:
        print(f"         Null scores: {null_scores}")
    if empty_reasons:
        print(f"         Empty reasons: {empty_reasons}")

    final = "PASS" if all_pass else "FAIL"
    print(f"{'='*50}")
    print(f"  OVERALL: {final}  ({len(ranked)} candidates)")
    print(f"{'='*50}\n")
    return all_pass


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    SAMPLE = [
        {
            "id": "C001", "title": "Senior Data Engineer",
            "final_score": 0.838, "llm_score": 9, "confidence": "high",
            "hidden_gem": False, "reason": "8 years on AWS Spark pipelines; direct match.",
            "skill_score": 0.75, "seniority_score": 1.0, "activity_score": 0.80,
            "composite_score": 0.727, "embedding_score": 0.91,
        },
        {
            "id": "C004", "title": "Data Engineer",
            "final_score": 0.682, "llm_score": 7, "confidence": "medium",
            "hidden_gem": True,
            "reason": "Limited evidence: self-taught, no degree, but ships Spark + Airflow on AWS.",
            "skill_score": 0.50, "seniority_score": 0.55, "activity_score": 0.0,
            "composite_score": 0.614, "embedding_score": 0.78,
        },
        {
            "id": "C003", "title": "Data Analyst",
            "final_score": 0.418, "llm_score": 4, "confidence": "low",
            "hidden_gem": False, "reason": "Limited evidence: no pipeline experience stated.",
            "skill_score": 0.25, "seniority_score": 0.55, "activity_score": 0.0,
            "composite_score": 0.435, "embedding_score": 0.62,
        },
    ]

    data_dir = Path(__file__).parent.parent / "data"
    csv_p, json_p = write_output(SAMPLE, out_dir=data_dir)
    print_summary(SAMPLE)
