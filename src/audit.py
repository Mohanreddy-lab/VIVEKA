"""
audit.py — Full audit trail for every VIVEKA ranking decision.

Hiring is legally sensitive — decisions must be inspectable, reproducible,
and explainable. Every pipeline run appends to data/audit.jsonl (one JSON
per line), keyed by run_id so all candidates from one run stay linked.

The run_id is a SHA-256 hash of (JD text + model + weights), so the same
inputs always produce the same run_id — i.e., reproducibility is provable.

Usage:
    from audit import AuditLogger
    logger = AuditLogger(out_dir, jd_text, parsed_jd, model, weights)
    logger.log_candidate(candidate, rank=1)
    run_id = logger.flush()   # returns the run_id string
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

log = logging.getLogger("viveka.audit")

# Fields written per candidate — internal bookkeeping (_id, _rank_jump) excluded
_CANDIDATE_FIELDS = (
    "rank", "candidate_id",
    "final_score", "composite_score",
    "embedding_score", "skill_score", "seniority_score", "activity_score",
    "llm_score", "confidence", "calibrated_confidence",
    "hidden_gem", "reason",
    "evidence_verified", "evidence_unsupported",
    "skill_evidence", "stuffing",
)


def _run_id(jd_text: str, model: str, weights: Tuple) -> str:
    payload = f"{jd_text[:300]}|{model}|{weights}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class AuditLogger:
    """
    Collect audit records for one pipeline run, then flush to JSONL.
    Designed to add zero overhead when audit logging is disabled.
    """

    def __init__(
        self,
        out_dir: Path,
        jd_text: str,
        parsed_jd: dict,
        model: str,
        weights: Tuple,  # (embed, skill, seniority, activity)
    ):
        self._out_dir   = Path(out_dir)
        self._run_id    = _run_id(jd_text, model, weights)
        self._model     = model
        self._weights   = dict(zip(("embed", "skill", "seniority", "activity"),
                                   [round(w, 4) for w in weights]))
        self._jd_summary = (
            f"seniority={parsed_jd.get('seniority')} "
            f"required={parsed_jd.get('required_skills', [])[:5]}"
        )
        self._timestamp = datetime.now(timezone.utc).isoformat()
        self._records: List[dict] = []

    def log_candidate(self, candidate: dict, rank: int) -> None:
        """Append one candidate's audit record to the in-memory buffer."""
        rec = {
            "timestamp":   self._timestamp,
            "run_id":      self._run_id,
            "model":       self._model,
            "weights":     self._weights,
            "jd_summary":  self._jd_summary,
        }
        cid = candidate.get("candidate_id") or candidate.get("id") or f"rank_{rank}"
        rec["rank"]         = rank
        rec["candidate_id"] = cid

        for field in _CANDIDATE_FIELDS:
            if field in candidate and field not in rec:
                rec[field] = candidate[field]

        self._records.append(rec)

    def flush(self) -> str:
        """Write all buffered records to audit.jsonl. Returns the run_id."""
        if not self._records:
            return self._run_id

        path = self._out_dir / "audit.jsonl"
        try:
            self._out_dir.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                for rec in self._records:
                    f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
            log.info(
                "Audit: %d records → %s  run_id=%s",
                len(self._records), path, self._run_id,
            )
        except Exception as exc:
            log.warning("Audit write failed: %s", exc)

        return self._run_id

    @property
    def run_id(self) -> str:
        return self._run_id


def is_audit_on() -> bool:
    return os.getenv("VIVEKA_AUDIT", "on").lower() not in ("off", "0", "false", "no")
