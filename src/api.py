"""
api.py — MANTHAN REST API

Wraps the 5-stage pipeline in a FastAPI server so any app can call it over HTTP.

Endpoints:
  GET  /health              — liveness check
  GET  /api/v1/synonyms     — full skill synonym dictionary
  POST /api/v1/rank         — run full pipeline, return ranked candidates

Run with:
  uvicorn src.api:app --reload --port 8000

Or from the repo root:
  python -m uvicorn src.api:app --reload --port 8000

Example request:
  curl -X POST http://localhost:8000/api/v1/rank \\
    -H "Content-Type: application/json" \\
    -d '{"jd_text": "Senior Data Engineer...", "rerank_n": 5}'
"""

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from skills import SKILL_SYNONYMS

app = FastAPI(
    title="MANTHAN Candidate Ranking API",
    description="Intelligent offline candidate ranking powered by local LLMs.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RankRequest(BaseModel):
    jd_text:   str           = Field(..., description="Raw job description text")
    profiles:  list[dict] | None = Field(
        None,
        description="List of candidate profile dicts. "
                    "If omitted, loads from data/profiles.json or demo profiles.",
    )
    rerank_n:  int           = Field(10, ge=1, le=100, description="LLM rerank cap")
    score_top_n: int         = Field(50, ge=1, description="How many to pass from scoring to rerank")


class HealthResponse(BaseModel):
    status:  str
    version: str


class RankResponse(BaseModel):
    ranked:    list[dict[str, Any]]
    total:     int
    hidden_gems: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    return HealthResponse(status="ok", version=app.version)


@app.get("/api/v1/synonyms", tags=["Meta"])
def get_synonyms():
    """Return the full skill synonym dictionary used by the pipeline."""
    return {"synonyms": SKILL_SYNONYMS}


@app.post("/api/v1/rank", response_model=RankResponse, tags=["Pipeline"])
def rank_candidates(req: RankRequest):
    """
    Run the full MANTHAN pipeline and return ranked candidates.

    If `profiles` is not provided, loads from data/profiles.json
    (or data/profiles.csv, or built-in demo profiles).
    """
    if not req.jd_text.strip():
        raise HTTPException(status_code=422, detail="jd_text must not be empty.")

    try:
        from jd_parser import parse_jd
        from recall    import RecallEngine
        from scoring   import score_candidates
        from rerank    import rerank_candidates
        from output    import normalize_scores

        # Load profiles
        if req.profiles:
            profiles = req.profiles
        else:
            from agent import load_profiles
            data_dir = Path(__file__).parent.parent / "data"
            profiles = load_profiles(data_dir)

        parsed   = parse_jd(req.jd_text)
        engine   = RecallEngine()
        engine.index_candidates(profiles)
        recalled = engine.recall(parsed, top_k=min(200, len(profiles)))
        scored   = score_candidates(recalled, parsed, top_n=req.score_top_n)
        ranked   = rerank_candidates(scored, parsed, top_n=req.rerank_n)
        ranked   = normalize_scores(ranked)

        # Remove internal bookkeeping keys before returning
        clean = []
        for c in ranked:
            entry = {k: v for k, v in c.items() if not k.startswith("_")}
            clean.append(entry)

        hidden_gems = sum(1 for c in clean if c.get("hidden_gem"))
        return RankResponse(ranked=clean, total=len(clean), hidden_gems=hidden_gems)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# CLI convenience — python src/api.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
