"""
api.py — VIVEKA REST API

Async job queue: POST /api/v1/rank returns a job_id immediately.
Poll GET /api/v1/jobs/{job_id} until status == "done".

Endpoints:
  GET  /health                    — liveness + Ollama connectivity
  GET  /api/v1/synonyms           — skill synonym dictionary
  POST /api/v1/rank               — submit ranking job, returns job_id
  GET  /api/v1/jobs/{job_id}      — poll job status / fetch result

Optional API key auth: set VIVEKA_API_KEY in .env to enable.
  All non-health endpoints require: Authorization: Bearer <key>

CORS: controlled via VIVEKA_CORS_ORIGINS (comma-separated).
  Default: "*" (open). Set to "https://yourdomain.com" in production.

Run:
  uvicorn src.api:app --reload --port 8000
"""

import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from skills import SKILL_SYNONYMS
from llm import check_ollama

log = logging.getLogger("viveka.api")

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="VIVEKA Candidate Ranking API",
    description=(
        "Intelligent offline candidate ranking powered by local LLMs.\n\n"
        "Submit a job via POST /api/v1/rank, then poll GET /api/v1/jobs/{job_id}."
    ),
    version="2.0.0",
)

# CORS — default open for local dev; restrict in production
_cors_origins = [o.strip() for o in os.getenv("VIVEKA_CORS_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)


# ---------------------------------------------------------------------------
# Optional API key auth
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)
_API_KEY = os.getenv("VIVEKA_API_KEY", "")


def _check_auth(creds: Optional[HTTPAuthorizationCredentials] = Security(_bearer)):
    if not _API_KEY:
        return  # auth disabled — no key configured
    if creds is None or creds.credentials != _API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key. Pass: Authorization: Bearer <key>",
        )


# ---------------------------------------------------------------------------
# In-memory job store  (replace with Redis for multi-process deployments)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class RankRequest(BaseModel):
    jd_text:     str            = Field(..., min_length=20, description="Raw job description text")
    profiles:    list[dict] | None = Field(
        None,
        description="Candidate profiles. Omit to load from data/profiles.json or demo.",
    )
    rerank_n:    int            = Field(10, ge=1, le=100, description="LLM rerank cap")
    score_top_n: int            = Field(50, ge=1,         description="Scoring → rerank pass")
    parallel:    bool           = Field(False,            description="Parallel LLM calls")


class JobSubmitted(BaseModel):
    job_id: str
    status: str = "queued"


class JobStatus(BaseModel):
    job_id:  str
    status:  str                   # queued | running | done | error
    result:  dict[str, Any] | None = None
    error:   str | None            = None


class HealthResponse(BaseModel):
    status:  str
    version: str
    ollama:  str


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline_sync(job_id: str, req: RankRequest) -> None:
    _jobs[job_id]["status"] = "running"
    try:
        from jd_parser import parse_jd
        from recall    import RecallEngine
        from scoring   import score_candidates
        from rerank    import rerank_candidates
        from output    import normalize_scores
        from agent     import load_profiles

        profiles = req.profiles or load_profiles(Path(__file__).parent.parent / "data")

        parsed   = parse_jd(req.jd_text)
        engine   = RecallEngine()
        engine.index_candidates(profiles)
        recalled = engine.recall(parsed, top_k=min(200, len(profiles)))
        scored   = score_candidates(recalled, parsed, top_n=req.score_top_n)
        ranked   = rerank_candidates(scored, parsed, top_n=req.rerank_n, parallel=req.parallel)
        ranked   = normalize_scores(ranked)

        clean = [{k: v for k, v in c.items() if not k.startswith("_")} for c in ranked]
        _jobs[job_id].update({
            "status": "done",
            "result": {
                "ranked":       clean,
                "total":        len(clean),
                "hidden_gems":  sum(1 for c in clean if c.get("hidden_gem")),
            },
        })
        log.info("Job %s done — %d candidates ranked.", job_id, len(clean))

    except Exception as exc:
        log.error("Job %s failed: %s", job_id, exc)
        _jobs[job_id].update({"status": "error", "error": str(exc)})


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
def health():
    _, ollama_msg = check_ollama()
    return HealthResponse(status="ok", version=app.version, ollama=ollama_msg)


@app.get("/api/v1/synonyms", tags=["Meta"], dependencies=[Depends(_check_auth)])
def get_synonyms():
    """Return the full skill synonym dictionary used by Stage 3 scoring."""
    return {"synonyms": SKILL_SYNONYMS}


@app.post(
    "/api/v1/rank",
    response_model=JobSubmitted,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Pipeline"],
    dependencies=[Depends(_check_auth)],
)
def rank_candidates(req: RankRequest, background_tasks: BackgroundTasks):
    """
    Submit a ranking job. Returns immediately with a job_id.
    Poll GET /api/v1/jobs/{job_id} until status is "done" or "error".
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"job_id": job_id, "status": "queued", "result": None, "error": None}
    background_tasks.add_task(_run_pipeline_sync, job_id, req)
    log.info("Job %s queued (rerank_n=%d parallel=%s).", job_id, req.rerank_n, req.parallel)
    return JobSubmitted(job_id=job_id)


@app.get(
    "/api/v1/jobs/{job_id}",
    response_model=JobStatus,
    tags=["Pipeline"],
    dependencies=[Depends(_check_auth)],
)
def get_job(job_id: str):
    """Poll a submitted job. Result is populated when status == 'done'."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatus(**job)


# ---------------------------------------------------------------------------
# CLI convenience — python src/api.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
