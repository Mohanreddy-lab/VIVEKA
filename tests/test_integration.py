"""
test_integration.py — End-to-end pipeline test with a mocked LLM.

Tests the full flow: JD text → parse → recall → score → rerank → output.
The LLM (jd_parser + rerank) is mocked so no Ollama server is required.
"""

import sys
import os
import json
import csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# LLM mock helpers
# ---------------------------------------------------------------------------

JD_PARSE_RESPONSE = json.dumps({
    "required_skills": ["Python", "Apache Spark", "Airflow", "AWS"],
    "implied_skills":  ["Git", "SQL", "Linux"],
    "seniority":       "senior",
    "latent_needs":    ["owns pipelines end-to-end"],
})

RERANK_RESPONSE = json.dumps({
    "analysis":   "Candidate has relevant experience with Spark and Airflow.",
    "llm_score":  8,
    "reason":     "Strong pipeline experience with Spark and AWS. Direct match.",
    "confidence": "high",
})


def _make_mock_llm(response_text: str):
    """Return a mock LLM that always responds with `response_text`."""
    msg = MagicMock()
    msg.content = response_text

    chain = MagicMock()
    chain.invoke.return_value = msg

    prompt = MagicMock()
    prompt.__or__ = MagicMock(return_value=chain)

    llm = MagicMock()
    return llm, prompt


def _patch_llms(monkeypatch):
    """Patch get_llm and ChatPromptTemplate for both jd_parser and rerank."""
    import jd_parser
    import rerank

    # jd_parser: always returns JD parse JSON
    jd_llm = MagicMock()
    jd_chain = MagicMock()
    jd_chain.invoke.return_value = MagicMock(content=JD_PARSE_RESPONSE)
    jd_prompt = MagicMock()
    jd_prompt.__or__ = MagicMock(return_value=jd_chain)
    monkeypatch.setattr(jd_parser, "get_llm", lambda **kw: jd_llm)

    # rerank: always returns rerank response
    rr_chain = MagicMock()
    rr_chain.invoke.return_value = MagicMock(content=RERANK_RESPONSE)
    rr_prompt = MagicMock()
    rr_prompt.__or__ = MagicMock(return_value=rr_chain)
    monkeypatch.setattr(rerank, "get_llm", lambda **kw: MagicMock())

    # Patch ChatPromptTemplate globally — used by both modules
    def _from_messages(*args, **kwargs):
        caller = sys._getframe(1).f_globals.get("__name__", "")
        if "jd_parser" in caller:
            return jd_prompt
        return rr_prompt

    monkeypatch.setattr(
        "langchain_core.prompts.ChatPromptTemplate.from_messages",
        _from_messages,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PROFILES = [
    {
        "id": "C001", "title": "Senior Data Engineer",
        "skills": ["Python", "Apache Spark", "Airflow", "AWS", "dbt", "SQL"],
        "summary": "8 years building large-scale data pipelines on AWS.",
        "github_repos": 42,
    },
    {
        "id": "C002", "title": "Junior Frontend Developer",
        "skills": ["React", "TypeScript", "CSS"],
        "summary": "2 years building web UIs.",
        "github_repos": 8,
    },
    {
        "id": "C003", "title": "Data Analyst",
        "skills": ["SQL", "Python", "Tableau"],
        "summary": "Strong SQL. No pipeline experience.",
    },
    {
        "id": "C004", "title": "Data Engineer",
        "skills": ["Spark", "Airflow", "Python", "S3"],
        "summary": "Built batch pipelines on AWS. Self-taught.",
    },
    {
        "id": "C005", "title": "ML Engineer",
        "skills": ["Python", "PyTorch", "Spark", "Kubernetes", "MLflow"],
        "summary": "Deploys ML models at scale with feature stores on AWS.",
        "github_repos": 25,
    },
]

SAMPLE_JD = """
Senior Data Engineer — Platform Team

Requirements:
- Python and SQL
- Apache Spark and Airflow
- AWS (S3, Glue, Redshift)
- 5+ years experience

Nice to have: dbt, Kafka, MLflow.

Responsibilities: own pipelines end-to-end, mentor junior engineers.
"""


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_pipeline_returns_ranked_list(self, monkeypatch):
        _patch_llms(monkeypatch)
        from jd_parser import parse_jd
        from recall    import RecallEngine
        from scoring   import score_candidates
        from rerank    import rerank_candidates

        parsed   = parse_jd(SAMPLE_JD)
        engine   = RecallEngine()
        engine.index_candidates(PROFILES)
        recalled = engine.recall(parsed, top_k=5)
        scored   = score_candidates(recalled, parsed, top_n=5)
        ranked   = rerank_candidates(scored, parsed, top_n=3)

        assert len(ranked) == 3
        assert all("final_score" in c for c in ranked)

    def test_pipeline_extracts_skills(self, monkeypatch):
        _patch_llms(monkeypatch)
        from jd_parser import parse_jd
        parsed = parse_jd(SAMPLE_JD)
        assert len(parsed["required_skills"]) >= 1
        assert parsed["seniority"] in ("junior", "mid", "senior", "lead", "unknown")

    def test_data_engineer_ranks_above_frontend(self, monkeypatch):
        _patch_llms(monkeypatch)
        from jd_parser import parse_jd
        from recall    import RecallEngine
        from scoring   import score_candidates
        from rerank    import rerank_candidates

        parsed   = parse_jd(SAMPLE_JD)
        engine   = RecallEngine()
        engine.index_candidates(PROFILES)
        recalled = engine.recall(parsed, top_k=5)
        scored   = score_candidates(recalled, parsed, top_n=5)
        ranked   = rerank_candidates(scored, parsed, top_n=5)

        ids = [c.get("id") for c in ranked]
        # C001 (Senior Data Engineer) must rank above C002 (Frontend)
        if "C001" in ids and "C002" in ids:
            assert ids.index("C001") < ids.index("C002"), (
                f"Senior Data Engineer should rank above Frontend Developer. Got: {ids}"
            )

    def test_ranked_sorted_descending(self, monkeypatch):
        _patch_llms(monkeypatch)
        from jd_parser import parse_jd
        from recall    import RecallEngine
        from scoring   import score_candidates
        from rerank    import rerank_candidates

        parsed   = parse_jd(SAMPLE_JD)
        engine   = RecallEngine()
        engine.index_candidates(PROFILES)
        recalled = engine.recall(parsed, top_k=5)
        scored   = score_candidates(recalled, parsed, top_n=5)
        ranked   = rerank_candidates(scored, parsed, top_n=5)

        scores = [c["final_score"] for c in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_output_writes_valid_csv(self, monkeypatch, tmp_path):
        _patch_llms(monkeypatch)
        from jd_parser import parse_jd
        from recall    import RecallEngine
        from scoring   import score_candidates
        from rerank    import rerank_candidates
        from output    import write_output

        parsed   = parse_jd(SAMPLE_JD)
        engine   = RecallEngine()
        engine.index_candidates(PROFILES)
        recalled = engine.recall(parsed, top_k=5)
        scored   = score_candidates(recalled, parsed, top_n=5)
        ranked   = rerank_candidates(scored, parsed, top_n=3)

        csv_p, json_p = write_output(ranked, out_dir=tmp_path)

        assert csv_p.exists()
        assert json_p.exists()

        with csv_p.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert all(r["hidden_gem"] in ("yes", "no") for r in rows)

        data = json.loads(json_p.read_text())
        assert data[0]["rank"] == 1

    def test_all_candidates_have_required_output_fields(self, monkeypatch):
        _patch_llms(monkeypatch)
        from jd_parser import parse_jd
        from recall    import RecallEngine
        from scoring   import score_candidates
        from rerank    import rerank_candidates

        parsed   = parse_jd(SAMPLE_JD)
        engine   = RecallEngine()
        engine.index_candidates(PROFILES)
        recalled = engine.recall(parsed, top_k=5)
        scored   = score_candidates(recalled, parsed, top_n=5)
        ranked   = rerank_candidates(scored, parsed, top_n=3)

        required_keys = {
            "final_score", "llm_score", "confidence", "reason",
            "skill_score", "seniority_score", "activity_score",
            "composite_score", "embedding_score", "hidden_gem",
        }
        for c in ranked:
            missing = required_keys - set(c.keys())
            assert not missing, f"Candidate missing keys: {missing}"
