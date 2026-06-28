"""Tests for src/recall.py — text builders and FAISS index."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from recall import build_jd_text, build_profile_text, RecallEngine


class TestBuildJdText:
    def test_includes_required_skills(self):
        jd = {"required_skills": ["Python", "Spark"], "implied_skills": [],
              "seniority": "senior", "latent_needs": []}
        text = build_jd_text(jd)
        assert "Python" in text
        assert "Spark" in text

    def test_includes_seniority(self):
        jd = {"required_skills": [], "implied_skills": [],
              "seniority": "junior", "latent_needs": []}
        assert "junior" in build_jd_text(jd)

    def test_empty_jd_returns_string(self):
        text = build_jd_text({})
        assert isinstance(text, str)


class TestBuildProfileText:
    def test_includes_title_and_skills(self):
        profile = {"title": "Data Engineer", "skills": ["Python", "Spark"]}
        text = build_profile_text(profile)
        assert "Data Engineer" in text
        assert "Python" in text

    def test_list_skills_joined(self):
        profile = {"skills": ["A", "B", "C"]}
        text = build_profile_text(profile)
        assert "A" in text and "B" in text and "C" in text

    def test_empty_profile_returns_string(self):
        text = build_profile_text({})
        assert isinstance(text, str)

    def test_ignores_short_fallback_values(self):
        # Short values like IDs ("C001") should not pollute the embedding text
        profile = {"id": "C001"}   # no known semantic fields
        text = build_profile_text(profile)
        # "C001" is 4 chars, below the 10-char threshold in the fallback
        assert "C001" not in text


class TestRecallEngine:
    PROFILES = [
        {"id": "C1", "title": "Data Engineer",
         "skills": ["Python", "Spark"], "summary": "Builds ETL pipelines."},
        {"id": "C2", "title": "Frontend Developer",
         "skills": ["React", "CSS"], "summary": "Builds web interfaces."},
        {"id": "C3", "title": "ML Engineer",
         "skills": ["Python", "PyTorch"], "summary": "Trains ML models."},
    ]

    JD = {
        "required_skills": ["Python", "Spark"],
        "implied_skills":  ["SQL"],
        "seniority":       "mid",
        "latent_needs":    ["data pipelines"],
    }

    def test_index_and_recall_returns_results(self):
        engine = RecallEngine()
        engine.index_candidates(self.PROFILES)
        results = engine.recall(self.JD, top_k=3)
        assert len(results) == 3

    def test_top_result_has_embedding_score(self):
        engine = RecallEngine()
        engine.index_candidates(self.PROFILES)
        results = engine.recall(self.JD, top_k=1)
        assert "embedding_score" in results[0]
        assert 0.0 <= results[0]["embedding_score"] <= 1.0

    def test_data_engineer_ranks_above_frontend(self):
        engine = RecallEngine()
        engine.index_candidates(self.PROFILES)
        results = engine.recall(self.JD, top_k=3)
        ids = [r["id"] for r in results]
        assert ids.index("C1") < ids.index("C2")

    def test_recall_before_index_raises(self):
        import pytest
        engine = RecallEngine()
        with pytest.raises(RuntimeError):
            engine.recall(self.JD)

    def test_top_k_capped_at_pool_size(self):
        engine = RecallEngine()
        engine.index_candidates(self.PROFILES)
        results = engine.recall(self.JD, top_k=999)
        assert len(results) == len(self.PROFILES)
