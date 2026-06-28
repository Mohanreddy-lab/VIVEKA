"""Tests for src/data_loader.py — flexible dataset loader."""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_candidates, auto_find_dataset, _normalise_profile


class TestNormaliseProfile:
    def test_canonical_fields_pass_through(self):
        raw = {"id": "C1", "title": "Engineer", "skills": ["Python"], "summary": "Good"}
        out = _normalise_profile(raw)
        assert out["id"] == "C1"
        assert out["title"] == "Engineer"
        assert out["skills"] == ["Python"]
        assert out["summary"] == "Good"

    def test_headline_maps_to_title(self):
        raw = {"id": "C1", "headline": "Senior Data Engineer"}
        out = _normalise_profile(raw)
        assert out["title"] == "Senior Data Engineer"

    def test_about_maps_to_summary(self):
        raw = {"id": "C1", "about": "I build pipelines."}
        out = _normalise_profile(raw)
        assert out["summary"] == "I build pipelines."

    def test_bio_maps_to_summary(self):
        raw = {"id": "C1", "bio": "Python expert."}
        out = _normalise_profile(raw)
        assert out["summary"] == "Python expert."

    def test_tech_skills_merged_into_skills(self):
        raw = {"id": "C1", "skills": ["Python"], "tech_skills": ["Spark", "Airflow"]}
        out = _normalise_profile(raw)
        assert "Python" in out["skills"]
        assert "Spark" in out["skills"]
        assert "Airflow" in out["skills"]

    def test_candidate_id_used_when_id_missing(self):
        raw = {"candidate_id": "CAND42", "title": "Engineer"}
        out = _normalise_profile(raw)
        assert out["id"] == "CAND42"

    def test_id_takes_priority_over_candidate_id(self):
        raw = {"id": "REAL", "candidate_id": "FALLBACK"}
        out = _normalise_profile(raw)
        assert out["id"] == "REAL"

    def test_name_not_mapped_to_id(self):
        raw = {"name": "Alice", "id": "C1"}
        out = _normalise_profile(raw)
        assert out["id"] == "C1"
        assert out.get("name") == "Alice"

    def test_work_history_maps_to_experience(self):
        raw = {"id": "C1", "work_history": "Led pipelines at Google."}
        out = _normalise_profile(raw)
        assert out["experience"] == "Led pipelines at Google."

    def test_skills_comma_string_split_to_list(self):
        raw = {"id": "C1", "skills": "Python, Spark, Airflow"}
        out = _normalise_profile(raw)
        assert out["skills"] == ["Python", "Spark", "Airflow"]

    def test_skills_deduplication(self):
        raw = {"id": "C1", "skills": ["Python", "Spark"], "tech_skills": ["Python", "Airflow"]}
        out = _normalise_profile(raw)
        assert out["skills"].count("Python") == 1

    def test_synthetic_id_assigned_when_missing(self):
        raw = {"title": "Engineer", "skills": ["Python"]}
        out = _normalise_profile(raw)
        assert out["id"].startswith("C")


class TestLoadCandidatesJson:
    def test_load_list_json(self, tmp_path):
        data = [{"id": "C1", "title": "Engineer"}, {"id": "C2", "title": "Analyst"}]
        p = tmp_path / "profiles.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        profiles = load_candidates(p)
        assert len(profiles) == 2
        assert profiles[0]["id"] == "C1"

    def test_load_wrapped_json(self, tmp_path):
        data = {"candidates": [{"id": "C1"}, {"id": "C2"}]}
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        profiles = load_candidates(p)
        assert len(profiles) == 2

    def test_normalisation_applied(self, tmp_path):
        data = [{"candidate_id": "X1", "headline": "Data Eng", "about": "Builds pipelines."}]
        p = tmp_path / "data.json"
        p.write_text(json.dumps(data), encoding="utf-8")
        profiles = load_candidates(p)
        assert profiles[0]["id"] == "X1"
        assert profiles[0]["title"] == "Data Eng"
        assert profiles[0]["summary"] == "Builds pipelines."

    def test_file_not_found_raises(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_candidates(Path("/nonexistent/data.json"))

    def test_unsupported_extension_raises(self, tmp_path):
        import pytest
        p = tmp_path / "data.xlsx"
        p.write_text("x")
        with pytest.raises(ValueError):
            load_candidates(p)


class TestLoadCandidatesCsv:
    def test_load_csv(self, tmp_path):
        csv_text = "id,title,skills\nC1,Engineer,Python\nC2,Analyst,SQL\n"
        p = tmp_path / "profiles.csv"
        p.write_text(csv_text, encoding="utf-8")
        profiles = load_candidates(p)
        assert len(profiles) == 2
        assert profiles[0]["id"] == "C1"

    def test_csv_headline_normalised(self, tmp_path):
        csv_text = "candidate_id,headline\nABC,Senior Engineer\n"
        p = tmp_path / "profiles.csv"
        p.write_text(csv_text, encoding="utf-8")
        profiles = load_candidates(p)
        assert profiles[0]["id"] == "ABC"
        assert profiles[0]["title"] == "Senior Engineer"


class TestAutoFindDataset:
    def test_finds_json_before_csv(self, tmp_path):
        (tmp_path / "data.json").write_text('[{"id":"C1"}]')
        (tmp_path / "data.csv").write_text("id\nC1\n")
        found = auto_find_dataset(tmp_path)
        assert found is not None
        assert found.suffix == ".json"

    def test_skips_ranked_output(self, tmp_path):
        (tmp_path / "ranked_output.json").write_text('[{"id":"C1"}]')
        (tmp_path / "profiles.json").write_text('[{"id":"C2"}]')
        found = auto_find_dataset(tmp_path)
        assert found is not None
        assert "ranked_output" not in found.name

    def test_returns_none_when_empty(self, tmp_path):
        found = auto_find_dataset(tmp_path)
        assert found is None

    def test_finds_sample_as_fallback(self, tmp_path):
        (tmp_path / "sample_candidates.json").write_text('[{"id":"C1"}]')
        found = auto_find_dataset(tmp_path)
        assert found is not None


class TestLoadSampleCandidates:
    """Integration test: the built-in sample dataset loads correctly."""

    def test_sample_loads(self):
        sample = Path(__file__).parent.parent / "data" / "sample_candidates.json"
        if not sample.exists():
            return  # skip if not present
        profiles = load_candidates(sample)
        assert len(profiles) >= 10
        for p in profiles:
            assert "id" in p
            assert "title" in p

    def test_sample_has_skills(self):
        sample = Path(__file__).parent.parent / "data" / "sample_candidates.json"
        if not sample.exists():
            return
        profiles = load_candidates(sample)
        with_skills = [p for p in profiles if p.get("skills")]
        assert len(with_skills) > 0
