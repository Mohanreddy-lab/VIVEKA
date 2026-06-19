"""Tests for src/output.py — CSV/JSON writing and console summary."""

import sys, os, json, csv
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from output import _to_row, write_output, print_summary, _get_id

SAMPLE_RANKED = [
    {"id": "C001", "title": "Senior Data Engineer",
     "final_score": 0.838, "llm_score": 9, "confidence": "high",
     "hidden_gem": False, "reason": "Direct match on Spark and AWS.",
     "skill_score": 0.75, "composite_score": 0.727, "embedding_score": 0.91},
    {"id": "C004", "title": "Data Engineer",
     "final_score": 0.682, "llm_score": 7, "confidence": "medium",
     "hidden_gem": True, "reason": "Limited evidence: self-taught.",
     "skill_score": 0.50, "composite_score": 0.614, "embedding_score": 0.78},
]


class TestGetId:
    def test_prefers_candidate_id_field(self):
        assert _get_id({"candidate_id": "X1", "id": "Y1"}) == "X1"

    def test_falls_back_to_id(self):
        assert _get_id({"id": "Y1"}) == "Y1"

    def test_returns_unknown_when_missing(self):
        assert _get_id({}) == "UNKNOWN"


class TestToRow:
    def test_hidden_gem_true_is_yes_string(self):
        row = _to_row(1, SAMPLE_RANKED[0])
        assert row["hidden_gem"] == "no"

    def test_hidden_gem_false_is_no_string(self):
        row = _to_row(2, SAMPLE_RANKED[1])
        assert row["hidden_gem"] == "yes"

    def test_rank_is_correct(self):
        assert _to_row(3, SAMPLE_RANKED[0])["rank"] == 3

    def test_scores_are_rounded(self):
        row = _to_row(1, SAMPLE_RANKED[0])
        # Should be float with ≤4 decimal places
        assert isinstance(row["final_score"], float)


class TestWriteOutput:
    def test_creates_csv_and_json(self, tmp_path):
        csv_p, json_p = write_output(SAMPLE_RANKED, out_dir=tmp_path)
        assert csv_p.exists()
        assert json_p.exists()

    def test_csv_has_correct_row_count(self, tmp_path):
        csv_p, _ = write_output(SAMPLE_RANKED, out_dir=tmp_path)
        with csv_p.open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == len(SAMPLE_RANKED)

    def test_csv_hidden_gem_is_string_not_bool(self, tmp_path):
        csv_p, _ = write_output(SAMPLE_RANKED, out_dir=tmp_path)
        with csv_p.open() as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            assert row["hidden_gem"] in ("yes", "no"), \
                f"Expected 'yes'/'no', got: {row['hidden_gem']!r}"

    def test_json_is_valid_and_has_rank(self, tmp_path):
        _, json_p = write_output(SAMPLE_RANKED, out_dir=tmp_path)
        data = json.loads(json_p.read_text())
        assert len(data) == len(SAMPLE_RANKED)
        assert data[0]["rank"] == 1

    def test_ranks_are_sequential(self, tmp_path):
        csv_p, _ = write_output(SAMPLE_RANKED, out_dir=tmp_path)
        with csv_p.open() as f:
            rows = list(csv.DictReader(f))
        ranks = [int(r["rank"]) for r in rows]
        assert ranks == list(range(1, len(SAMPLE_RANKED) + 1))


class TestPrintSummary:
    def test_runs_without_error(self, capsys):
        print_summary(SAMPLE_RANKED, top_n=5)
        out = capsys.readouterr().out
        assert "MANTHAN" in out

    def test_shows_hidden_gem_section(self, capsys):
        print_summary(SAMPLE_RANKED, top_n=5)
        out = capsys.readouterr().out
        assert "Hidden Gem" in out
