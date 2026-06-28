"""Tests for src/scoring.py — skill matching, activity scoring, composite scoring,
stuffing detection, and counterfactual explainer."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scoring import (
    _skill_match,
    _skill_score,
    _activity_score,
    _is_hidden_gem,
    score_candidates,
    detect_stuffing,
    explain_why_not_higher,
)

PARSED_JD = {
    "required_skills": ["Python", "Apache Spark", "Airflow", "AWS"],
    "implied_skills":  ["Git", "SQL", "Linux", "dbt"],
    "seniority":       "senior",
    "latent_needs":    ["owns pipelines end-to-end"],
}


class TestSkillMatch:
    """The single most important function: synonym-aware word-boundary matching."""

    def test_exact_match(self):
        assert _skill_match("Python", "strong python background")

    def test_case_insensitive(self):
        assert _skill_match("SQL", "expert in sql and data modeling")

    def test_sql_does_not_match_nosql(self):
        assert not _skill_match("SQL", "nosql expert")

    def test_sql_matches_in_context(self):
        assert _skill_match("SQL", "sql, python, and tableau")

    def test_synonym_spark_matches_apache_spark(self):
        assert _skill_match("Spark", "experience with apache spark at scale")

    def test_pyspark_matches_apache_spark(self):
        assert _skill_match("Apache Spark", "pyspark developer")

    def test_csharp_special_chars(self):
        assert _skill_match("C#", "built services in c# and .net")

    def test_dotnet_special_chars(self):
        assert _skill_match(".NET", "senior .net developer")

    def test_cpp_special_chars(self):
        assert _skill_match("C++", "wrote low-latency c++ code")

    def test_r_language_no_false_positive(self):
        assert not _skill_match("R", "react and rest api experience")

    def test_r_language_matches_alone(self):
        assert _skill_match("R", "proficient in r, python, and stata")


class TestSkillScore:
    def test_perfect_match(self):
        candidate = {
            "skills": ["Python", "Apache Spark", "Airflow", "AWS", "Git", "SQL", "Linux", "dbt"],
        }
        score = _skill_score(candidate, PARSED_JD)
        assert score == 1.0

    def test_zero_match(self):
        candidate = {"skills": ["React", "TypeScript", "Figma"]}
        score = _skill_score(candidate, PARSED_JD)
        assert score == 0.0

    def test_partial_match_between_zero_and_one(self):
        candidate = {"skills": ["Python", "Airflow"]}
        score = _skill_score(candidate, PARSED_JD)
        assert 0.0 < score < 1.0

    def test_no_skill_data_returns_zero(self):
        score = _skill_score({}, {"required_skills": [], "implied_skills": []})
        assert score == 0.0

    def test_required_skills_weighted_higher_than_implied(self):
        c_required = {"skills": ["Python"]}   # 1 required skill
        c_implied  = {"skills": ["Git"]}       # 1 implied skill
        s_req = _skill_score(c_required, PARSED_JD)
        s_imp = _skill_score(c_implied,  PARSED_JD)
        assert s_req > s_imp

    def test_returns_float(self):
        score = _skill_score({"skills": ["Python"]}, PARSED_JD)
        assert isinstance(score, float)


class TestActivityScore:
    def test_no_data_returns_zero(self):
        assert _activity_score({}) == 0.0

    def test_github_repos(self):
        score = _activity_score({"github_repos": 15})
        assert 0.0 < score <= 1.0

    def test_max_repos_capped_at_one(self):
        assert _activity_score({"github_repos": 9999}) == 1.0

    def test_projects_as_list(self):
        score = _activity_score({"projects": ["proj1", "proj2", "proj3"]})
        assert 0.0 < score <= 1.0

    def test_endorsements(self):
        score = _activity_score({"endorsements": 10})
        assert 0.0 < score <= 1.0

    def test_multiple_signals_averaged(self):
        score_both  = _activity_score({"github_repos": 30, "endorsements": 20})
        score_repos = _activity_score({"github_repos": 30})
        assert score_both == score_repos == 1.0


class TestIsHiddenGem:
    def test_high_score_modest_title_is_gem(self):
        candidate = {"title": "Data Engineer", "skills": ["Spark", "Airflow", "Python"]}
        assert _is_hidden_gem(candidate, composite=0.70, skill=0.60)

    def test_high_score_senior_title_not_gem(self):
        candidate = {"title": "Senior Data Engineer", "skills": ["A", "B", "C", "D", "E"]}
        assert not _is_hidden_gem(candidate, composite=0.70, skill=0.60)

    def test_lead_title_not_gem(self):
        candidate = {"title": "Lead Data Engineer"}
        assert not _is_hidden_gem(candidate, composite=0.70, skill=0.60)

    def test_low_composite_not_gem(self):
        candidate = {"title": "Engineer", "skills": ["Spark"]}
        assert not _is_hidden_gem(candidate, composite=0.30, skill=0.60)

    def test_low_skill_not_gem(self):
        candidate = {"title": "Engineer", "skills": ["Spark"]}
        assert not _is_hidden_gem(candidate, composite=0.70, skill=0.20)


class TestScoreCandidates:
    CANDIDATES = [
        {"id": "C1", "title": "Senior Data Engineer",
         "skills": ["Python", "Apache Spark", "Airflow", "AWS", "dbt", "SQL"],
         "embedding_score": 0.91},
        {"id": "C2", "title": "Frontend Developer",
         "skills": ["React", "CSS"],
         "embedding_score": 0.30},
        {"id": "C3", "title": "Data Analyst",
         "skills": ["SQL", "Python"],
         "embedding_score": 0.60},
    ]

    def test_returns_sorted_by_composite(self):
        results = score_candidates(self.CANDIDATES, PARSED_JD, top_n=3)
        scores = [c["composite_score"] for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_required_keys_present(self):
        results = score_candidates(self.CANDIDATES, PARSED_JD)
        for c in results:
            assert "composite_score" in c
            assert "skill_score" in c
            assert "seniority_score" in c
            assert "activity_score" in c
            assert "hidden_gem" in c
            assert "skill_evidence" in c

    def test_skill_evidence_has_matched_and_missing(self):
        results = score_candidates(self.CANDIDATES, PARSED_JD)
        for c in results:
            ev = c["skill_evidence"]
            assert "required_matched" in ev
            assert "required_missing" in ev

    def test_top_n_respected(self):
        results = score_candidates(self.CANDIDATES, PARSED_JD, top_n=2)
        assert len(results) == 2

    def test_best_candidate_ranks_first(self):
        results = score_candidates(self.CANDIDATES, PARSED_JD, top_n=3)
        assert results[0]["id"] == "C1"

    def test_worst_candidate_ranks_last(self):
        results = score_candidates(self.CANDIDATES, PARSED_JD, top_n=3)
        assert results[-1]["id"] == "C2"

    def test_rank_jump_key_present(self):
        results = score_candidates(self.CANDIDATES, PARSED_JD)
        for c in results:
            assert "_rank_jump" in c

    def test_stuffing_key_present(self):
        results = score_candidates(self.CANDIDATES, PARSED_JD)
        for c in results:
            assert "stuffing" in c
            assert "stuffing_ratio" in c["stuffing"]
            assert "claimed_unsupported" in c["stuffing"]

    def test_counterfactual_key_present(self):
        results = score_candidates(self.CANDIDATES, PARSED_JD)
        for c in results:
            assert "counterfactual" in c
            assert "ceiling" in c["counterfactual"]


class TestStuffingDetector:
    """Keyword-stuffing detection: listed skills vs narrative evidence."""

    PARSED_JD = {
        "required_skills": ["Python", "Apache Spark", "Airflow"],
        "implied_skills":  ["SQL"],
        "seniority":       "mid",
        "latent_needs":    [],
    }

    def test_stuffed_profile_high_ratio(self):
        """Candidate lists 6 skills but summary never mentions any of them."""
        candidate = {
            "id": "STUFF",
            "skills": ["Python", "Spark", "Airflow", "Kafka", "dbt", "Kubernetes"],
            # summary is vague — no skill names appear
            "summary": "Experienced professional with various technical competencies.",
        }
        result = detect_stuffing(candidate, self.PARSED_JD)
        assert result["stuffing_ratio"] > 0.5
        assert len(result["claimed_unsupported"]) > 0

    def test_genuine_profile_zero_ratio(self):
        """Candidate whose skills are all backed by narrative text."""
        candidate = {
            "id": "GENUINE",
            "skills": ["Python", "Airflow"],
            "summary": "Built Python-based Airflow DAGs for batch processing.",
            "experience": "5 years writing Python. Managed Airflow pipelines daily.",
        }
        result = detect_stuffing(candidate, self.PARSED_JD)
        assert result["stuffing_ratio"] == 0.0
        assert result["claimed_unsupported"] == []

    def test_penalty_applied_to_stuffed_profile(self):
        """Stuffed profile gets lower skill_score than genuine profile."""
        stuffed = {
            "id": "S1", "title": "Engineer",
            "skills": ["Python", "Apache Spark", "Airflow", "Kafka", "dbt"],
            "summary": "Experienced professional.",
            "embedding_score": 0.70,
        }
        genuine = {
            "id": "G1", "title": "Engineer",
            "skills": ["Python", "Apache Spark", "Airflow"],
            "summary": "Used Python and Apache Spark to build Airflow pipelines.",
            "embedding_score": 0.70,
        }
        results = score_candidates([stuffed, genuine], self.PARSED_JD, top_n=2)
        # Stuffed candidate's skill_score should be discounted relative to raw
        stuffed_r = next(c for c in results if c["id"] == "S1")
        assert stuffed_r["skill_score"] < stuffed_r["skill_score_raw"]

    def test_no_skills_field_zero_ratio(self):
        """Profile with no structured skills field → nothing to stuff."""
        candidate = {"id": "EMPTY", "summary": "Good engineer."}
        result = detect_stuffing(candidate, self.PARSED_JD)
        assert result["stuffing_ratio"] == 0.0


class TestCounterfactual:
    """Counterfactual explainer: 'would rank higher with evidence of X'."""

    PARSED_JD = {
        "required_skills": ["Python", "Apache Spark", "Airflow", "AWS"],
        "implied_skills":  ["SQL"],
        "seniority":       "senior",
        "latent_needs":    [],
    }

    def test_missing_skills_in_output(self):
        candidate = {
            "composite_score": 0.50,
            "embedding_score": 0.70,
            "skill_score":     0.40,
            "seniority_score": 1.0,
            "activity_score":  0.0,
            "skill_evidence": {
                "required_matched": ["Python"],
                "required_missing": ["Apache Spark", "Airflow", "AWS"],
                "implied_matched":  [],
                "implied_missing":  ["SQL"],
            },
        }
        result = explain_why_not_higher(candidate, self.PARSED_JD, 5, [0.90, 0.80, 0.70, 0.60, 0.50])
        assert "Apache Spark" in result["ceiling"] or "Airflow" in result["ceiling"]
        assert len(result["missing_skills"]) == 3
        assert result["estimated_rank_gain"] >= 0

    def test_no_missing_skills_returns_ceiling_message(self):
        candidate = {
            "composite_score": 0.90,
            "skill_evidence": {
                "required_matched": ["Python", "Apache Spark", "Airflow", "AWS"],
                "required_missing": [],
                "implied_matched":  ["SQL"],
                "implied_missing":  [],
            },
        }
        result = explain_why_not_higher(candidate, self.PARSED_JD, 1, [0.90])
        assert result["estimated_rank_gain"] == 0
        assert "ceiling" in result["ceiling"].lower()
