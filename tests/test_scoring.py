"""Tests for src/scoring.py — skill matching, activity scoring, composite scoring."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from scoring import (
    _skill_match,
    _skill_score,
    _activity_score,
    _is_hidden_gem,
    score_candidates,
)

PARSED_JD = {
    "required_skills": ["Python", "Apache Spark", "Airflow", "AWS"],
    "implied_skills":  ["Git", "SQL", "Linux", "dbt"],
    "seniority":       "senior",
    "latent_needs":    ["owns pipelines end-to-end"],
}


class TestSkillMatch:
    """The single most important function to test: boundary matching."""

    def test_exact_match(self):
        assert _skill_match("Python", "strong python background")

    def test_case_insensitive(self):
        assert _skill_match("SQL", "expert in sql and data modeling")

    def test_sql_does_not_match_nosql(self):
        # The original bug — verify it's fixed
        assert not _skill_match("SQL", "nosql expert")

    def test_sql_matches_in_context(self):
        assert _skill_match("SQL", "sql, python, and tableau")

    def test_partial_name_matches(self):
        # "Spark" should match "Apache Spark"
        assert _skill_match("Spark", "experience with apache spark at scale")

    def test_csharp_special_chars(self):
        assert _skill_match("C#", "built services in c# and .net")

    def test_dotnet_special_chars(self):
        assert _skill_match(".NET", "senior .net developer")

    def test_cpp_special_chars(self):
        assert _skill_match("C++", "wrote low-latency c++ code")

    def test_r_language_no_false_positive(self):
        # "R" should not match "React" or "REST"
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
        # Candidate matching only required vs only implied
        c_required = {"skills": ["Python"]}   # 1 required skill
        c_implied  = {"skills": ["Git"]}       # 1 implied skill
        s_req = _skill_score(c_required, PARSED_JD)
        s_imp = _skill_score(c_implied,  PARSED_JD)
        assert s_req > s_imp


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
        score_both   = _activity_score({"github_repos": 30, "endorsements": 20})
        score_repos  = _activity_score({"github_repos": 30})
        # Both signals at max → same as one signal at max
        assert score_both == score_repos == 1.0


class TestIsHiddenGem:
    def test_high_score_modest_title_is_gem(self):
        candidate = {"title": "Data Engineer", "skills": ["Spark", "Airflow", "Python"]}
        assert _is_hidden_gem(candidate, composite=0.70, skill=0.60)

    def test_high_score_senior_title_not_gem(self):
        candidate = {"title": "Senior Data Engineer", "skills": ["A", "B", "C", "D", "E"]}
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
            assert "activity_score" in c
            assert "hidden_gem" in c

    def test_top_n_respected(self):
        results = score_candidates(self.CANDIDATES, PARSED_JD, top_n=2)
        assert len(results) == 2

    def test_best_candidate_ranks_first(self):
        results = score_candidates(self.CANDIDATES, PARSED_JD, top_n=3)
        assert results[0]["id"] == "C1"

    def test_worst_candidate_ranks_last(self):
        results = score_candidates(self.CANDIDATES, PARSED_JD, top_n=3)
        assert results[-1]["id"] == "C2"
