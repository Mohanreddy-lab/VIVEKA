"""Shared fixtures for all VIVEKA tests."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


@pytest.fixture
def sample_profiles():
    return [
        {
            "id": "C001", "title": "Senior Data Engineer",
            "skills": ["Python", "Apache Spark", "Airflow", "AWS", "dbt", "SQL"],
            "summary": "8 years building large-scale data pipelines on AWS. Led Hadoop → S3 migration.",
            "github_repos": 42,
        },
        {
            "id": "C002", "title": "Junior Frontend Developer",
            "skills": ["React", "TypeScript", "CSS", "Figma"],
            "summary": "2 years building web UIs. Passionate about accessibility.",
            "github_repos": 8,
        },
        {
            "id": "C003", "title": "Data Analyst",
            "skills": ["SQL", "Python", "Tableau", "Excel"],
            "summary": "Analyst at a fintech. Strong SQL and storytelling. No pipeline experience.",
        },
        {
            "id": "C004", "title": "Data Engineer",
            "skills": ["Spark", "Airflow", "Python", "S3"],
            "summary": "Built batch pipelines on AWS. Self-taught. Ships fast, works well under ambiguity.",
        },
        {
            "id": "C005", "title": "ML Engineer",
            "skills": ["Python", "PyTorch", "Spark", "Kubernetes", "MLflow"],
            "summary": "Deploys ML models at scale. Feature stores on AWS.",
            "github_repos": 25,
        },
    ]


@pytest.fixture
def sample_parsed_jd():
    return {
        "required_skills": ["Python", "Apache Spark", "Airflow", "AWS"],
        "implied_skills":  ["Git", "SQL", "Linux", "dbt"],
        "seniority":       "senior",
        "latent_needs":    ["owns pipelines end-to-end", "works under ambiguity"],
    }


@pytest.fixture
def sample_ranked():
    return [
        {
            "id": "C001", "title": "Senior Data Engineer",
            "final_score": 0.838, "llm_score": 9, "confidence": "high",
            "hidden_gem": False, "reason": "8 years on AWS Spark pipelines; direct match.",
            "skill_score": 0.75, "seniority_score": 1.0, "activity_score": 0.80,
            "composite_score": 0.727, "embedding_score": 0.91,
            "skill_evidence": {
                "required_matched": ["Python", "Airflow", "AWS"],
                "required_missing": ["Apache Spark"],
                "implied_matched":  ["SQL"],
                "implied_missing":  ["Git", "Linux", "dbt"],
            },
        },
        {
            "id": "C004", "title": "Data Engineer",
            "final_score": 0.682, "llm_score": 7, "confidence": "medium",
            "hidden_gem": True,
            "reason": "Limited evidence: self-taught, ships Spark + Airflow on AWS.",
            "skill_score": 0.50, "seniority_score": 0.55, "activity_score": 0.0,
            "composite_score": 0.614, "embedding_score": 0.78,
            "skill_evidence": {
                "required_matched": ["Python", "Airflow", "AWS"],
                "required_missing": ["Apache Spark"],
                "implied_matched":  [],
                "implied_missing":  ["Git", "SQL", "Linux", "dbt"],
            },
        },
    ]
