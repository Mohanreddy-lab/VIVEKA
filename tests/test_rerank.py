"""Tests for src/rerank.py — LLM rerank with mocked model responses."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

CANDIDATES = [
    {
        "id": "C001", "_id": "C001", "title": "Senior Data Engineer",
        "skills": ["Python", "Apache Spark", "Airflow", "AWS"],
        "summary": "8 years building data pipelines on AWS.",
        "embedding_score": 0.91, "skill_score": 0.80,
        "seniority_score": 1.0,  "activity_score": 0.50, "composite_score": 0.742,
        "hidden_gem": False, "skill_evidence": {},
    },
    {
        "id": "C004", "_id": "C004", "title": "Data Engineer",
        "skills": ["Spark", "Airflow", "Python"],
        "summary": "Self-taught. Ships fast.",
        "embedding_score": 0.78, "skill_score": 0.55,
        "seniority_score": 0.55, "activity_score": 0.0, "composite_score": 0.598,
        "hidden_gem": True, "skill_evidence": {},
    },
]

PARSED_JD = {
    "required_skills": ["Python", "Apache Spark", "Airflow", "AWS"],
    "implied_skills":  ["Git", "SQL"],
    "seniority":       "senior",
    "latent_needs":    ["owns pipelines end-to-end"],
}

GOOD_LLM_RESPONSE = '{"analysis": "Candidate has all required skills.", "llm_score": 9, "reason": "Direct match on Spark, Airflow, and AWS.", "confidence": "high", "evidence": ["8 years building data pipelines on AWS"]}'
LOW_CONF_RESPONSE  = '{"analysis": "Limited profile info.", "llm_score": 5, "reason": "Limited evidence: sparse profile.", "confidence": "low", "evidence": []}'
# Evidence that doesn't appear in the candidate profile — should be flagged
HALLUCINATED_EVIDENCE_RESPONSE = '{"llm_score": 8, "reason": "Has 15 years experience.", "confidence": "high", "evidence": ["8 years building data pipelines on AWS", "invented text that is not in profile at all"]}'


def _mock_chain_from_response(text: str):
    msg = MagicMock()
    msg.content = text

    chain = MagicMock()
    chain.invoke.return_value = msg
    return chain


def _patch_rerank(monkeypatch, response_text: str):
    """Patch rerank so ChatPromptTemplate | llm always returns `response_text`."""
    import rerank

    chain = _mock_chain_from_response(response_text)

    # prompt | llm → chain
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=chain)

    # ChatPromptTemplate.from_messages(...) → mock_prompt
    mock_cpt = MagicMock()
    mock_cpt.from_messages.return_value = mock_prompt

    monkeypatch.setattr(rerank, "ChatPromptTemplate", mock_cpt)
    monkeypatch.setattr(rerank, "get_llm", lambda **kw: MagicMock())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseResult:
    def test_valid_json_parsed_correctly(self):
        from rerank import _parse_result
        result = _parse_result(GOOD_LLM_RESPONSE)
        assert result["llm_score"] == 9
        assert result["confidence"] == "high"
        assert "Spark" in result["reason"]

    def test_evidence_parsed(self):
        from rerank import _parse_result
        result = _parse_result(GOOD_LLM_RESPONSE)
        assert "evidence" in result
        assert isinstance(result["evidence"], list)

    def test_evidence_defaults_to_empty_list(self):
        from rerank import _parse_result
        result = _parse_result('{"llm_score": 7, "reason": "ok", "confidence": "medium"}')
        assert result["evidence"] == []

    def test_score_clamped_to_1_10(self):
        from rerank import _parse_result
        assert _parse_result('{"llm_score": 99, "reason": "x", "confidence": "low"}')["llm_score"] == 10
        assert _parse_result('{"llm_score": -5, "reason": "x", "confidence": "low"}')["llm_score"] == 1

    def test_bad_json_returns_fallback(self):
        from rerank import _parse_result
        result = _parse_result("not json")
        assert result["llm_score"] == 5
        assert result["confidence"] == "low"

    def test_analysis_key_present(self):
        from rerank import _parse_result
        result = _parse_result(GOOD_LLM_RESPONSE)
        assert "analysis" in result

    def test_malformed_score_defaults_to_5(self):
        from rerank import _parse_result
        result = _parse_result('{"llm_score": "great", "reason": "x", "confidence": "high"}')
        assert result["llm_score"] == 5


class TestFinalScore:
    def test_high_confidence_uses_llm_score_fully(self):
        from rerank import _final_score
        cfg = {"blend_composite": 0.5, "blend_llm": 0.5}
        score = _final_score(0.6, 10, "high", cfg)
        # effective_llm = 1.0 * (10/10) + 0 * 0.6 = 1.0
        # final = 0.5 * 0.6 + 0.5 * 1.0 = 0.8
        assert abs(score - 0.8) < 0.01

    def test_low_confidence_drifts_toward_composite(self):
        from rerank import _final_score
        cfg = {"blend_composite": 0.5, "blend_llm": 0.5}
        score_high = _final_score(0.6, 10, "high", cfg)
        score_low  = _final_score(0.6, 10, "low",  cfg)
        # Low confidence LLM should be discounted
        assert score_low < score_high

    def test_score_between_zero_and_one(self):
        from rerank import _final_score
        cfg = {"blend_composite": 0.5, "blend_llm": 0.5}
        for composite in [0.0, 0.5, 1.0]:
            for llm in [1, 5, 10]:
                for conf in ["high", "medium", "low"]:
                    s = _final_score(composite, llm, conf, cfg)
                    assert 0.0 <= s <= 1.0, f"Out of range: {s}"


class TestMakeProfileText:
    def test_includes_title_and_skills(self):
        from rerank import _make_profile_text
        profile = {
            "title": "Senior Data Engineer",
            "skills": ["Python", "Spark"],
            "summary": "Builds pipelines.",
        }
        text = _make_profile_text(profile, max_chars=500)
        assert "Senior Data Engineer" in text
        assert "Python" in text

    def test_truncates_at_max_chars(self):
        from rerank import _make_profile_text
        long_profile = {
            "title": "Engineer",
            "summary": "x" * 1000,
        }
        text = _make_profile_text(long_profile, max_chars=100)
        assert len(text) <= 110   # minor slack for the field label

    def test_list_skills_joined(self):
        from rerank import _make_profile_text
        profile = {"skills": ["A", "B", "C"]}
        text = _make_profile_text(profile, max_chars=200)
        assert "A, B, C" in text


class TestRerankedOutput:
    def test_all_candidates_get_final_score(self, monkeypatch):
        _patch_rerank(monkeypatch, GOOD_LLM_RESPONSE)
        from rerank import rerank_candidates
        results = rerank_candidates(CANDIDATES, PARSED_JD, top_n=2)
        for c in results:
            assert "final_score" in c
            assert 0.0 <= c["final_score"] <= 1.0

    def test_results_sorted_by_final_score(self, monkeypatch):
        _patch_rerank(monkeypatch, GOOD_LLM_RESPONSE)
        from rerank import rerank_candidates
        results = rerank_candidates(CANDIDATES, PARSED_JD, top_n=2)
        scores = [c["final_score"] for c in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_n_respected(self, monkeypatch):
        _patch_rerank(monkeypatch, GOOD_LLM_RESPONSE)
        from rerank import rerank_candidates
        results = rerank_candidates(CANDIDATES, PARSED_JD, top_n=1)
        assert len(results) == 1

    def test_llm_score_and_reason_present(self, monkeypatch):
        _patch_rerank(monkeypatch, GOOD_LLM_RESPONSE)
        from rerank import rerank_candidates
        results = rerank_candidates(CANDIDATES, PARSED_JD, top_n=2)
        for c in results:
            assert "llm_score" in c
            assert "reason" in c
            assert "confidence" in c

    def test_evidence_verified_and_unsupported_present(self, monkeypatch):
        _patch_rerank(monkeypatch, GOOD_LLM_RESPONSE)
        from rerank import rerank_candidates
        results = rerank_candidates(CANDIDATES, PARSED_JD, top_n=2)
        for c in results:
            assert "evidence_verified" in c
            assert "evidence_unsupported" in c

    def test_calibrated_confidence_present(self, monkeypatch):
        _patch_rerank(monkeypatch, GOOD_LLM_RESPONSE)
        from rerank import rerank_candidates
        results = rerank_candidates(CANDIDATES, PARSED_JD, top_n=2)
        for c in results:
            assert "calibrated_confidence" in c
            assert 0.0 <= c["calibrated_confidence"] <= 1.0


class TestCitationGrounding:
    """Verify the evidence verifier catches hallucinated citations."""

    def test_real_snippet_lands_in_verified(self):
        from rerank import verify_evidence
        profile_text = "8 years building data pipelines on AWS. Expert in Python."
        evidence = ["8 years building data pipelines on AWS"]
        verified, unsupported = verify_evidence(evidence, profile_text)
        assert "8 years building data pipelines on AWS" in verified
        assert unsupported == []

    def test_fake_snippet_lands_in_unsupported(self):
        from rerank import verify_evidence
        profile_text = "8 years building data pipelines on AWS. Expert in Python."
        evidence = ["invented text that is not in profile at all"]
        verified, unsupported = verify_evidence(evidence, profile_text)
        assert unsupported == ["invented text that is not in profile at all"]
        assert verified == []

    def test_mixed_evidence_split_correctly(self):
        from rerank import verify_evidence
        profile_text = "8 years building data pipelines on AWS."
        evidence = ["8 years building data pipelines on AWS", "hallucinated fact"]
        verified, unsupported = verify_evidence(evidence, profile_text)
        assert len(verified) == 1
        assert len(unsupported) == 1

    def test_confidence_downgraded_on_unsupported(self, monkeypatch):
        """When the model returns hallucinated evidence, confidence must drop."""
        _patch_rerank(monkeypatch, HALLUCINATED_EVIDENCE_RESPONSE)
        from rerank import rerank_candidates
        # C001 profile text: "8 years building data pipelines on AWS"
        # Evidence includes one real + one fake snippet → confidence downgraded
        results = rerank_candidates(CANDIDATES, PARSED_JD, top_n=1, parallel=False)
        c = results[0]
        # Original confidence was "high"; unsupported evidence → "medium"
        assert c["confidence"] in ("medium", "low")
        assert len(c["evidence_unsupported"]) > 0

    def test_empty_evidence_no_confidence_change(self, monkeypatch):
        """Empty evidence array → verifier runs but doesn't downgrade confidence."""
        _patch_rerank(monkeypatch, LOW_CONF_RESPONSE)
        from rerank import rerank_candidates
        results = rerank_candidates(CANDIDATES, PARSED_JD, top_n=1, parallel=False)
        c = results[0]
        assert c["confidence"] == "low"   # unchanged — was already low


class TestCalibratedConfidence:
    def test_high_conf_all_verified_near_1(self):
        from rerank import calibrate_confidence
        score = calibrate_confidence("high", ["real text"], [], 9)
        assert score > 0.7

    def test_low_conf_no_evidence_near_0(self):
        from rerank import calibrate_confidence
        score = calibrate_confidence("low", [], [], 5)
        assert score < 0.5

    def test_high_conf_with_unsupported_lower_than_fully_verified(self):
        from rerank import calibrate_confidence
        verified   = calibrate_confidence("high", ["text"], [],       9)
        unsupported = calibrate_confidence("high", [],       ["fake"], 9)
        assert verified > unsupported

    def test_output_clamped_0_to_1(self):
        from rerank import calibrate_confidence
        for label in ("high", "medium", "low"):
            for llm in (1, 5, 10):
                score = calibrate_confidence(label, [], [], llm)
                assert 0.0 <= score <= 1.0


class TestGracefulDegradation:
    def test_fallback_on_llm_failure(self, monkeypatch):
        """If the LLM is down, rerank_candidates returns composite-ordered results."""
        import rerank

        def _boom(*a, **kw):
            raise ConnectionError("Ollama is not running")

        monkeypatch.setattr(rerank, "rerank_stream_parallel", _boom)
        monkeypatch.setattr(rerank, "rerank_stream", _boom)

        results = rerank.rerank_candidates(CANDIDATES, PARSED_JD, top_n=2, parallel=True)
        assert len(results) == 2
        for c in results:
            assert c["llm_unavailable"] is True
            assert c["final_score"] == c["composite_score"]
