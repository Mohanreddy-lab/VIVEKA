"""Tests for src/jd_parser.py — structured JD extraction with mocked LLM."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Mock helper
# Patches ChatPromptTemplate + get_llm inside jd_parser so no Ollama needed.
# ---------------------------------------------------------------------------

def _patch_jd_parser(monkeypatch, response_text: str):
    """
    Make jd_parser.parse_jd() return `response_text` as the model output.
    Patches both get_llm and ChatPromptTemplate inside the jd_parser module.
    """
    import jd_parser

    # Build a fake chain whose .invoke() returns an object with .content = text
    mock_response = MagicMock()
    mock_response.content = response_text

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    # prompt | llm → mock_chain
    mock_prompt = MagicMock()
    mock_prompt.__or__ = MagicMock(return_value=mock_chain)

    # ChatPromptTemplate.from_messages(...) → mock_prompt
    mock_cpt = MagicMock()
    mock_cpt.from_messages.return_value = mock_prompt

    monkeypatch.setattr(jd_parser, "ChatPromptTemplate", mock_cpt)
    monkeypatch.setattr(jd_parser, "get_llm", lambda **kw: MagicMock())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParseJd:
    GOOD_RESPONSE = """{
        "required_skills": ["Python", "Apache Spark", "Airflow", "AWS"],
        "implied_skills":  ["Git", "Linux", "Docker"],
        "seniority":       "senior",
        "latent_needs":    ["owns outcomes end-to-end", "works under ambiguity"]
    }"""

    def test_returns_all_four_keys(self, monkeypatch):
        _patch_jd_parser(monkeypatch, self.GOOD_RESPONSE)
        from jd_parser import parse_jd
        result = parse_jd("Some JD text")
        assert set(result.keys()) == {"required_skills", "implied_skills", "seniority", "latent_needs"}

    def test_required_skills_are_list(self, monkeypatch):
        _patch_jd_parser(monkeypatch, self.GOOD_RESPONSE)
        from jd_parser import parse_jd
        result = parse_jd("Some JD text")
        assert isinstance(result["required_skills"], list)
        assert "Python" in result["required_skills"]

    def test_seniority_is_lowercase_string(self, monkeypatch):
        _patch_jd_parser(monkeypatch, self.GOOD_RESPONSE)
        from jd_parser import parse_jd
        result = parse_jd("Some JD text")
        assert result["seniority"] == "senior"

    def test_malformed_json_falls_back_gracefully(self, monkeypatch):
        _patch_jd_parser(monkeypatch, "This is not JSON at all.")
        from jd_parser import parse_jd
        result = parse_jd("Some JD text")
        assert "required_skills" in result
        assert isinstance(result["required_skills"], list)
        assert result["seniority"] in ("unknown", "junior", "mid", "senior", "lead")

    def test_empty_jd_returns_safe_defaults(self, monkeypatch):
        empty_resp = '{"required_skills": [], "implied_skills": [], "seniority": "unknown", "latent_needs": []}'
        _patch_jd_parser(monkeypatch, empty_resp)
        from jd_parser import parse_jd
        result = parse_jd("")
        assert result["required_skills"] == []
        assert result["seniority"] == "unknown"

    def test_markdown_fenced_json_is_parsed(self, monkeypatch):
        fenced = '```json\n{"required_skills": ["Python"], "implied_skills": ["Git"], "seniority": "mid", "latent_needs": []}\n```'
        _patch_jd_parser(monkeypatch, fenced)
        from jd_parser import parse_jd
        result = parse_jd("Some JD text")
        assert "Python" in result["required_skills"]

    def test_embedded_json_block_is_extracted(self, monkeypatch):
        with_preamble = 'Here is the JSON: {"required_skills": ["SQL"], "implied_skills": [], "seniority": "junior", "latent_needs": []}'
        _patch_jd_parser(monkeypatch, with_preamble)
        from jd_parser import parse_jd
        result = parse_jd("Some JD text")
        assert "SQL" in result["required_skills"]

    def test_string_skill_coerced_to_list(self, monkeypatch):
        _patch_jd_parser(monkeypatch, '{"required_skills": "Python", "implied_skills": [], "seniority": "senior", "latent_needs": []}')
        from jd_parser import parse_jd
        result = parse_jd("Some JD text")
        assert isinstance(result["required_skills"], list)

    def test_implied_skills_are_list(self, monkeypatch):
        _patch_jd_parser(monkeypatch, self.GOOD_RESPONSE)
        from jd_parser import parse_jd
        result = parse_jd("Some JD text")
        assert isinstance(result["implied_skills"], list)
        assert "Git" in result["implied_skills"]
