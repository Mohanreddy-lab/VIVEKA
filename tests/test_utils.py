"""Tests for src/utils.py — shared JSON parsing helpers."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import try_json, safe_parse_json, as_list


class TestTryJson:
    def test_valid_json(self):
        assert try_json('{"a": 1}') == {"a": 1}

    def test_invalid_json_returns_none(self):
        assert try_json("not json") is None

    def test_empty_string_returns_none(self):
        assert try_json("") is None

    def test_none_input_returns_none(self):
        assert try_json(None) is None


class TestSafeParseJson:
    FALLBACK = {"ok": False}

    def test_clean_json(self):
        result = safe_parse_json('{"required_skills": ["Python"]}', self.FALLBACK)
        assert result["required_skills"] == ["Python"]

    def test_strips_markdown_fences(self):
        raw = '```json\n{"a": 1}\n```'
        assert safe_parse_json(raw, self.FALLBACK) == {"a": 1}

    def test_extracts_embedded_json_block(self):
        raw = 'Here is the result: {"a": 2} and some trailing text.'
        assert safe_parse_json(raw, self.FALLBACK) == {"a": 2}

    def test_returns_fallback_on_garbage(self):
        result = safe_parse_json("total garbage %%##", self.FALLBACK)
        assert result is self.FALLBACK

    def test_returns_fallback_on_empty(self):
        result = safe_parse_json("", self.FALLBACK)
        assert result is self.FALLBACK


class TestAsList:
    def test_list_passthrough(self):
        assert as_list(["a", "b"]) == ["a", "b"]

    def test_string_wraps_to_list(self):
        assert as_list("Python") == ["Python"]

    def test_none_returns_empty(self):
        assert as_list(None) == []

    def test_empty_string_returns_empty(self):
        assert as_list("") == []

    def test_int_returns_empty(self):
        assert as_list(42) == []
