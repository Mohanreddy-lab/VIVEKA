"""Tests for src/pii.py — PII firewall."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pii import redact_profile, is_firewall_on


DIRTY_PROFILE = {
    "id":          "C001",
    "name":        "Jane Smith",
    "first_name":  "Jane",
    "last_name":   "Smith",
    "email":       "jane@example.com",
    "phone":       "+91-9876543210",
    "gender":      "female",
    "age":         28,
    "location":    "Bangalore",
    "nationality": "Indian",
    "title":       "Senior Data Engineer",
    "skills":      ["Python", "Spark", "Airflow"],
    "summary":     "Jane Smith is a great engineer. Contact: jane@example.com or +91-9876543210",
    "github_repos": 42,
    "experience":  "5 years building pipelines.",
}


class TestRedactProfile:
    def test_identity_fields_removed(self):
        out = redact_profile(DIRTY_PROFILE)
        for field in ("name", "first_name", "last_name", "email", "phone",
                      "gender", "age", "location", "nationality"):
            assert field not in out, f"PII field '{field}' not removed"

    def test_id_always_kept(self):
        assert redact_profile(DIRTY_PROFILE)["id"] == "C001"

    def test_professional_fields_kept(self):
        out = redact_profile(DIRTY_PROFILE)
        assert out["title"] == "Senior Data Engineer"
        assert out["skills"] == ["Python", "Spark", "Airflow"]
        assert out["github_repos"] == 42
        assert out["experience"] == "5 years building pipelines."

    def test_email_in_summary_masked(self):
        out = redact_profile(DIRTY_PROFILE)
        assert "jane@example.com" not in out["summary"]
        assert "[redacted]" in out["summary"]

    def test_phone_in_summary_masked(self):
        out = redact_profile(DIRTY_PROFILE)
        assert "+91-9876543210" not in out["summary"]
        assert "[redacted]" in out["summary"]

    def test_original_dict_not_mutated(self):
        redact_profile(DIRTY_PROFILE)
        assert DIRTY_PROFILE["name"] == "Jane Smith"
        assert DIRTY_PROFILE["email"] == "jane@example.com"

    def test_missing_fields_not_error(self):
        # Profile without any PII — should pass through unchanged (minus no PII)
        clean = {"id": "C002", "title": "Engineer", "skills": ["Python"]}
        out = redact_profile(clean)
        assert out == clean

    def test_non_text_fields_not_regex_scanned(self):
        profile = {"id": "C003", "github_repos": 42, "skills": ["Python"]}
        out = redact_profile(profile)
        assert out["github_repos"] == 42


class TestFirewallToggle:
    def test_default_is_on(self, monkeypatch):
        monkeypatch.delenv("MANTHAN_PII_FIREWALL", raising=False)
        assert is_firewall_on() is True

    def test_off_disables(self, monkeypatch):
        monkeypatch.setenv("MANTHAN_PII_FIREWALL", "off")
        assert is_firewall_on() is False

    def test_zero_disables(self, monkeypatch):
        monkeypatch.setenv("MANTHAN_PII_FIREWALL", "0")
        assert is_firewall_on() is False

    def test_on_enables(self, monkeypatch):
        monkeypatch.setenv("MANTHAN_PII_FIREWALL", "on")
        assert is_firewall_on() is True
