"""
pii.py — PII Firewall

Strips identity fields from candidate profiles BEFORE they enter scoring
or LLM rerank. Ranking is based on skills and evidence alone.

Toggle: MANTHAN_PII_FIREWALL=off to disable (default: on).
"""

import os
import re

# Fields that carry identity, not skill — remove entirely before scoring
_IDENTITY_FIELDS = frozenset({
    "name", "full_name", "first_name", "last_name", "display_name",
    "email", "email_address", "phone", "mobile", "tel", "telephone",
    "gender", "sex",
    "age", "dob", "date_of_birth", "birth_date", "birthdate", "year_of_birth",
    "photo", "image", "picture", "avatar", "profile_picture", "profile_photo",
    "address", "street", "city", "state", "zip", "postal_code", "pincode",
    "location", "country", "nationality", "region", "hometown",
    "marital_status", "religion", "caste", "ethnicity", "race",
    "linkedin", "twitter", "facebook", "instagram", "github_url",
    "personal_website", "portfolio_url",
})

# Free-text fields that might contain inline email/phone
_TEXT_FIELDS = frozenset({"summary", "bio", "about", "description", "overview", "objective"})

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"(?<!\d)[\+\(]?\d[\d\s\-\(\)\.]{7,}\d(?!\d)")


def redact_profile(profile: dict) -> dict:
    """
    Return a copy of profile with all PII fields removed.
    Also masks inline email/phone inside free-text fields.
    The 'id' field is always preserved so downstream output stays linked.
    """
    out = {}
    for key, val in profile.items():
        if key.lower() in _IDENTITY_FIELDS:
            continue  # drop entirely — not passed to scoring or LLM
        if isinstance(val, str) and key.lower() in _TEXT_FIELDS:
            # Mask leftover contact info in narrative text
            val = _EMAIL_RE.sub("[redacted]", val)
            val = _PHONE_RE.sub("[redacted]", val)
        out[key] = val
    return out


def is_firewall_on() -> bool:
    """Return True if PII firewall is enabled (default: on)."""
    return os.getenv("MANTHAN_PII_FIREWALL", "on").lower() not in ("off", "0", "false", "no")
