"""Tests for src.redaction.redact() — the core of Step 1.

All PII inputs are synthetic (see conftest.py).
"""

import json

from src.redaction import redact
from tests.conftest import (
    FAKE_EMAIL,
    FAKE_IBAN,
    FAKE_ISRAELI_ID,
    FAKE_LICENSE_PLATE,
    FAKE_NAME,
    FAKE_PHONE_LANDLINE,
    FAKE_PHONE_MOBILE,
)


def test_redacts_israeli_id():
    text = f"בעל תעודת זהות {FAKE_ISRAELI_ID} מבוטח."
    redacted, log = redact(text)
    assert FAKE_ISRAELI_ID not in redacted
    assert "[ת\"ז]" in redacted
    assert log["total"] == 1


def test_redacts_phone_mobile_and_landline():
    text = f"נייד {FAKE_PHONE_MOBILE} ובבית {FAKE_PHONE_LANDLINE}."
    redacted, log = redact(text)
    assert FAKE_PHONE_MOBILE not in redacted
    assert FAKE_PHONE_LANDLINE not in redacted
    assert redacted.count("[טלפון]") == 2


def test_redacts_email():
    text = f"כתובת דוא\"ל: {FAKE_EMAIL} לפניות."
    redacted, log = redact(text)
    assert FAKE_EMAIL not in redacted
    assert "[אימייל]" in redacted


def test_redacts_license_plate():
    text = f"מספר רישוי {FAKE_LICENSE_PLATE} לרכב."
    redacted, log = redact(text)
    assert FAKE_LICENSE_PLATE not in redacted
    assert "[רישוי]" in redacted


def test_redacts_iban():
    text = f"חשבון לזיכוי {FAKE_IBAN} בבנק."
    redacted, log = redact(text)
    assert FAKE_IBAN not in redacted
    assert "[IBAN]" in redacted


def test_redacts_known_string(known_strings):
    text = f"המבוטח {FAKE_NAME} חתם על הפוליסה."
    redacted, log = redact(text, known_strings)
    assert FAKE_NAME not in redacted
    assert "[שם]" in redacted


def test_log_has_samples_but_no_raw_pii(mixed_pii_markdown, known_strings):
    _, log = redact(mixed_pii_markdown, known_strings)
    # Every redaction entry carries at least one context sample.
    assert all(entry.get("samples") for entry in log["redactions"])
    # No raw PII value appears anywhere in the serialized log.
    blob = json.dumps(log, ensure_ascii=False)
    for secret in (
        FAKE_ISRAELI_ID,
        FAKE_PHONE_MOBILE,
        FAKE_PHONE_LANDLINE,
        FAKE_EMAIL,
        FAKE_LICENSE_PLATE,
        FAKE_IBAN,
        FAKE_NAME,
    ):
        assert secret not in blob


def test_log_counts_correct(mixed_pii_markdown, known_strings):
    _, log = redact(mixed_pii_markdown, known_strings)
    by_type = {e["type"]: e["count"] for e in log["redactions"]}
    assert by_type["israeli_id"] == 1
    assert by_type["phone"] == 2
    assert by_type["email"] == 1
    assert by_type["license_plate"] == 1
    assert by_type["iban"] == 1
    assert by_type["known_string"] == 1  # FAKE_NAME once; FAKE_NAME_2 absent
    assert log["total"] == 7


def test_deterministic(mixed_pii_markdown, known_strings):
    assert redact(mixed_pii_markdown, known_strings) == redact(
        mixed_pii_markdown, known_strings
    )


def test_multiple_pii_in_one_doc(mixed_pii_markdown, known_strings):
    redacted, _ = redact(mixed_pii_markdown, known_strings)
    for secret in (
        FAKE_ISRAELI_ID,
        FAKE_PHONE_MOBILE,
        FAKE_PHONE_LANDLINE,
        FAKE_EMAIL,
        FAKE_LICENSE_PLATE,
        FAKE_IBAN,
        FAKE_NAME,
    ):
        assert secret not in redacted
    # Legitimate, non-PII content survives untouched.
    assert "50,000" in redacted
    assert "כיסויים" in redacted


def test_works_without_known_strings():
    text = f"תעודת זהות {FAKE_ISRAELI_ID}."
    redacted, log = redact(text)
    assert FAKE_ISRAELI_ID not in redacted
    assert log["total"] == 1
