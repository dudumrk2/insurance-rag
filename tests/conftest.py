"""Shared test fixtures.

All PII values here are SYNTHETIC — invented for testing, not real. They never
touch the real corpus or get committed as redacted output.
"""

import pytest

# --- Synthetic PII samples (fake) ------------------------------------------
FAKE_ISRAELI_ID = "987654321"
FAKE_PHONE_MOBILE = "050-1112233"
FAKE_PHONE_LANDLINE = "03-7654321"
FAKE_EMAIL = "test.person@example.com"
FAKE_LICENSE_PLATE = "12-345-67"
FAKE_IBAN = "IL620108000000099999999"
FAKE_NAME = "פלוני אלמוני"
FAKE_NAME_2 = "אלמונית פלונית"


@pytest.fixture
def known_strings() -> list[str]:
    """Known PII strings a caller would supply (names, IDs)."""
    return [FAKE_NAME, FAKE_NAME_2]


@pytest.fixture
def mixed_pii_markdown() -> str:
    """A markdown doc containing one of every PII type plus a known name."""
    return (
        "## פרטי המבוטח\n"
        f"שם המבוטח: {FAKE_NAME}, בעל תעודת זהות {FAKE_ISRAELI_ID}.\n"
        f"טלפון נייד: {FAKE_PHONE_MOBILE}, טלפון בבית: {FAKE_PHONE_LANDLINE}.\n"
        f"כתובת דוא\"ל: {FAKE_EMAIL}.\n"
        f"מספר רישוי הרכב: {FAKE_LICENSE_PLATE}.\n"
        f"חשבון לזיכוי: {FAKE_IBAN}.\n"
        "## כיסויים\n"
        "הפוליסה מכסה נזקי גניבה עד 50,000 ש\"ח.\n"
    )
