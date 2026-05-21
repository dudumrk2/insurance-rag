"""PII redaction for markdown text.

``redact()`` is a pure function: same input -> same output, no file IO. The CLI
(``scripts/redact.py``) handles reading PDFs and writing results.

Two passes:
  1. Regex pass — structured PII (Israeli ID, phone, email, license plate, IBAN).
  2. Known-strings pass — exact substring match of caller-supplied names/IDs.

The returned log records only the redaction *type*, *count*, and a few
post-redaction context windows — never a raw PII value.
"""

import re

# Context-sample settings for the log.
_SAMPLE_WINDOW = 25  # chars on each side of a placeholder
_MAX_SAMPLES = 2     # samples kept per PII type

# Placeholder tokens that replace each PII type.
PLACEHOLDERS = {
    "israeli_id": '[ת"ז]',
    "phone": "[טלפון]",
    "email": "[אימייל]",
    "license_plate": "[רישוי]",
    "iban": "[IBAN]",
    "known_string": "[שם]",
}

# Ordered regex patterns. Structured/longer patterns run before the bare
# 9-digit ID so they claim their text first.
_REGEX_PATTERNS = [
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    # Israeli IBAN: IL + 2 check digits + 19 BBAN digits (23 chars total),
    # optionally space-grouped (e.g. "IL62 0108 0000 0009 9999 999").
    ("iban", re.compile(r"\bIL\d{2}(?:\s?\d){19}\b")),
    ("license_plate", re.compile(r"\b\d{2,3}-\d{3}-\d{2,3}\b")),
    ("phone", re.compile(r"\b0(?:5\d|[2-4]|7\d|8|9)-?\d{7}\b")),
    ("israeli_id", re.compile(r"\b\d{9}\b")),
]


def redact(md_text: str, known_strings: list[str] | None = None) -> tuple[str, dict]:
    """Remove PII from ``md_text``.

    Returns ``(redacted_text, log_dict)``.
    """
    text = md_text
    counts: dict[str, int] = {}

    # Pass 1 — regex.
    for pii_type, pattern in _REGEX_PATTERNS:
        placeholder = PLACEHOLDERS[pii_type]
        text, n = pattern.subn(placeholder, text)
        if n:
            counts[pii_type] = counts.get(pii_type, 0) + n

    # Pass 2 — known strings (exact substring match). Longer strings first so a
    # full name is redacted before any substring of it.
    placeholder = PLACEHOLDERS["known_string"]
    for known in sorted(filter(None, known_strings or []), key=len, reverse=True):
        n = text.count(known)
        if n:
            text = text.replace(known, placeholder)
            counts["known_string"] = counts.get("known_string", 0) + n

    log = {
        "redactions": [
            {
                "type": t,
                "count": c,
                "samples": _context_samples(text, PLACEHOLDERS[t]),
            }
            for t, c in counts.items()
        ],
        "total": sum(counts.values()),
    }
    return text, log


def _context_samples(text: str, placeholder: str) -> list[str]:
    """Return up to ``_MAX_SAMPLES`` context windows around ``placeholder``.

    Windows are sliced from the already-redacted ``text``, so they can only ever
    contain placeholders — never a raw PII value.
    """
    samples: list[str] = []
    start = 0
    while len(samples) < _MAX_SAMPLES:
        idx = text.find(placeholder, start)
        if idx == -1:
            break
        left = max(0, idx - _SAMPLE_WINDOW)
        right = min(len(text), idx + len(placeholder) + _SAMPLE_WINDOW)
        samples.append(text[left:right].replace("\n", " ").strip())
        start = idx + len(placeholder)
    return samples
