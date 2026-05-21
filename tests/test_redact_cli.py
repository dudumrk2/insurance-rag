"""Tests for the redact CLI's pure logic (load_known_strings)."""

from scripts import redact
from src import config


def test_load_known_strings_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "KNOWN_PII_FILE", tmp_path / "absent.json")
    assert redact.load_known_strings() == []


def test_load_known_strings_reads_and_filters(monkeypatch, tmp_path):
    pii = tmp_path / "known_pii.json"
    pii.write_text(
        '{"_comment": "ignore me", "strings": ["שם", "", "  ", "123456789"]}',
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "KNOWN_PII_FILE", pii)
    # blank / whitespace-only entries dropped; _comment ignored
    assert redact.load_known_strings() == ["שם", "123456789"]
