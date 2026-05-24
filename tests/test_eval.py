"""Unit tests for eval/run_eval.py metric helpers.

These tests cover the pure-computation functions only — no LLM calls,
no disk I/O, no ChromaDB.  All heavy retrieval is mocked via the
dependency-injection hooks already present in src/retrieval.py.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# Make sure the project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))

import run_eval  # noqa: E402 — must be after sys.path manipulation


# ---------------------------------------------------------------------------
# _gold_rank
# ---------------------------------------------------------------------------

def _make_result(raw_text: str) -> dict:
    """Build a minimal retrieval result dict."""
    return {
        "chunk_id": "x",
        "text": f"passage: {raw_text}",
        "source_doc": "doc",
        "score": 1.0,
        "anchor": raw_text[:80],
        "section": "",
        "family_id": "demo",
    }


class TestGoldRank:
    def test_exact_anchor_match_at_rank_1(self):
        anchor = "## כיסוי גניבה\n\nהפוליסה מכסה גניבה עד 50,000"
        results = [_make_result(anchor + " שקל")]
        assert run_eval._gold_rank(results, anchor) == 1

    def test_anchor_at_rank_3(self):
        anchor = "הפרנשייז הוא 3,000"
        results = [
            _make_result("טקסט לא רלוונטי"),
            _make_result("עוד טקסט"),
            _make_result("פסקה: " + anchor + " ש\"ח נוסף"),
        ]
        assert run_eval._gold_rank(results, anchor) == 3

    def test_not_found_returns_none(self):
        anchor = "טקסט שלא קיים"
        results = [_make_result("משהו אחר לגמרי")]
        assert run_eval._gold_rank(results, anchor) is None

    def test_empty_results_returns_none(self):
        assert run_eval._gold_rank([], "כלשהו") is None

    def test_anchor_with_leading_whitespace_stripped(self):
        anchor = "  הפוליסה מכסה  "
        results = [_make_result("הפוליסה מכסה נזק")]
        # _gold_rank strips the anchor before checking
        assert run_eval._gold_rank(results, anchor) == 1


# ---------------------------------------------------------------------------
# _hit_at_k
# ---------------------------------------------------------------------------

class TestHitAtK:
    def test_all_hits_at_1(self):
        ranks = [1, 1, 1]
        assert run_eval._hit_at_k(ranks, k=1) == pytest.approx(1.0)

    def test_no_hits(self):
        ranks = [None, None, None]
        assert run_eval._hit_at_k(ranks, k=5) == pytest.approx(0.0)

    def test_partial_hits_at_3(self):
        # 2 hits at rank ≤3, 1 miss (rank 5 > 3), 1 none
        ranks = [1, 3, 5, None]
        assert run_eval._hit_at_k(ranks, k=3) == pytest.approx(2 / 4)

    def test_rank_5_counts_for_hit_at_5_not_at_3(self):
        ranks = [5]
        assert run_eval._hit_at_k(ranks, k=5) == pytest.approx(1.0)
        assert run_eval._hit_at_k(ranks, k=3) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _mrr
# ---------------------------------------------------------------------------

class TestMrr:
    def test_perfect_mrr(self):
        ranks = [1, 1, 1]
        assert run_eval._mrr(ranks) == pytest.approx(1.0)

    def test_all_misses(self):
        ranks = [None, None]
        assert run_eval._mrr(ranks) == pytest.approx(0.0)

    def test_mixed(self):
        # (1/1 + 1/2 + 0) / 3
        ranks = [1, 2, None]
        expected = (1.0 + 0.5 + 0.0) / 3
        assert run_eval._mrr(ranks) == pytest.approx(expected)

    def test_single_rank_2(self):
        ranks = [2]
        assert run_eval._mrr(ranks) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# _eval_run  (integration: mocked retrieve via dependency injection)
# ---------------------------------------------------------------------------

class TestEvalRun:
    """Verify _eval_run computes metrics correctly with a fake retriever."""

    def _fake_retrieve(self, results_by_question: dict):
        """Return a retrieve_fn that returns preset results for each question."""
        def _retrieve(query, strategy, family_id, top_k, collection=None):
            return results_by_question.get(query, [])
        return _retrieve

    def test_perfect_retrieval(self):
        gold = [
            {"id": "q001", "question": "שאלה אחת", "answer": "תשובה", "anchor": "אנקור אחד", "source_doc": "doc"},
        ]
        anchor_text = "אנקור אחד — הטקסט המלא של הצ'אנק"
        results = [_make_result(anchor_text)]
        retrieve_fn = self._fake_retrieve({"שאלה אחת": results})

        metrics = run_eval._eval_run(
            gold=gold,
            strategy="section_aware",
            family_id="demo",
            top_k=5,
            retrieve_fn=retrieve_fn,
        )
        assert metrics["hit@1"] == pytest.approx(1.0)
        assert metrics["hit@3"] == pytest.approx(1.0)
        assert metrics["hit@5"] == pytest.approx(1.0)
        assert metrics["mrr"] == pytest.approx(1.0)

    def test_zero_retrieval(self):
        gold = [
            {"id": "q001", "question": "שאלה", "answer": "תשובה", "anchor": "אנקור", "source_doc": "doc"},
        ]
        retrieve_fn = self._fake_retrieve({})  # returns nothing

        metrics = run_eval._eval_run(
            gold=gold,
            strategy="fixed",
            family_id="demo",
            top_k=5,
            retrieve_fn=retrieve_fn,
        )
        assert metrics["hit@1"] == pytest.approx(0.0)
        assert metrics["mrr"] == pytest.approx(0.0)
