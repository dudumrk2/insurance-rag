"""Tests for src.generation — RAG answer generation via Gemini.

All tests use injected _retrieve_fn and _generate_fn so there are no real
API calls or ChromaDB dependencies. Tests verify answer() returns the correct
structure, handles context properly, and gracefully degrades on empty results.
"""

import pytest

from src.generation import answer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_retrieve_fn(query: str, **kwargs) -> list[dict]:
    """Mock retrieve function that returns 2 fake chunks."""
    return [
        {
            "chunk_id": "chunk_1",
            "text": "passage: כיסוי הגניבה עד 50,000 ש\"ח כולל גניבת חלקים.",
            "source_doc": "policy_1.md",
            "score": 0.95,
            "anchor": "כיסוי הגניבה",
            "section": "## כיסויים",
            "family_id": "demo_family_001",
        },
        {
            "chunk_id": "chunk_2",
            "text": "passage: השתתפות המבוטח בתביעה גניבה היא 500 ש\"ח.",
            "source_doc": "policy_1.md",
            "score": 0.88,
            "anchor": "השתתפות",
            "section": "## דמי השתתפות",
            "family_id": "demo_family_001",
        },
    ]


def _fake_generate_fn(prompt: str) -> str:
    """Mock generate function that returns a fake Hebrew answer."""
    return "כיסוי הגניבה הוא עד 50,000 ש\"ח עם השתתפות של 500 ש\"ח."


def _fake_retrieve_fn_empty(query: str, **kwargs) -> list[dict]:
    """Mock retrieve function that returns no chunks."""
    return []


# ---------------------------------------------------------------------------
# Tests: Basic Structure
# ---------------------------------------------------------------------------


def test_answer_returns_dict():
    """answer() returns a dictionary."""
    result = answer(
        question="מה כוסה תחת הגניבה?",
        _retrieve_fn=_fake_retrieve_fn,
        _generate_fn=_fake_generate_fn,
    )
    assert isinstance(result, dict)


def test_answer_has_required_keys():
    """answer() dict has all required keys: answer, sources, strategy, question."""
    result = answer(
        question="מה כוסה תחת הגניבה?",
        _retrieve_fn=_fake_retrieve_fn,
        _generate_fn=_fake_generate_fn,
    )
    required_keys = {"answer", "sources", "strategy", "question"}
    assert required_keys.issubset(result.keys()), (
        f"Missing keys. Got: {result.keys()}, expected: {required_keys}"
    )


def test_answer_field_is_nonempty_string():
    """The 'answer' field is a non-empty string."""
    result = answer(
        question="מה כוסה תחת הגניבה?",
        _retrieve_fn=_fake_retrieve_fn,
        _generate_fn=_fake_generate_fn,
    )
    assert isinstance(result["answer"], str)
    assert len(result["answer"]) > 0


def test_sources_field_is_list():
    """The 'sources' field is a list."""
    result = answer(
        question="מה כוסה תחת הגניבה?",
        _retrieve_fn=_fake_retrieve_fn,
        _generate_fn=_fake_generate_fn,
    )
    assert isinstance(result["sources"], list)


def test_sources_contains_anchor_strings():
    """Each source is an anchor string from retrieved chunks."""
    result = answer(
        question="מה כוסה תחת הגניבה?",
        _retrieve_fn=_fake_retrieve_fn,
        _generate_fn=_fake_generate_fn,
    )
    # Mock returns 2 chunks with anchors "כיסוי הגניבה" and "השתתפות"
    assert result["sources"] == ["כיסוי הגניבה", "השתתפות"]


def test_strategy_field_echoes_input():
    """The 'strategy' field matches the input strategy."""
    result = answer(
        question="מה כוסה תחת הגניבה?",
        strategy="section_aware",
        _retrieve_fn=_fake_retrieve_fn,
        _generate_fn=_fake_generate_fn,
    )
    assert result["strategy"] == "section_aware"

    result = answer(
        question="מה כוסה תחת הגניבה?",
        strategy="fixed",
        _retrieve_fn=_fake_retrieve_fn,
        _generate_fn=_fake_generate_fn,
    )
    assert result["strategy"] == "fixed"


def test_question_field_echoes_input():
    """The 'question' field echoes the input question."""
    q = "מה כוסה תחת הגניבה?"
    result = answer(
        question=q,
        _retrieve_fn=_fake_retrieve_fn,
        _generate_fn=_fake_generate_fn,
    )
    assert result["question"] == q


# ---------------------------------------------------------------------------
# Tests: Context Handling
# ---------------------------------------------------------------------------


def test_context_passed_to_generator_excludes_passage_prefix():
    """The context passed to _generate_fn must NOT contain 'passage: ' prefix."""
    captured_prompt = None

    def capture_prompt(prompt: str) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return "דמה תשובה"

    result = answer(
        question="מה כוסה?",
        _retrieve_fn=_fake_retrieve_fn,
        _generate_fn=capture_prompt,
    )

    # The prompt should contain the chunk text WITHOUT the "passage: " prefix
    assert captured_prompt is not None
    assert "כיסוי הגניבה עד 50,000 ש\"ח כולל גניבת חלקים." in captured_prompt
    # Make sure the passage prefix is NOT in the prompt
    assert "passage: כיסוי הגניבה עד 50,000" not in captured_prompt


def test_context_includes_both_chunks():
    """Context passed to generator includes text from all retrieved chunks."""
    captured_prompt = None

    def capture_prompt(prompt: str) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return "דמה תשובה"

    answer(
        question="מה כוסה?",
        _retrieve_fn=_fake_retrieve_fn,
        _generate_fn=capture_prompt,
    )

    # Both chunk texts (without prefix) should be in the prompt
    assert "כיסוי הגניבה עד 50,000 ש\"ח כולל גניבת חלקים." in captured_prompt
    assert "השתתפות המבוטח בתביעה גניבה היא 500 ש\"ח." in captured_prompt


def test_context_includes_question():
    """Context passed to generator includes the question."""
    captured_prompt = None

    def capture_prompt(prompt: str) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return "דמה תשובה"

    q = "מה כוסה תחת הגניבה?"
    answer(
        question=q,
        _retrieve_fn=_fake_retrieve_fn,
        _generate_fn=capture_prompt,
    )

    # Question should be in the prompt
    assert q in captured_prompt


# ---------------------------------------------------------------------------
# Tests: Graceful Degradation
# ---------------------------------------------------------------------------


def test_empty_retrieval_returns_dict():
    """When retrieve() returns no chunks, answer() still returns a valid dict."""
    result = answer(
        question="מה כוסה?",
        _retrieve_fn=_fake_retrieve_fn_empty,
        _generate_fn=_fake_generate_fn,
    )

    # Should still have required keys
    assert isinstance(result, dict)
    assert "answer" in result
    assert "sources" in result
    assert "strategy" in result
    assert "question" in result

    # sources should be empty
    assert result["sources"] == []


def test_empty_retrieval_passes_minimal_context():
    """When no chunks are retrieved, pass empty context but still ask question."""
    captured_prompt = None

    def capture_prompt(prompt: str) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return "לא יכול לענות"

    answer(
        question="שאלה מסוימת?",
        _retrieve_fn=_fake_retrieve_fn_empty,
        _generate_fn=capture_prompt,
    )

    # Prompt should still contain the question
    assert "שאלה מסוימת?" in captured_prompt
    # But no chunk content
    assert "כיסוי" not in captured_prompt


# ---------------------------------------------------------------------------
# Tests: Injection Verification
# ---------------------------------------------------------------------------


def test_uses_injected_retrieve_function():
    """answer() uses the injected _retrieve_fn instead of real retrieve()."""
    call_count = 0

    def counting_retrieve(query: str, **kwargs) -> list[dict]:
        nonlocal call_count
        call_count += 1
        return _fake_retrieve_fn(query, **kwargs)

    answer(
        question="שאלה?",
        _retrieve_fn=counting_retrieve,
        _generate_fn=_fake_generate_fn,
    )

    # Should have been called exactly once
    assert call_count == 1


def test_uses_injected_generate_function():
    """answer() uses the injected _generate_fn instead of real Gemini API."""
    call_count = 0

    def counting_generate(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "תשובה דמיוני"

    answer(
        question="שאלה?",
        _retrieve_fn=_fake_retrieve_fn,
        _generate_fn=counting_generate,
    )

    # Should have been called exactly once
    assert call_count == 1


# ---------------------------------------------------------------------------
# Tests: Default Behavior (Integration Hints)
# ---------------------------------------------------------------------------


def test_answer_passes_correct_parameters_to_retrieve():
    """answer() passes question, strategy, family_id, top_k to _retrieve_fn."""
    captured_kwargs = {}

    def capturing_retrieve(query: str, **kwargs) -> list[dict]:
        captured_kwargs.update(kwargs)
        return []

    answer(
        question="שאלה מיוחדת?",
        family_id="custom_family_123",
        strategy="fixed",
        top_k=3,
        _retrieve_fn=capturing_retrieve,
        _generate_fn=_fake_generate_fn,
    )

    # Should pass these to retrieve
    assert captured_kwargs["family_id"] == "custom_family_123"
    assert captured_kwargs["strategy"] == "fixed"
    assert captured_kwargs["top_k"] == 3
