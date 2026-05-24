"""RAG answer generation via Gemini.

answer() retrieves relevant chunks and generates a Gemini response using them
as context. Designed for multi-tenant deployment (filters by family_id).

Usage::

    from src.generation import answer

    result = answer(
        question="מה כוסה תחת הגניבה?",
        family_id="demo_family_001",
        strategy="section_aware",
    )
    print(result["answer"])
    print(result["sources"])
"""

from __future__ import annotations

import re

from src.config import DEFAULT_FAMILY_ID, DEFAULT_TOP_K
from src.retrieval import retrieve
from src.utils import get_logger

logger = get_logger(__name__)

# Returned when the generator yields no text (e.g. Gemini safety block or a
# non-STOP finish reason makes response.text None). Keeps the answer contract
# a non-empty string instead of leaking None to callers.
NO_ANSWER_FALLBACK = "מצטער, לא הצלחתי להפיק תשובה על בסיס המידע הזמין."

# Patterns that may indicate prompt injection attempts
_INJECTION_PATTERNS = [
    r"ignore.*your.*instruction",
    r"forget.*your.*system",
    r"override.*prompt",
    r"disregard.*previous",
    r"pretend.*you.*are",
    r"act.*as.*if",
    r"bypass.*security",
    r"jailbreak",
]
_INJECTION_REGEX = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)

# Only scan the start of the question. Caps regex backtracking cost on
# untrusted input (the patterns use multiple ``.*``) and is where an injection
# preamble would appear anyway.
_INJECTION_SCAN_LIMIT = 2000


def _check_for_injection(question: str) -> None:
    """Log a warning if question contains patterns suggesting prompt injection."""
    if _INJECTION_REGEX.search(question[:_INJECTION_SCAN_LIMIT]):
        logger.warning(f"Potential prompt injection detected in question: {question[:100]!r}")


def answer(
    question: str,
    family_id: str = DEFAULT_FAMILY_ID,
    strategy: str = "section_aware",
    top_k: int = DEFAULT_TOP_K,
    _retrieve_fn=None,  # injected for tests
    _generate_fn=None,  # injected for tests
) -> dict:
    """Generate an answer to a question using RAG.

    Retrieves top-k chunks relevant to the question, builds context from them,
    and calls Gemini to generate a Hebrew answer.

    Args:
        question:     The question in Hebrew or English.
        family_id:    Multi-tenant family ID to filter retrieved chunks.
        strategy:     Chunking strategy: ``"fixed"`` or ``"section_aware"``.
        top_k:        Maximum chunks to retrieve for context.
        _retrieve_fn: Function to retrieve chunks (injected for tests).
                      Signature: (query, strategy, family_id, top_k) -> list[dict]
        _generate_fn: Function to generate answer (injected for tests).
                      Signature: (prompt) -> str

    Returns:
        Dict with keys:
          - answer:   str, the Gemini-generated answer in Hebrew. Falls back to
                      a default message if generation fails (e.g., safety block or None response).
          - sources:  list[str], anchor strings from retrieved chunks
          - strategy: str, the chunking strategy used
          - question: str, the original question
    """
    # Use injected functions or real ones
    if _retrieve_fn is None:
        _retrieve_fn = retrieve
    if _generate_fn is None:
        _generate_fn = _call_gemini

    # Check for potential prompt injection attempts
    _check_for_injection(question)

    # Retrieve relevant chunks
    chunks = _retrieve_fn(
        query=question,
        strategy=strategy,
        family_id=family_id,
        top_k=top_k,
    )

    # Build context from chunks (strip "passage: " prefix)
    context_parts = [chunk["text"].removeprefix("passage: ") for chunk in chunks]
    context = "\n".join(context_parts)

    # Build prompt and generate answer. The generator may return None (e.g. a
    # Gemini safety block), so fall back to a fixed message to honor the
    # non-empty-string answer contract.
    prompt = _build_prompt(question, context)
    generated_answer = _generate_fn(prompt) or NO_ANSWER_FALLBACK

    # Extract sources (anchor strings)
    sources = [chunk["anchor"] for chunk in chunks]

    return {
        "answer": generated_answer,
        "sources": sources,
        "strategy": strategy,
        "question": question,
    }


def _build_prompt(question: str, context: str) -> str:
    """Build the prompt for Gemini.

    Args:
        question: The user question.
        context:  Concatenated chunk texts (without "passage: " prefix).

    Returns:
        A formatted prompt string.
    """
    system_message = "אתה עוזר המתמחה בפוליסות ביטוח. ענה בעברית בלבד על בסיס ההקשר שסופק."
    user_message = f"הקשר:\n{context}\n\nשאלה: {question}"

    return f"System: {system_message}\n\nUser: {user_message}"


def _call_gemini(prompt: str) -> str:
    """Call Gemini API with the prompt.

    Args:
        prompt: The full prompt string.

    Returns:
        The model's response text.
    """
    from src.config import GENERATION_MODEL

    from google import genai

    client = genai.Client()
    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
    )
    return response.text
