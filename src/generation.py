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

from src.config import DEFAULT_FAMILY_ID, DEFAULT_TOP_K
from src.retrieval import retrieve


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
          - answer:   str, the Gemini-generated answer in Hebrew
          - sources:  list[str], anchor strings from retrieved chunks
          - strategy: str, the chunking strategy used
          - question: str, the original question
    """
    # Use injected functions or real ones
    if _retrieve_fn is None:
        _retrieve_fn = retrieve
    if _generate_fn is None:
        _generate_fn = _call_gemini

    # Retrieve relevant chunks
    chunks = _retrieve_fn(
        query=question,
        strategy=strategy,
        family_id=family_id,
        top_k=top_k,
    )

    # Build context from chunks (strip "passage: " prefix)
    context_parts = []
    for chunk in chunks:
        text = chunk["text"]
        # Strip "passage: " prefix if present
        if text.startswith("passage: "):
            text = text[9:]  # len("passage: ") = 9
        context_parts.append(text)

    context = "\n".join(context_parts)

    # Build prompt and generate answer
    prompt = _build_prompt(question, context)
    generated_answer = _generate_fn(prompt)

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
