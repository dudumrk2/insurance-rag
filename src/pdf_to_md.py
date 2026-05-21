"""PDF -> Markdown conversion via Docling.

Docling is a heavy, ML-based dependency, so it is imported lazily inside
``convert()`` — importing this module does not require Docling to be installed.
Install it with ``pip install -e ".[pdf]"`` before running the conversion.

This module is not unit-tested (Docling is slow and one-time); it is exercised
manually by ``scripts/redact.py`` once real PDFs are added to ``data/raw/``.
"""

from pathlib import Path


class DoclingConversionError(Exception):
    """Raised when Docling fails to convert a PDF. The CLI catches this and
    skips the offending file, continuing with the rest of the corpus."""


def convert(pdf_path: Path) -> str:
    """Convert a single PDF to Markdown.

    Returns Markdown text with ``##`` section headings (used later by the
    section-aware chunker). Raises :class:`DoclingConversionError` on any
    failure, including a missing Docling install.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise DoclingConversionError(f"PDF not found: {pdf_path}")

    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise DoclingConversionError(
            "Docling is not installed. Run: pip install -e \".[pdf]\""
        ) from exc

    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()
    except Exception as exc:  # noqa: BLE001 - surface any Docling error uniformly
        raise DoclingConversionError(
            f"Docling failed to convert {pdf_path.name}: {exc}"
        ) from exc
