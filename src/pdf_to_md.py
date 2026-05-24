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
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise DoclingConversionError(
            "Docling is not installed. Run: pip install -e \".[pdf]\""
        ) from exc

    # The corpus is digital (text-based) Hebrew PDFs, so OCR is unnecessary
    # (and Docling's default CJK OCR models are useless for Hebrew). The real
    # killer is memory: the default pipeline buffers up to 100 rasterized pages
    # (queue_max_size=100) and runs the layout model in batches of 4 across 4
    # threads, which exhausts RAM -> std::bad_alloc -> segfault, dropping pages.
    # Configure a low-memory pipeline: no OCR, one page at a time, tiny queue,
    # single thread, and the lighter (FAST) table model.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.generate_page_images = False
    pipeline_options.layout_batch_size = 1
    pipeline_options.table_batch_size = 1
    pipeline_options.queue_max_size = 2
    try:
        pipeline_options.accelerator_options.num_threads = 1
    except (AttributeError, ValueError):
        # Field name or configuration differs across Docling versions; continue with defaults
        pass
    try:
        from docling.datamodel.pipeline_options import TableFormerMode

        pipeline_options.table_structure_options.mode = TableFormerMode.FAST
    except (ImportError, AttributeError):
        # TableFormerMode or table_structure_options unavailable in this Docling version; use default ACCURATE
        pass

    try:
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
        result = converter.convert(str(pdf_path))
        return result.document.export_to_markdown()
    except Exception as exc:  # noqa: BLE001 - surface any Docling error uniformly
        raise DoclingConversionError(
            f"Docling failed to convert {pdf_path.name}: {exc}"
        ) from exc
