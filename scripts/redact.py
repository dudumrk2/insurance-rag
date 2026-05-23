"""CLI: convert and redact the raw policy corpus.

    python scripts/redact.py [--reset]

Reads every ``data/raw/*.pdf``, converts it to Markdown (Docling), removes PII
(regex + known strings from ``data/known_pii.json``), and writes:

  - ``data/redacted/<name>.md``      (committed)
  - ``data/redaction_log.json``      (committed; counts + safe context, no PII)

Docling failures are logged per file and skipped; the script continues and
exits with code 2 if any file failed (0 otherwise).
"""

import argparse
import sys

from src import config
from src.pdf_to_md import DoclingConversionError, convert
from src.redaction import redact
from src.utils import get_logger, read_json, write_json, write_text

logger = get_logger("redact")


def load_known_strings() -> list[str]:
    """Load names/IDs to redact from data/known_pii.json (if present).

    Expected shape: ``{"strings": ["...", ...]}``. Missing file -> empty list.
    """
    if not config.KNOWN_PII_FILE.exists():
        return []
    data = read_json(config.KNOWN_PII_FILE)
    strings = data.get("strings", []) if isinstance(data, dict) else []
    return [s for s in strings if isinstance(s, str) and s.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert + redact raw policy PDFs.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete existing data/redacted/*.md before running",
    )
    args = parser.parse_args()

    config.REDACTED_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset:
        removed = 0
        for md in config.REDACTED_DIR.glob("*.md"):
            md.unlink()
            removed += 1
        logger.info("Reset: cleared %d existing redacted file(s).", removed)

    known = load_known_strings()
    if known:
        logger.info("Loaded %d known-PII string(s) from %s.", len(known), config.KNOWN_PII_FILE.name)
    else:
        logger.warning(
            "No %s found — regex-only redaction. Names will NOT be removed by "
            "the known-strings pass.",
            config.KNOWN_PII_FILE.name,
        )

    pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
    if not pdfs:
        logger.warning("No PDFs in %s. Add files and re-run.", config.RAW_DIR)
        return 0

    per_file: dict[str, dict] = {}
    failures: list[str] = []
    total = 0

    for pdf in pdfs:
        try:
            md = convert(pdf)
        except DoclingConversionError as exc:
            logger.error("Skipping %s: %s", pdf.name, exc)
            failures.append(pdf.name)
            continue

        redacted, log = redact(md, known)
        out_path = config.REDACTED_DIR / f"{pdf.stem}.md"
        write_text(out_path, redacted)
        per_file[pdf.name] = log
        total += log["total"]
        logger.info("Redacted %s -> %s (%d removals).", pdf.name, out_path.name, log["total"])

    write_json(config.REDACTION_LOG, {"files": per_file, "grand_total": total})

    _print_summary(per_file, failures, total)
    return 2 if failures else 0


def _print_summary(per_file: dict, failures: list, total: int) -> None:
    print("\n=== Redaction summary ===")
    for name, log in per_file.items():
        print(f"  {name}: {log['total']} removals")
    print(f"  Total: {total} removals across {len(per_file)} file(s)")
    if failures:
        print(f"  FAILED ({len(failures)}): {', '.join(failures)}")
    print(f"\nLog written to {config.REDACTION_LOG}")
    print("\n[!] REVIEW data/redaction_log.json before committing or submitting.")
    print("    Regex + known-strings is not perfect — manual review is the safety net.\n")


if __name__ == "__main__":
    sys.exit(main())
