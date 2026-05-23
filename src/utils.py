"""Thin shared helpers: logging and file IO.

No domain logic lives here. Hebrew content is always written with
``ensure_ascii=False`` so the JSON stays human-readable.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any


def get_logger(name: str) -> logging.Logger:
    """Return a configured console logger.

    Level is taken from the ``INSURANCE_RAG_LOG_LEVEL`` env var (default INFO).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        logger.addHandler(handler)
    level = os.environ.get("INSURANCE_RAG_LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    logger.propagate = False
    return logger


def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return Path(path).read_text(encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    """Write a UTF-8 text file, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read and parse a UTF-8 JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    """Write an object as pretty UTF-8 JSON (Hebrew preserved)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )
