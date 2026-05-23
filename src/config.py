"""Central configuration: paths, model names, and constants.

This module holds no logic — it is the single source of truth for every path
and tunable used across the pipeline. Importing it has no side effects.

A ``.env`` file in the project root is loaded automatically via python-dotenv
(if installed), so secrets like ``GEMINI_API_KEY`` can be stored there instead
of being set manually in the shell each time. ``.env`` is gitignored.
"""

from pathlib import Path

try:
    from dotenv import load_dotenv  # noqa: PLC0415

    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass  # python-dotenv is optional; fall back to existing env vars

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REDACTED_DIR = DATA_DIR / "redacted"
PROCESSED_DIR = DATA_DIR / "processed"
INDICES_DIR = ROOT / "indices"

KNOWN_PII_FILE = DATA_DIR / "known_pii.json"
REDACTION_LOG = DATA_DIR / "redaction_log.json"

# --- Models (used from Step 3 onward; declared here as one source of truth) -
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
GENERATION_MODEL = "gemini-2.5-flash"

# --- Chunking (used from Step 2) -------------------------------------------
FIXED_CHUNK_SIZE = 500
FIXED_CHUNK_OVERLAP = 50
SECTION_MAX_TOKENS = 700

# --- Multi-tenancy ---------------------------------------------------------
# Assignment corpus runs under this fixed family_id. Integration overrides it
# with the real demo uid from ai-wealth-monitor.
DEFAULT_FAMILY_ID = "demo_family_001"

# --- Retrieval (used from Step 4) ------------------------------------------
DEFAULT_TOP_K = 5
