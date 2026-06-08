"""Flask API server for the insurance-RAG live demo.

Run:
    python server.py

Exposes:
    POST /ask  { "question": str, "strategy": str? }
            -> { "answer": str, "sources": [str], "strategy": str }
"""
from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.generation import answer  # noqa: E402
from src.utils import get_logger  # noqa: E402

logger = get_logger(__name__)

_DOCS_DIR = Path(__file__).resolve().parent / "docs"

app = Flask(__name__, static_folder=str(_DOCS_DIR), static_url_path="")
CORS(app)  # allow requests from file:// origin


@app.route("/")
def index():
    return send_from_directory(_DOCS_DIR, "project_site.html")


@app.route("/ask", methods=["POST"])
def ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "missing question"}), 400

    strategy = body.get("strategy", "section_aware")

    try:
        result = answer(question, strategy=strategy)
        return jsonify({
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "strategy": result.get("strategy", strategy),
        })
    except Exception as exc:  # noqa: BLE001
        # Log full error server-side; return generic message to client
        logger.error(f"Error answering question: {exc}", exc_info=True)
        return jsonify({"error": "Failed to generate answer"}), 500


def _do_warmup():
    try:
        logger.info("Starting background warmup...")
        # 1. Warm up SentenceTransformer model
        from src.embedder import _get_model
        _get_model()
        logger.info("SentenceTransformer model loaded.")

        # 2. Warm up ChromaDB collections
        from src.indexer import load_collection
        load_collection("fixed")
        load_collection("section_aware")
        logger.info("ChromaDB collections loaded.")
        logger.info("Background warmup completed successfully.")
    except Exception as exc:
        logger.error(f"Error during background warmup: {exc}", exc_info=True)


@app.route("/warmup", methods=["GET", "POST"])
def warmup():
    # Run synchronously so that Cloud Run keeps the CPU active (unthrottled)
    # while the model and database are loading.
    _do_warmup()
    return jsonify({"status": "warmed_up"}), 200


if __name__ == "__main__":
    print("RAG server running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
