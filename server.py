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

from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.generation import answer  # noqa: E402

app = Flask(__name__)
CORS(app)  # allow requests from file:// origin


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
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("RAG server running on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
