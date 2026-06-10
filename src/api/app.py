"""
src/api/app.py

Flask REST API exposing the RAG pipeline.

Endpoints:
  POST /ask    — answer a question from the document
  GET  /health — liveness check
"""

from __future__ import annotations
import sys
import os
conversation_history = []

# Allow imports from project root
# Add project root to path (works on Windows)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from flask import Flask, request, jsonify, Response
from flask_cors import CORS

import config
from src.retrieval.retriever import retrieve, build_context
from src.generation.llm_client import generate_answer, FALLBACK_RESPONSE

config.validate()

app = Flask(__name__)
CORS(app)


# ── helpers ───────────────────────────────────────────────────────────────────

def _error(message: str, status: int) -> tuple[Response, int]:
    return jsonify({"error": message}), status


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health() -> tuple[Response, int]:
    """Liveness check."""
    return jsonify({"status": "ok"}), 200


@app.route("/ask", methods=["POST"])
def ask() -> tuple[Response, int]:
    """
    Answer a question based on the ingested document.

    Request body (JSON):
        { "question": "What is business communication?" }

    Response (JSON):
        { "answer": "Business communication is ..." }
    """
    # ── input validation ──────────────────────────────────────────────────────
    if not request.is_json:
        return _error("Request must be JSON (Content-Type: application/json).", 415)

    data = request.get_json(silent=True)
    if data is None:
        return _error("Invalid JSON body.", 400)

    question = data.get("question", "").strip()
    if not question:
        return _error("'question' field is required and cannot be empty.", 400)

    if len(question) > 1000:
        return _error("'question' must be 1000 characters or fewer.", 400)

    # ── RAG pipeline ──────────────────────────────────────────────────────────
    try:
        # 1. Retrieve relevant chunks
        chunks = retrieve(question)

        if not chunks:
            return jsonify({"answer": FALLBACK_RESPONSE}), 200

        # 2. Build context string
        context = build_context(chunks)

        # 3. Extract top relevance score for threshold check
        top_score = chunks[0]["score"] if chunks else 0.0

        # 4. Generate answer via Groq
        answer = generate_answer(question, context, top_score=top_score)

        return jsonify({"answer": answer}), 200

    except Exception as exc:
        # Log to stderr so it appears in server logs
        import traceback
        traceback.print_exc()
        return _error(f"Internal server error: {str(exc)}", 500)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(
        f"[API] Starting Flask on {config.FLASK_HOST}:{config.FLASK_PORT} "
        f"(debug={config.FLASK_DEBUG})"
    )
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )