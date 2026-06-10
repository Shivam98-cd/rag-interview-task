"""
tests/test_pipeline.py

Unit tests for the RAG pipeline components.
Run with: pytest tests/ -v

Tests are designed to be runnable WITHOUT API keys (mocking external calls).
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Chunker tests ─────────────────────────────────────────────────────────────

class TestChunker(unittest.TestCase):

    def setUp(self):
        from src.ingestion.chunker import chunk_pages
        self.chunk_pages = chunk_pages

    def test_basic_chunking(self):
        pages = [{"page": 1, "text": "Hello world. " * 50, "source": "test.pdf"}]
        chunks = self.chunk_pages(pages, chunk_size=100, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertIn("chunk_id", c)
            self.assertIn("text", c)
            self.assertIn("page", c)

    def test_chunk_ids_are_unique(self):
        pages = [{"page": 1, "text": "A " * 200, "source": "test.pdf"}]
        chunks = self.chunk_pages(pages, chunk_size=50, chunk_overlap=10)
        ids = [c["chunk_id"] for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_short_text_single_chunk(self):
        pages = [{"page": 1, "text": "Short text.", "source": "test.pdf"}]
        chunks = self.chunk_pages(pages, chunk_size=500, chunk_overlap=100)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "Short text.")

    def test_metadata_preserved(self):
        pages = [{"page": 5, "text": "Some content here.", "source": "doc.pdf"}]
        chunks = self.chunk_pages(pages)
        self.assertEqual(chunks[0]["page"], 5)
        self.assertEqual(chunks[0]["source"], "doc.pdf")


# ── PDF Loader tests ──────────────────────────────────────────────────────────

class TestPDFLoader(unittest.TestCase):

    def test_table_to_text(self):
        from src.ingestion.pdf_loader import _table_to_text
        table = [["Header1", "Header2"], ["val1", "val2"], [None, "val3"]]
        result = _table_to_text(table)
        self.assertIn("Header1 | Header2", result)
        self.assertIn("val1 | val2", result)
        self.assertIn(" | val3", result)

    def test_clean(self):
        from src.ingestion.pdf_loader import _clean
        dirty = "Hello   World\n\n\n\nNext paragraph"
        cleaned = _clean(dirty)
        self.assertNotIn("   ", cleaned)
        self.assertNotIn("\n\n\n", cleaned)


# ── Embedder tests ────────────────────────────────────────────────────────────

class TestEmbedder(unittest.TestCase):

    @patch("src.embedding.embedder._get_model")
    def test_embed_texts_shape(self, mock_get_model):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
        mock_get_model.return_value = mock_model

        from src.embedding.embedder import embed_texts
        result = embed_texts(["text one", "text two"])
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 384)

    @patch("src.embedding.embedder._get_model")
    def test_embed_query_returns_list(self, mock_get_model):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.5] * 384])
        mock_get_model.return_value = mock_model

        from src.embedding.embedder import embed_query
        result = embed_query("What is communication?")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 384)

    @patch("src.embedding.embedder._get_model")
    def test_embed_chunks_adds_embedding_key(self, mock_get_model):
        import numpy as np
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384])
        mock_get_model.return_value = mock_model

        from src.embedding.embedder import embed_chunks
        chunks = [{"chunk_id": "c0", "text": "Hello", "page": 1, "source": "x.pdf"}]
        result = embed_chunks(chunks)
        self.assertIn("embedding", result[0])
        self.assertEqual(len(result[0]["embedding"]), 384)


# ── Retriever tests ───────────────────────────────────────────────────────────

class TestRetriever(unittest.TestCase):

    def test_build_context_format(self):
        from src.retrieval.retriever import build_context
        chunks = [
            {"text": "Business communication is...", "page": 1, "score": 0.9},
            {"text": "Verbal communication involves...", "page": 3, "score": 0.8},
        ]
        context = build_context(chunks)
        self.assertIn("[Chunk 1 | Page 1]", context)
        self.assertIn("[Chunk 2 | Page 3]", context)
        self.assertIn("---", context)

    @patch("src.retrieval.retriever.embed_query")
    @patch("src.retrieval.retriever.query_index")
    def test_retrieve_calls_embed_and_query(self, mock_query, mock_embed):
        mock_embed.return_value = [0.1] * 384
        mock_query.return_value = [
            {"chunk_id": "c0", "text": "some text", "page": 1, "score": 0.85, "source": "test.pdf"}
        ]

        from src.retrieval.retriever import retrieve
        result = retrieve("What is communication?")
        mock_embed.assert_called_once_with("What is communication?")
        mock_query.assert_called_once()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["score"], 0.85)


# ── LLM Client tests ──────────────────────────────────────────────────────────

class TestLLMClient(unittest.TestCase):

    def test_low_score_returns_fallback(self):
        from src.generation.llm_client import generate_answer, FALLBACK_RESPONSE
        result = generate_answer("any question", "any context", top_score=0.05)
        self.assertEqual(result, FALLBACK_RESPONSE)

    @patch("src.generation.llm_client.Groq")
    def test_generate_answer_calls_groq(self, mock_groq_class):
        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Communication is key."
        mock_client.chat.completions.create.return_value = mock_response

        from src.generation.llm_client import generate_answer
        answer = generate_answer("What is communication?", "Context text", top_score=0.8)
        self.assertEqual(answer, "Communication is key.")
        mock_client.chat.completions.create.assert_called_once()


# ── Flask API tests ───────────────────────────────────────────────────────────

class TestAPI(unittest.TestCase):

    def setUp(self):
        # Patch config.validate so it doesn't require real keys during tests
        patcher = patch("config.validate")
        patcher.start()
        self.addCleanup(patcher.stop)

        import importlib
        import src.api.app as app_module
        importlib.reload(app_module)
        self.client = app_module.app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json(), {"status": "ok"})

    def test_ask_missing_question(self):
        response = self.client.post(
            "/ask",
            json={},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())

    def test_ask_empty_question(self):
        response = self.client.post(
            "/ask",
            json={"question": "   "},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    def test_ask_non_json_returns_415(self):
        response = self.client.post("/ask", data="not json")
        self.assertEqual(response.status_code, 415)

    @patch("src.api.app.retrieve")
    @patch("src.api.app.generate_answer")
    def test_ask_valid_question(self, mock_gen, mock_ret):
        mock_ret.return_value = [
            {"chunk_id": "c0", "text": "Business communication is...", "page": 1, "score": 0.9, "source": "x.pdf"}
        ]
        mock_gen.return_value = "Business communication involves exchanging information."

        response = self.client.post(
            "/ask",
            json={"question": "What is business communication?"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("answer", data)
        self.assertEqual(data["answer"], "Business communication involves exchanging information.")

    @patch("src.api.app.retrieve")
    def test_ask_no_chunks_returns_fallback(self, mock_ret):
        mock_ret.return_value = []
        response = self.client.post(
            "/ask",
            json={"question": "What is quantum physics?"},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("not available", data["answer"])


if __name__ == "__main__":
    unittest.main(verbosity=2)