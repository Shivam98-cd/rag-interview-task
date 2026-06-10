# 📚 RAG QA Application — Business Communication PDF

A production-ready **Retrieval-Augmented Generation (RAG)** backend that answers questions strictly from the provided Business Communication PDF document.

---

## 🏗️ Architecture

```
Data Ingestion → Chunking → Embedding → Pinecone VectorDB → Retrieval → LLM Answer
```

| Layer         | Technology                                         |
|---------------|----------------------------------------------------|
| PDF Parsing   | `pdfplumber` (handles text + tables)               |
| Chunking      | Recursive character text splitter (custom)         |
| Embedding     | `sentence-transformers/all-MiniLM-L6-v2` (free, open-source, 384-dim) |
| Vector Store  | Pinecone (serverless)                              |
| LLM           | Groq API — `llama3-8b-8192`                        |
| API           | Flask                                              |

---

## 📂 Project Structure

```
rag_app/
├── src/
│   ├── ingestion/
│   │   └── pdf_loader.py          # PDF text + table extraction
│   ├── embedding/
│   │   └── embedder.py            # Sentence-transformer embeddings
│   ├── vectordb/
│   │   └── pinecone_client.py     # Pinecone upsert & query
│   ├── retrieval/
│   │   └── retriever.py           # Semantic search
│   ├── generation/
│   │   └── llm_client.py          # Groq LLM answer generation
│   └── api/
│       └── app.py                 # Flask POST /ask endpoint
├── data/
│   └── business_communication.pdf # Your PDF goes here
├── tests/
│   └── test_pipeline.py           # Unit tests
├── ingest.py                      # One-time ingestion script
├── config.py                      # All configuration (env-driven)
├── requirements.txt
└── .env.example
```

---

## ⚙️ Setup

### 1. Clone & install

```bash
git clone <repo>
cd rag_app
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 3. Place the PDF

Copy `Business_Communication.pdf` → `data/business_communication.pdf`

### 4. Ingest the document (one-time)

```bash
python ingest.py
```

This loads the PDF, chunks it, embeds every chunk, and upserts all vectors into Pinecone.

### 5. Run the API

```bash
python src/api/app.py
```

---

## 🔌 API Usage

### `POST /ask`

**Request:**
```json
{
  "question": "What are the 7 Cs of effective communication?"
}
```

**Response:**
```json
{
  "answer": "The 7 Cs of effective communication are: Clear, Concise, Concrete, Correct, Coherent, Complete, and Courteous."
}
```

**Error Response (question not in document):**
```json
{
  "answer": "The answer is not available in the provided document."
}
```

### `GET /health`
```json
{ "status": "ok" }
```

---

## 🤖 Embedding Model Choice

**Model:** `sentence-transformers/all-MiniLM-L6-v2`

- ✅ Completely free and open-source (Apache 2.0)
- ✅ 384-dimensional vectors — perfect for Pinecone free tier
- ✅ Fast inference (~14,000 sentences/sec on CPU)
- ✅ Strong semantic similarity on English text
- ✅ ~22MB download, no GPU required

**Alternatives considered:**
| Model | Dims | Notes |
|-------|------|-------|
| `all-MiniLM-L6-v2` | 384 | ✅ Recommended — fast, accurate |
| `all-mpnet-base-v2` | 768 | Better accuracy, slower |
| `multi-qa-MiniLM-L6-cos-v1` | 384 | Tuned for QA tasks |
| `BAAI/bge-small-en-v1.5` | 384 | SOTA small model |

---

## 📝 Assumptions

1. The PDF has a native text layer (not scanned) — `pdfplumber` is used instead of OCR.
2. Pinecone free tier (serverless, us-east-1) is used with index dimension 384.
3. Groq `llama3-8b-8192` is the LLM (free tier available).
4. Top-5 chunks are retrieved for each question.
5. Chunk size: 500 chars with 100-char overlap for context continuity.
