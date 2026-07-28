# 📄 FindDoc

**FindDoc** is an AI-powered Retrieval-Augmented Generation (RAG) application that allows users to ask natural language questions about NVIDIA's 2025 Annual Report and receive accurate, context-aware answers with supporting citations.

The project demonstrates a production-oriented RAG pipeline built with FastAPI, Streamlit, FAISS, Redis, and LangChain.

---

## Features

* Semantic search using **FAISS**
* Hybrid retrieval (Dense + BM25)
* Streaming responses with **FastAPI Server-Sent Events (SSE)**
* Multi-query retrieval for improved recall
* Conversation memory using **Redis**
* Follow-up question support
* Streamlit chat interface
* PDF preprocessing and caching for faster startup
* Prebuilt FAISS index included for quick setup

---

## Tech Stack

### Backend

* FastAPI
* LangChain
* FAISS
* Redis
* Sentence Transformers
* Groq LLM
* BM25
* Unstructured
* Python

### Frontend

* Streamlit
* Requests

---

## Project Structure

```text
Rag/
│
├── backend/
│   ├── api/
│   ├── generation/
│   ├── retriever/
│   ├── cache/
│   ├── faiss_index/
│   ├── data/
│   └── requirements.txt
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
├── README.md
└── .gitignore
```

backend/json_database/
├── final_processed_chunks.json
├── final_processed_chunks_FIXED.json
└── merge_log.json

# Purpose

- final_processed_chunks.json — Initial processed chunks.
- final_processed_chunks_FIXED.json — Corrected version after metadata cleanup and processing fixes.
- merge_log.json — Records the chunk merge operations performed during preprocessing.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/finddoc-rag.git

cd finddoc-rag
```

---

### Backend

Create and activate a virtual environment.

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

Create a `.env` file inside the project containing:

```env
GROQ_API_KEY=your_key_here
REDIS_URL=your_redis_url
HF_TOKEN=your_huggingface_token
```

Start the backend:

```bash
uvicorn backend.api.main:app --reload
```

---

### Frontend

Install frontend dependencies:

```bash
pip install -r frontend/requirements.txt
```

Run the application:

```bash
streamlit run frontend/app.py
```

---

## Notes

This repository includes:

* The original NVIDIA Annual Report PDF
* The preprocessed cache
* A prebuilt FAISS index

These are intentionally committed to the repository so contributors and reviewers can run the project immediately without waiting for document parsing, embedding generation, or index creation.

If you'd like to regenerate everything from scratch, simply delete the cache and FAISS index folders and rerun the ingestion pipeline.

---

## Future Improvements

* Cross-encoder reranking
* Structured citation UI
* Rate limiting and API authentication
* Automated evaluation dataset
* Docker support
* AWS deployment
* Multi-document support

---

## License

This project is licensed under the MIT License.
