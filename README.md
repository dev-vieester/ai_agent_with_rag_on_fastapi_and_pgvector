# 🚀 Uten Assistant API

A **production-ready RAG (Retrieval-Augmented Generation) backend** built with FastAPI, PostgreSQL, and vector search.

This project is designed as a **modular, scalable AI backend** that supports document ingestion, semantic search, and AI-powered responses using LLMs.

---

## 🧠 What This Project Does

Uten Assistant allows you to:

* 📄 Upload and process documents
* ✂️ Chunk and embed text into vector representations
* 🔍 Retrieve relevant context using semantic search
* 🤖 Generate intelligent answers using LLMs
* 🔐 Manage users with authentication & RBAC
* ⚡ Serve responses via a clean FastAPI architecture

---

## 🏗️ Architecture Overview

The system follows a **clean, modular architecture**:

```
.
├── core/                # Shared domain logic (models, schemas, access control)
├── fast_api/           # API layer (routes, dependencies, security)
├── rag/                # RAG core logic (chunking, embedding, retrieval, generation)
│   ├── core/
│   ├── pipeline/
│   └── storage/
├── main.py             # App entry point (lifespan, middleware, routers)
└── .env                # Environment variables
```

---

## 🔄 RAG Pipeline Flow

1. **Document Ingestion**

   * Upload document
   * Clean & preprocess text

2. **Chunking**

   * Split text into smaller segments

3. **Embedding**

   * Convert chunks into vector embeddings

4. **Storage**

   * Save vectors in PostgreSQL (pgvector)
   * Store metadata separately

5. **Retrieval**

   * Query similar chunks using vector similarity

6. **Generation**

   * Pass retrieved context to LLM (Anthropic)
   * Generate final answer

---

## ⚙️ Tech Stack

* **Backend:** FastAPI
* **Database:** PostgreSQL + pgvector
* **ORM:** SQLAlchemy
* **LLM Provider:** Anthropic
* **Auth:** JWT-based authentication
* **Architecture:** Clean Architecture + Modular RAG pipeline

---

## 🔐 Features

* ✅ RAG pipeline (ingest + query)
* ✅ Authentication (JWT)
* ✅ Role-Based Access Control (RBAC)
* ✅ Rate limiting middleware
* ✅ CORS configuration
* ✅ Health check endpoint
* ✅ Environment-based config

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/uten-assistant.git
cd uten-assistant
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup environment variables

Create a `.env` file:

```env
DATABASE_URL=postgresql://user:password@localhost:5432/uten
ANTHROPIC_API_KEY=your_api_key
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_WINDOW_SECONDS=60
```

### 5. Run the server

```bash
uvicorn main:app --reload --port 8000
```

---

## 📡 API Endpoints

### Health Check

```
GET /health
```

### Auth

```
POST /api/auth
```

### Users

```
GET /api/users
```

### Documents

```
POST /api/documents
```

### Ask (RAG Query)

```
POST /api/ask
```

---

## 🧩 Key Design Decisions

* **Lifespan-based initialization** → shared pipeline instance across requests
* **Separation of concerns** → API, core logic, and RAG pipeline are independent
* **Extensibility** → easy to swap LLMs or vector DB
* **Production mindset** → includes rate limiting, validation, and structured errors

---

## 📌 Future Improvements

* 🔄 Streaming responses (real-time AI answers)
* 📊 Token usage tracking & analytics
* 🌐 Multi-tenant support
* 📁 File storage (S3 / cloud)
* 🧠 Hybrid search (keyword + vector)

---

## 👨‍💻 Author

**Victor Adeniyi**

* Software Engineer (Flutter + AI)
* Founder of Xedlapay

---

## ⭐ Why This Project Matters

This project demonstrates:

* Real-world **AI backend engineering**
* Understanding of **RAG systems**
* Clean, scalable **API architecture**
* Production-ready engineering practices

---

## 🪪 License

MIT License
