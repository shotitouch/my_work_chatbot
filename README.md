📚 RAG PDF Chatbot API  
(LangChain v1.x + FastAPI + ChromaDB + OpenRouter)

A production-ready Retrieval-Augmented Generation (RAG) chatbot API built with:
- FastAPI
- LangChain v1.x (Core / OpenAI / Community)
- ChromaDB (persistent vector store)
- OpenRouter LLMs (ChatOpenAI interface)
- Session-based conversation memory

This project loads your PDF(s), chunks them, embeds them, stores them in Chroma, and exposes a /ask REST API endpoint that returns context-aware answers.

------------------------------------------------------------

🚀 Features

- Load any PDF file
- Automatic text chunking with overlap
- Embedding + vector storage (Chroma)
- Semantic search retrieval
- Chat-style Q&A with per-session memory
- Strict context-only answers to reduce hallucinations
- FastAPI server with automatic Swagger UI
- Persistent vectorstore (survives restarts)

------------------------------------------------------------

🧰 Tech Stack

- Python 3.10+
- FastAPI 0.123+
- Uvicorn 0.38+
- LangChain v1.x
  - langchain
  - langchain-openai
  - langchain-community
  - langchain-text-splitters
- ChromaDB 0.5.4 (legacy API for LangChain retriever)
- OpenRouter API
- pypdf

------------------------------------------------------------

📦 Installation

1. Create virtual environment:
   python -m venv .venv

2. Activate:

   Windows:
   .venv\Scripts\activate

   Mac/Linux:
   source .venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

------------------------------------------------------------

🔑 Environment Variables

Create a file named .env:

OPENROUTER_API_KEY=your_key_here

Get your key from:
https://openrouter.ai/

------------------------------------------------------------

▶️ Building the Vectorstore

If using a setup script, run:

python setup_vectorstore.py

This will:
- Load the PDF
- Split it into text chunks
- Embed each chunk
- Save embeddings into /vectorstore

------------------------------------------------------------

▶️ Running the FastAPI Server

Start FastAPI:
uvicorn app:app --reload

Documentation available at:
Swagger UI → http://127.0.0.1:8000/docs
ReDoc       → http://127.0.0.1:8000/redoc

------------------------------------------------------------

📡 REST API Usage

POST /ask

Example request:
{
  "session_id": 1,
  "question": "What is the main idea of page 2?"
}

Example response:
{
  "question": "What is the main idea of page 2?",
  "answer": "...",
  "context_used": "...",
  "history_size": 4
}

------------------------------------------------------------

🧠 RAG Pipeline

1. Load PDF
2. Split into overlapping chunks
3. Embed chunks using OpenAI embeddings
4. Store embeddings inside ChromaDB
5. Retrieve top-k relevant chunks
6. Insert the chunks into prompt template
7. LLM answers using ONLY retrieved context
8. HumanMessage / AIMessage stored as conversation memory

------------------------------------------------------------

📌 Future Improvements

- Streaming responses
- Web UI (React / Gradio)
- Support multiple PDFs
- Source citations (page-level)
- Reranking (bge-reranker)
- API key authentication
- Docker deployment
