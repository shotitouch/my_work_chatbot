
📚 RAG PDF Chatbot (LangChain v1.x + Chroma + OpenRouter)

A Retrieval-Augmented Generation (RAG) chatbot that can answer questions from a PDF using:

- LangChain v1.x
- ChatOpenAI (OpenRouter)
- ChromaDB
- OpenAI embeddings
- Conversation memory with MessagesPlaceholder

This project loads your PDF, chunks it, generates embeddings, stores them in a vector DB, and answers user questions using retrieved context.

------------------------------------------------------------

🚀 Features

- Load any PDF file
- Automatic text chunking with overlap
- Embedding + Vector storage via ChromaDB
- Semantic search retrieval
- Chat-style Q&A with memory (HumanMessage / AIMessage)
- Strict context-only answers (hallucination reduced)

------------------------------------------------------------

🧰 Tech Stack

- Python 3.10+
- LangChain Core 1.x
- langchain-openai
- langchain-community
- ChromaDB
- OpenRouter API (ChatOpenAI models)

------------------------------------------------------------

📦 Installation

1. Create venv:
   python -m venv .venv

2. Activate:

   Windows:
   .venv\Scripts\activate

   Mac/Linux:
   source .venv/bin/activate

3. Install:
   pip install -r requirements.txt

------------------------------------------------------------

🔑 Environment Variables

Create .env:

OPENROUTER_API_KEY=your_key

Get key from https://openrouter.ai

------------------------------------------------------------

▶️ Running the Chatbot

python rag_pdf.py

------------------------------------------------------------

🧠 RAG Pipeline

1. Load PDF
2. Split into text chunks
3. Embed chunks
4. Store in Chroma
5. Retrieve relevant chunks
6. Inject into prompt
7. LLM answers using ONLY context
8. Memory stored via HumanMessage / AIMessage

------------------------------------------------------------

📌 Future Improvements

- Streaming responses
- Web UI (Gradio)
- Multi-PDF support
- Source citations

