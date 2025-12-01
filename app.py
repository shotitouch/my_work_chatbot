
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os

load_dotenv()

app = FastAPI()

# setup vectorstore
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory="vectorstore"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # pull 3 chunks

# build llm chain
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4.1-mini",
    max_tokens=300
)

# keep chat history
session_history = {}
MAX_HISTORY = 5

prompt = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful assistant. Use ONLY the context. If it does not contain the answer, reply: 'Not found in context'."),
    MessagesPlaceholder(variable_name="history"),
    ('user', "Question: {question}\n\nContext:\n{context}")
])

parser = StrOutputParser()

# rag search Chroma and output
def rag_answer(chain, question, history):
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    return chain.invoke({"history": history, "question": question, "context": context}), context

# Request schema
class Query(BaseModel):
    session_id: int
    question: str

# ----------------------------------------
# REST API endpoint
# ----------------------------------------
@app.post("/ask")
async def ask(q: Query):
    # Retrieve relevant chunks
    session = q.session_id

    if session not in session_history:
        session_history[session] = []

    history = session_history[session]
    chain = prompt | llm | parser
    answer, context = rag_answer(chain, q.question, history)

    history.append(HumanMessage(content=q.question))
    history.append(AIMessage(content=answer))
    session_history[session] = history[-MAX_HISTORY:]
    return {
        "question": q.question,
        "answer": answer,
        "context_used": context,
    }