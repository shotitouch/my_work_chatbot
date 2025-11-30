
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
import os

load_dotenv()
# pdf loader
loader = PyPDFLoader("NLP_midterm_juyeong_shotitouch.pdf")
docs = loader.load()

# split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

# embedding chunks -> store in vector db
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

vectorstore = Chroma.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever()

# build llm chain
llm = ChatOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    model="openai/gpt-4.1-mini",
    max_tokens=300
)

# keep chat history
history = []

prompt = ChatPromptTemplate.from_messages([
    ('system', "You are a helpful assistant. Use ONLY the context. If it does not contain the answer, reply: 'Not found in context'."),
    MessagesPlaceholder(variable_name="history"),
    ('user', "Question: {question}\n\nContext:\n{context}")
])

parser = StrOutputParser()
chain = prompt | llm | parser

# rag search Chroma and output
def rag_answer(question):
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    return chain.invoke({"history": history, "question": question, "context": context})

# chatbot
if __name__ == "__main__":
    while True:
        # question: user
        q = input("\nAsk (or 'exit'): ")
        if q.lower() == "exit":
            break

        # answer: assistant
        a = rag_answer(q)
        print("\nAnswer:\n", a)
        
        # store history
        history.append(HumanMessage(content=q))
        history.append(AIMessage(content=a))
