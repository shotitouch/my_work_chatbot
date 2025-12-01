from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredExcelLoader
import os

load_dotenv()

def load_all_pdfs(path="data/pdf"):
    docs = []
    for file in os.listdir(path):
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, file))
            docs.extend(loader.load())
    return docs

def load_all_excels(path="data/excel"):
    docs = []
    for file in os.listdir(path):
        if file.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(os.path.join(path, file))
            docs.extend(loader.load())
    return docs

# Load PDF
docs_pdf = load_all_pdfs(path="data/pdf")
print(f"Loaded {len(docs_pdf)} PDF docs")

# Load Excel
docs_excel = load_all_pdfs(path="data/excel")
print(f"Loaded {len(docs_excel)} Excel docs")

docs = docs_pdf + docs_excel

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Save to local chroma DB
Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="vectorstore"
)

print("Vectorstore created")
