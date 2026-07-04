import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI(title="Private RAG API")

# Setup Ollama Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
DB_DIR = "/app/data/chroma_db"
TEMP_DIR = "/app/data/temp"

# Ensure directories exist
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize models
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)
llm = Ollama(model=LLM_MODEL, base_url=OLLAMA_URL)

# Setup Vector Store
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Setup RAG Chain
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Keep the answer concise and professional.

Context:
{context}

Question: {question}

Answer:"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class ChatRequest(BaseModel):
    message: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    temp_path = os.path.join(TEMP_DIR, file.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Load and split
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Add to vectorstore
        vectorstore.add_documents(documents=splits)
        
        return {"status": "success", "message": f"Successfully ingested {len(splits)} chunks from {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = rag_chain.invoke(request.message)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
