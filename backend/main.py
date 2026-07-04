import os
import shutil
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# SQLite for chat history
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import declarative_base, sessionmaker

app = FastAPI(title="Private RAG API V2")

# Setup Ollama Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "/app/data/chroma_db"
TEMP_DIR = "/app/data/temp"
SQLITE_DB = "sqlite:////app/data/chat_history.db"

# Ensure directories exist
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Database Setup
Base = declarative_base()
engine = create_engine(SQLITE_DB)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class ChatMessage(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    citations = Column(Text, nullable=True) # JSON string of citations

Base.metadata.create_all(bind=engine)

# Initialize models
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_URL)

# Setup Vector Store
vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
search_tool = DuckDuckGoSearchRun()

# Setup Prompts
qa_system_prompt = """You are an intelligent assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Keep the answer concise and professional.

{context}"""

qa_prompt = PromptTemplate.from_template(
    qa_system_prompt + "\n\nQuestion: {input}\n\nAnswer:"
)

class ChatRequest(BaseModel):
    message: str
    session_id: str
    model: str = "llama3"

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    temp_path = os.path.join(TEMP_DIR, file.filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        # Add filename to metadata
        for doc in docs:
            doc.metadata["source"] = file.filename
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
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
        # Save user message
        db = SessionLocal()
        user_msg = ChatMessage(session_id=request.session_id, role="user", content=request.message)
        db.add(user_msg)
        db.commit()

        # Dynamic LLM initialization
        llm = Ollama(model=request.model, base_url=OLLAMA_URL)
        
        # Check if RAG has answers
        docs = retriever.invoke(request.message)
        
        citations = []
        if len(docs) > 0:
            # We have local documents, use RAG
            combine_docs_chain = create_stuff_documents_chain(llm, qa_prompt)
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
            res = retrieval_chain.invoke({"input": request.message})
            answer = res["answer"]
            
            # Extract citations
            for doc in res["context"]:
                source = doc.metadata.get("source", "Unknown Document")
                page = doc.metadata.get("page", 0) + 1
                citations.append(f"{source} (Page {page})")
        else:
            # Agentic Fallback: Web Search
            try:
                web_results = search_tool.invoke(request.message)
                fallback_prompt = f"Answer the question based on this web search result: {web_results}\n\nQuestion: {request.message}"
                answer = llm.invoke(fallback_prompt)
                citations.append("🌐 Web Search (DuckDuckGo)")
            except Exception as e:
                answer = "I could not find the answer in your documents, and web search is currently unavailable."
                citations.append("Error")

        citations_str = ", ".join(list(set(citations)))
        
        # Save assistant message
        bot_msg = ChatMessage(session_id=request.session_id, role="assistant", content=answer, citations=citations_str)
        db.add(bot_msg)
        db.commit()
        db.close()

        return {
            "response": answer,
            "citations": citations_str
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}")
async def get_history(session_id: str):
    db = SessionLocal()
    messages = db.query(ChatMessage).filter(ChatMessage.session_id == session_id).all()
    db.close()
    return [{"role": m.role, "content": m.content, "citations": m.citations} for m in messages]
