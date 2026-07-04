<h1 align="center">🛡️ Private-RAG-in-a-Box V2</h1>

<p align="center">
  <strong>1-Click Local ChatGPT for your Private Documents. 100% Offline. Zero Data Leaks.</strong><br>
  <em>Now with Agentic Web Search, Multi-Model Support, and Source Citations!</em>
</p>

## ✨ What's New in V2?
- **📄 Source Citations:** AI answers now include the exact PDF file and page number they were sourced from. No more hallucinations.
- **🌐 Agentic Web Fallback:** If the answer isn't in your PDF, the AI automatically searches the internet (via DuckDuckGo) to find it.
- **🧠 Multi-Model Selector:** Switch instantly between `Llama-3`, `Mistral`, and `Phi-3` directly from the UI. Models download automatically in the background!
- **💾 Chat History:** Your conversations are now securely saved in a local SQLite database.
- **🎨 Premium UI:** Glassmorphism design, Dark Mode, and live status metrics.

## 📌 Why this exists?
Companies and individuals are terrified of uploading sensitive documents (financials, medical records, proprietary code) to OpenAI or Anthropic. 

**Private-RAG-in-a-Box** solves this by giving you a production-ready, fully local RAG (Retrieval-Augmented Generation) pipeline that runs entirely on your own hardware. No API keys required.

## 🏗️ Architecture
- **LLM Engine:** [Ollama](https://ollama.com/) (Running `Llama-3` and `nomic-embed-text`)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Persistent local storage)
- **Backend:** Python (FastAPI + LangChain)
- **Frontend:** Python (Streamlit)

## 🚀 Live Demo (100% Free Cloud GPU)

Don't want to install Docker? You can run the entire Private RAG system on a free Google GPU in the cloud.

<a href="https://colab.research.google.com/github/karidasd/Private-RAG-in-a-Box/blob/main/Colab_Demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

1. Click the badge above to open the Colab notebook.
2. Click **Runtime -> Run all** (or `Ctrl+F9`).
3. Scroll to the bottom and click the generated LocalTunnel link to access the Premium Chat UI live on the web!

---

## 💻 Local Quick Start (Docker)

Make sure you have [Docker](https://docs.docker.com/get-docker/) installed. Then run:

```bash
git clone https://github.com/karidasd/Private-RAG-in-a-Box.git
cd Private-RAG-in-a-Box
docker-compose up --build
```

**That's it!** The system will automatically download the LLM models on first run. 
Once it's ready, open your browser:
👉 **[http://localhost:8501](http://localhost:8501)**

## 💡 How to Use
1. Open the UI and upload your PDF files on the left sidebar.
2. Click **"Ingest Document"**. The backend will chunk and embed the text locally into the Vector Database.
3. Chat with your documents in the main window! The local Llama-3 model will answer based *only* on the context of your files.

---
*Created by [Dimitris Karydas](https://github.com/karidasd) - Senior AI Engineer.*
