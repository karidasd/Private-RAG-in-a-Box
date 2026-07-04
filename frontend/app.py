import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(page_title="DarkAIs Private RAG", page_icon="🛡️", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0E1117;
        color: #C9D1D9;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #58A6FF !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #238636 0%, #2EA043 100%);
        color: white;
        border: none;
        border-radius: 6px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.5);
        transform: translateY(-1px);
    }
    
    /* Chat Messages */
    .stChatMessage {
        background-color: #1C2128;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid #30363D;
    }
    
    /* Glassmorphism Title */
    .glass-title {
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363D;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-title">
    <h1 style='margin-bottom: 0;'>🛡️ DarkAIs Private RAG</h1>
    <p style='color: #8B949E; font-size: 1.1rem;'>Enterprise-Grade Local AI for Sensitive Documents.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=60)
    st.title("Control Panel")
    st.markdown("---")
    
    st.header("📂 1. Knowledge Base")
    uploaded_file = st.file_uploader("Drop PDF here", type=["pdf"], label_visibility="collapsed")
    
    if st.button("🚀 Ingest Document"):
        if uploaded_file is not None:
            with st.spinner("Processing Document... (Chunking & Embedding)"):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(f"✅ {response.json().get('message')}")
                    else:
                        st.error(f"❌ Error: {response.text}")
                except Exception as e:
                    st.error(f"❌ Connection failed: Ensure backend is running.")
        else:
            st.warning("⚠️ Please upload a PDF first.")
            
    st.markdown("---")
    st.markdown("### 📊 System Status")
    col1, col2 = st.columns(2)
    col1.metric("LLM Engine", "Llama-3", "Online")
    col2.metric("Vector DB", "Chroma", "Active")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! Upload a document on the left, then ask me anything about it. I process everything 100% locally."}]

for message in st.session_state.messages:
    avatar = "🤖" if message["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        with st.spinner("Analyzing vector embeddings..."):
            try:
                res = requests.post(f"{API_URL}/chat", json={"message": prompt})
                if res.status_code == 200:
                    answer = res.json().get("response", "No response.")
                else:
                    answer = f"Error: {res.text}"
            except Exception as e:
                answer = f"API Connection failed. Ensure the backend container is running."
        
        message_placeholder.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
