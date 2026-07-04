import streamlit as st
import requests
import os
import uuid

API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(page_title="DarkAIs Private RAG V2", page_icon="🛡️", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: #C9D1D9; }
    h1, h2, h3 { color: #58A6FF !important; font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: #161B22; border-right: 1px solid #30363D; }
    .stButton>button { background: linear-gradient(90deg, #238636 0%, #2EA043 100%); color: white; border: none; border-radius: 6px; font-weight: 600; width: 100%; transition: all 0.3s ease; }
    .stButton>button:hover { box-shadow: 0 0 10px rgba(46, 160, 67, 0.5); transform: translateY(-1px); }
    .glass-title { background: rgba(22, 27, 34, 0.7); backdrop-filter: blur(10px); padding: 20px; border-radius: 12px; border: 1px solid #30363D; text-align: center; margin-bottom: 30px; }
    .citation-badge { background-color: #1F6FEB; color: white; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; margin-top: 10px; display: inline-block; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="glass-title">
    <h1 style='margin-bottom: 0;'>🛡️ DarkAIs Private RAG V2</h1>
    <p style='color: #8B949E; font-size: 1.1rem;'>Enterprise-Grade Local AI with Agentic Search & Citations.</p>
</div>
""", unsafe_allow_html=True)

# Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=60)
    st.title("Control Panel")
    
    st.header("🧠 1. Select Model")
    selected_model = st.selectbox("LLM Engine", ["llama3", "mistral", "phi3"], help="Llama3 is default. If you select Mistral/Phi3, Ollama will download it automatically!")
    
    st.markdown("---")
    st.header("📂 2. Knowledge Base")
    uploaded_file = st.file_uploader("Drop PDF here", type=["pdf"], label_visibility="collapsed")
    
    if st.button("🚀 Ingest Document"):
        if uploaded_file is not None:
            with st.spinner("Processing Document..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(f"✅ {response.json().get('message')}")
                    else:
                        st.error(f"❌ Error: {response.text}")
                except Exception as e:
                    st.error("❌ Connection failed.")
        else:
            st.warning("⚠️ Please upload a PDF first.")
            
    st.markdown("---")
    st.header("💬 3. Chat Session")
    st.code(f"Session: {st.session_state.session_id[:8]}", language="text")
    if st.button("🔄 New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# Load History
messages = []
try:
    history_res = requests.get(f"{API_URL}/history/{st.session_state.session_id}")
    if history_res.status_code == 200:
        messages = history_res.json()
except:
    pass

if not messages:
    messages = [{"role": "assistant", "content": "Welcome! Upload a document on the left, or ask me anything. If I don't know it, I will automatically search the web (Agentic Fallback)!", "citations": None}]

# Display chat messages
for msg in messages:
    avatar = "🤖" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("citations") and msg["citations"] != "Error" and msg["citations"] != "":
            st.markdown(f"<div class='citation-badge'>📚 Sources: {msg['citations']}</div>", unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        with st.spinner(f"Analyzing with {selected_model}..."):
            try:
                payload = {
                    "message": prompt,
                    "session_id": st.session_state.session_id,
                    "model": selected_model
                }
                res = requests.post(f"{API_URL}/chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("response", "No response.")
                    citations = data.get("citations", "")
                else:
                    answer = f"Error: {res.text}"
                    citations = ""
            except Exception as e:
                answer = "API Connection failed."
                citations = ""
        
        message_placeholder.markdown(answer)
        if citations and citations != "Error" and citations != "":
            st.markdown(f"<div class='citation-badge'>📚 Sources: {citations}</div>", unsafe_allow_html=True)
            
    st.rerun()
