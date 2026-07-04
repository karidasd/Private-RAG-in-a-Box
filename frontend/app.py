import streamlit as st
import requests
import os
import uuid

API_URL = os.getenv("API_URL", "http://backend:8000")

st.set_page_config(page_title="DarkAIs Private RAG V3", page_icon="🛡️", layout="wide")

# Custom CSS
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
    <h1 style='margin-bottom: 0;'>🛡️ DarkAIs Private RAG V3</h1>
    <p style='color: #8B949E; font-size: 1.1rem;'>Semantic Chunking • Web URL Ingestion • Vision RAG (Llava) • Citations</p>
</div>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=60)
    st.title("Control Panel")
    
    st.header("🧠 1. Select Model")
    selected_model = st.selectbox("LLM Engine", ["llama3", "mistral", "phi3", "llava"], help="Llama3 is default. Llava is for Vision RAG.")
    
    st.markdown("---")
    st.header("📂 2. Knowledge Base")
    
    tab1, tab2, tab3 = st.tabs(["📄 PDF", "🌐 URL", "👁️ Image"])
    
    with tab1:
        uploaded_pdf = st.file_uploader("Drop PDF", type=["pdf"], label_visibility="collapsed")
        if st.button("🚀 Ingest PDF"):
            if uploaded_pdf:
                with st.spinner("Processing Semantic Chunks..."):
                    files = {"file": (uploaded_pdf.name, uploaded_pdf, "application/pdf")}
                    try:
                        res = requests.post(f"{API_URL}/upload", files=files)
                        st.success(res.json().get('message')) if res.status_code == 200 else st.error("Error")
                    except Exception:
                        st.error("Connection failed.")
            else:
                st.warning("Upload a PDF first.")
                
    with tab2:
        url_input = st.text_input("Enter Website URL", placeholder="https://en.wikipedia.org/wiki/AI")
        if st.button("🚀 Ingest URL"):
            if url_input:
                with st.spinner("Scraping and Chunking URL..."):
                    try:
                        res = requests.post(f"{API_URL}/ingest-url", data={"url": url_input})
                        st.success(res.json().get('message')) if res.status_code == 200 else st.error("Error")
                    except Exception:
                        st.error("Connection failed.")
            else:
                st.warning("Enter a URL first.")
                
    with tab3:
        st.info("Upload an image and switch model to 'llava' to ask questions about it.")
        uploaded_img = st.file_uploader("Drop Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
        if st.button("🚀 Load Image"):
            if uploaded_img:
                with st.spinner("Loading Image..."):
                    files = {"file": (uploaded_img.name, uploaded_img, "image/jpeg")}
                    try:
                        res = requests.post(f"{API_URL}/ingest-image", files=files)
                        st.success(res.json().get('message')) if res.status_code == 200 else st.error("Error")
                    except Exception:
                        st.error("Connection failed.")
            else:
                st.warning("Upload an image first.")
            
    st.markdown("---")
    st.header("💬 3. Chat Session")
    st.code(f"Session: {st.session_state.session_id[:8]}", language="text")
    if st.button("🔄 New Chat"):
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

messages = []
try:
    history_res = requests.get(f"{API_URL}/history/{st.session_state.session_id}")
    if history_res.status_code == 200:
        messages = history_res.json()
except:
    pass

if not messages:
    messages = [{"role": "assistant", "content": "Welcome to V3! I can now read PDFs (with Semantic Chunking), scrape Web URLs, and analyze Images (if you select Llava). What do you want to do?", "citations": None}]

for msg in messages:
    avatar = "🤖" if msg["role"] == "assistant" else "🧑‍💻"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("citations") and msg["citations"] != "Error" and msg["citations"] != "":
            st.markdown(f"<div class='citation-badge'>📚 Sources: {msg['citations']}</div>", unsafe_allow_html=True)

if prompt := st.chat_input("Ask a question..."):
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        with st.spinner(f"Analyzing with {selected_model}..."):
            try:
                payload = {"message": prompt, "session_id": st.session_state.session_id, "model": selected_model}
                res = requests.post(f"{API_URL}/chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    answer, citations = data.get("response", "No response."), data.get("citations", "")
                else:
                    answer, citations = f"Error: {res.text}", ""
            except Exception:
                answer, citations = "API Connection failed.", ""
        
        message_placeholder.markdown(answer)
        if citations and citations != "Error" and citations != "":
            st.markdown(f"<div class='citation-badge'>📚 Sources: {citations}</div>", unsafe_allow_html=True)
            
    st.rerun()
