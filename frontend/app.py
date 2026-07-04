import streamlit as st
import requests

API_URL = "http://backend:8000"

st.set_page_config(page_title="Private RAG Box", page_icon="🛡️")

st.title("🛡️ Private-RAG-in-a-Box")
st.markdown("Chat with your PDFs 100% locally and privately.")

# Sidebar for file upload
with st.sidebar:
    st.header("1. Upload Knowledge")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if st.button("Ingest Document"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF (Chunking & Embedding)..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                try:
                    response = requests.post(f"{API_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(response.json().get("message", "Success!"))
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: {str(e)}")
        else:
            st.warning("Please select a file first.")

st.header("2. Chat with your Data")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask something about your documents..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    with st.spinner("Thinking..."):
        try:
            res = requests.post(f"{API_URL}/chat", json={"message": prompt})
            if res.status_code == 200:
                answer = res.json().get("response", "No response.")
            else:
                answer = f"Error: {res.text}"
        except Exception as e:
            answer = f"API Connection failed: {str(e)}"
            
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
