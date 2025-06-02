import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import os
from groq import Groq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Groq Friendly AI Assistant", page_icon="🤖")
st.title("🤖 Groq Friendly AI Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_text" not in st.session_state:
    st.session_state.processed_text = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

GROQ_MODEL = "llama-3.3-70b-versatile"  # Latest as of July 2025

EMBEDDINGS = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L12-v1",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

SMART_FRIENDLY_PROMPT = """
You are a very friendly, helpful, and smart AI assistant.
- Always answer in the style
- If the user asks for an explanation, explain clearly and simply.
- If the user asks a math question, show steps and the answer unless told otherwise.
- If the user asks for help or has a doubt, answer as a helpful, supportive friend.
- If the user's question is funny or asks for a joke, add a light joke or witty remark to your answer, but keep it relevant and concise.
- If you are unsure, say so honestly.
- Never provide extra information unless the user requests it.
- Always be positive and encouraging. You can also joke in between.
if you want you can use emoji too. but don't always answer in one word
"""

def process_files(files):
    text = ""
    for file in files:
        try:
            if file.type == "application/pdf":
                pdf_reader = PdfReader(file)
                text += "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
            elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(file)
                text += df.to_string()
            elif file.type == "text/plain":
                text += file.getvalue().decode()
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
    if text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        st.session_state.vector_store = FAISS.from_texts(chunks, EMBEDDINGS)
    return text

api_key = st.sidebar.text_input(
    "Enter Groq API Key:",
    type="password",
    value=os.getenv("GROQ_API_KEY") or ""
)

uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF/Excel/Text)",
    type=["pdf", "xlsx", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Processing documents..."):
        st.session_state.processed_text = process_files(uploaded_files)

user_query = st.chat_input("Ask me anything...")

if user_query and api_key:
    try:
        client = Groq(api_key=api_key)
        context = ""
        if st.session_state.vector_store:
            docs = st.session_state.vector_store.similarity_search(user_query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])

        # for conversation history
        messages = []
        for msg in st.session_state.chat_history:
            messages.append({"role": "user", "content": msg["user"]})
            messages.append({"role": "assistant", "content": msg["assistant"]})

        # Add current user message, with prompt and context if any
        if context:
            user_content = f"{SMART_FRIENDLY_PROMPT}\n\nContext: {context}\n\nUser: {user_query}\nAI:"
        else:
            user_content = f"{SMART_FRIENDLY_PROMPT}\n\nUser: {user_query}\nAI:"
        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            messages=messages,
            model=GROQ_MODEL,
            temperature=0.7
        )
        # Some Groq clients use .choices[0].message.content, others .choices.message.content
        answer = response.choices[0].message.content if hasattr(response.choices[0], "message") else response.choices[0].content
        st.session_state.chat_history.append({
            "user": user_query,
            "assistant": answer,
            "mode": "Document Analysis" if context else "General Chat"
        })
    except Exception as e:
        st.error(f"Error: {str(e)}")

for message in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(message["user"])
    with st.chat_message("assistant"):
        st.write(message["assistant"])
        st.caption(f"Mode: {message['mode']}")

if st.session_state.chat_history:
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if st.session_state.processed_text:
    if st.sidebar.button("Clear Documents"):
        st.session_state.processed_text = ""
        st.session_state.vector_store = None
        st.rerun()

