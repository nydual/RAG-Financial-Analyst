# src/config.py

from dotenv import load_dotenv
import os

load_dotenv()

# try os.getenv first (local .env file)
# fall back to st.secrets (Streamlit Cloud)
def get_secret(key: str) -> str:
    value = os.getenv(key)
    if not value:
        try:
            import streamlit as st
            value = st.secrets[key]
        except Exception:
            pass
    return value

GROQ_API_KEY = get_secret("GROQ_API_KEY")
VECTORSTORE_DIR = "data/vectorstore"
LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RESULTS = 8

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found — check your .env file or Streamlit secrets")