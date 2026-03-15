from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VECTORSTORE_DIR = "data/vectorstore"
LLM_MODEL = "llama-3.1-8b-instant"   # free, fast, great for RAG
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_RESULTS = 5

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found — check your .env file")