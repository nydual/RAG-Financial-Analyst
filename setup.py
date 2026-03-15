# setup.py
import os
from pathlib import Path

def setup():
    vectorstore_path = Path("data/vectorstore")

    if vectorstore_path.exists() and any(vectorstore_path.iterdir()):
        print("Vectorstore already exists — skipping rebuild")
        return

    print("Vectorstore not found — rebuilding...")
    from src.embed import main as build_embeddings
    build_embeddings()
    print("Vectorstore ready")

if __name__ == "__main__":
    setup()
