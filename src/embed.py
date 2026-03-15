# src/embed.py

import json
import pickle
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter # pyright: ignore[reportMissingImports]
from langchain_community.embeddings import HuggingFaceEmbeddings # pyright: ignore[reportMissingImports]


PROCESSED_DIR = Path("data/processed")
VECTORSTORE_DIR = Path("data/vectorstore")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""],
)


def load_documents() -> list[dict]:
    """
    LEARNING NOTE: We load all processed JSONs and return a flat
    list of page dicts, each with text + metadata attached.
    Metadata (company, ticker, page number) is crucial — it's what
    lets us later tell the user WHERE the answer came from.
    """
    documents = []
    for json_path in PROCESSED_DIR.glob("*.json"):
        with open(json_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        for page in doc["pages"]:
            documents.append({
                "text": page["text"],
                "metadata": {
                    "company":     doc["company"],
                    "ticker":      doc["ticker"],
                    "year":        doc["year"],
                    "source":      doc["source"],
                    "page_number": page["page_number"],
                }
            })

    print(f"Loaded {len(documents)} pages from {PROCESSED_DIR}")
    return documents


def chunk_documents(documents: list[dict]) -> tuple[list, list]:
    """
    LEARNING NOTE: We split each page into chunks, but we carry
    the metadata forward onto every chunk. This is important —
    when FAISS returns a chunk, we need to know which company
    and page it came from so we can cite it in the answer.
    """
    all_chunks = []
    all_metadata = []

    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadata.append(doc["metadata"])

    print(f"Created {len(all_chunks):,} chunks from {len(documents)} pages")
    return all_chunks, all_metadata


def build_vectorstore(chunks: list, metadata: list):
    """
    LEARNING NOTE: What is HuggingFaceEmbeddings?
    We use the 'all-MiniLM-L6-v2' model from sentence-transformers.
    This is a small, fast, free embedding model that runs locally
    on your machine — no API key needed, no cost per embedding.

    It converts each chunk into a 384-dimensional vector.
    For 7 large annual reports we might have ~20,000 chunks,
    meaning we create ~20,000 vectors. FAISS stores them all
    and can search them in milliseconds.

    Why not use OpenAI embeddings?
    You could — they're more powerful. But they cost money per
    embedding call and require an API key. For building and
    iterating locally, a free local model is much faster.
    You can always swap to OpenAI embeddings later.
    """
    print("Loading embedding model (downloading on first run ~80MB)...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )

    print(f"Embedding {len(chunks):,} chunks into vectors...")
    print("This will take 2–5 minutes on first run...")

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embedding_model,
        metadatas=metadata,
    )

    # Save to disk so we never have to re-embed
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"Vectorstore saved to {VECTORSTORE_DIR}")

    return vectorstore


def main():
    print("--- embed.py started ---")

    # Step 1: load all processed JSON files
    documents = load_documents()

    # Step 2: chunk every page into smaller pieces
    chunks, metadata = chunk_documents(documents)

    # Step 3: embed chunks and store in FAISS
    vectorstore = build_vectorstore(chunks, metadata)

    # Step 4: quick sanity test — search for something
    print("\n--- Sanity check: searching for 'credit risk' ---")
    results = vectorstore.similarity_search("credit risk", k=3)
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Company: {result.metadata['company']}")
        print(f"  Page:    {result.metadata['page_number']}")
        print(f"  Text:    {result.page_content[:150]}...")

    print("\n--- embed.py finished ---")


if __name__ == "__main__":
    main()