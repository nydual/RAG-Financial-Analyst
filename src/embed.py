# src/embed.py
import json
import pickle
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PROCESSED_DIR = Path("data/processed")
VECTORSTORE_DIR = Path("data/vectorstore")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " ", ""],
)

def load_documents() -> list[dict]:

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

    #load all processed JSON files
    documents = load_documents()

    #chunk every page into smaller pieces
    chunks, metadata = chunk_documents(documents)

    #embed chunks and store in FAISS
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