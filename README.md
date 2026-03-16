# FinSight — Canadian Financial Analyst

> A conversational AI that answers questions about Canadian public company annual reports using Retrieval-Augmented Generation (RAG).

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://rag-financial-analyst-hhuxki3k5chdkhabubpxjq.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/nydual/RAG-Financial-Analyst)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge)](https://langchain.com)

---

## Live Demo

**Try it here:** [https://rag-financial-analyst-hhuxki3k5chdkhabubpxjq.streamlit.app](https://rag-financial-analyst-hhuxki3k5chdkhabubpxjq.streamlit.app/)

> Ask questions like:
> - *"What are RBC's main credit risk factors?"*
> - *"How did Shopify's revenue grow in FY2025?"*
> - *"Compare TD Bank and BMO's capital ratios"*
> - *"What is Suncor's energy transition strategy?"*

---

## Overview

FinSight is a RAG (Retrieval-Augmented Generation) pipeline that allows users to chat with 7 Canadian public company annual reports. Instead of relying on an LLM's training data, FinSight retrieves relevant passages directly from the source documents and grounds every answer in cited evidence preventing hallucination and ensuring accuracy.

This project demonstrates end-to-end ML engineering: data acquisition, document processing, vector indexing, LLM integration, evaluation, and production deployment.

---

## Companies Covered

| Company | Ticker | Report Year |
|---|---|---|
| Shopify | SHOP | FY2025 |
| Royal Bank of Canada | RY | FY2025 |
| TD Bank | TD | FY2025 |
| BMO | BMO | FY2025 |
| CIBC | CM | FY2025 |
| Suncor Energy | SU | FY2025 |
| CN Rail | CNR | FY2024 |

All reports sourced from SEDAR+ and company investor relations pages.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                    │
│                                                          │
│  SEDAR+ PDFs → pdfplumber → JSON → Chunker → Embedder   │
│                                    ↓            ↓        │
│                               500-char      MiniLM-L6   │
│                               chunks        vectors      │
│                                                  ↓       │
│                                            FAISS index   │
└─────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────┐
│                     QUERY PIPELINE                       │
│                                                          │
│  User question → Embed → FAISS search → Top-5 chunks    │
│                                               ↓          │
│                                      Prompt template     │
│                                               ↓          │
│                                     LLaMA 3.1 via Groq  │
│                                               ↓          │
│                                   Answer + citations     │
└─────────────────────────────────────────────────────────┘
```

**Key design decision:** The LLM is instructed to answer using *only* the retrieved context and cite every claim with `[Company: X, Page: Y]`. This grounds answers in evidence and prevents hallucination.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Document parsing | `pdfplumber` |
| Text chunking | `LangChain RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local) |
| Vector store | `FAISS` (~12,000 chunks) |
| LLM | `LLaMA 3.1 8B` via `Groq API` (free, fast) |
| Framework | `LangChain` |
| UI | `Streamlit` |
| Evaluation | `RAGAS` (LLM-as-judge) |
| Deployment | `Streamlit Cloud` |

---

## Evaluation (RAGAS)

The pipeline was evaluated using RAGAS — an LLM-as-judge framework that measures RAG quality without requiring human annotation.

| Metric | Score | Description |
|---|---|---|
| **Answer relevancy** | **0.912** | Answers directly address what was asked |
| **Context recall** | **0.700** | FAISS retrieves passages containing the answer |
| **Faithfulness** | **0.739** | Claims are grounded in retrieved documents |

**Notable finding:** Context recall was lower for RBC documents due to table-of-contents chunks polluting the FAISS index. Proposed fix: filter chunks under 200 characters during ingestion.

---

## Project Structure

```
rag-financial-analyst/
├── data/
│   ├── raw/              # Annual report PDFs (excluded from git)
│   ├── processed/        # Extracted text as JSON (7 files)
│   └── vectorstore/      # FAISS index (rebuilt on deploy)
├── src/
│   ├── ingest.py         # PDF text extraction with pdfplumber
│   ├── embed.py          # Chunking + embeddings + FAISS build
│   ├── chain.py          # LangChain RAG chain + LLM
│   ├── retriever.py      # FAISS retrieval wrapper
│   ├── config.py         # Environment variables
│   └── eval_dataset.py   # Ground truth Q&A pairs for RAGAS
├── app.py                # Streamlit chat UI
├── evaluate.py           # RAGAS evaluation script
├── setup.py              # Vectorstore rebuild on deployment
└── requirements.txt
```

---

## Local Setup

### Prerequisites
- Python 3.11
- Groq API key (free at [console.groq.com](https://console.groq.com))
- Annual report PDFs from SEDAR+ or company investor pages

### Installation

```bash
# Clone the repo
git clone https://github.com/nydual/RAG-Financial-Analyst.git
cd RAG-Financial-Analyst

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy the example env file
cp .env.example .env

# Add your Groq API key to .env
GROQ_API_KEY=your_key_here
```

### Running the pipeline

```bash
# Step 1: Add your PDFs to data/raw/
# Name them: SHOP_2025.pdf, RBC_2025.pdf, TD_2025.pdf, etc.

# Step 2: Extract text from PDFs
python src/ingest.py

# Step 3: Build FAISS vector index
python src/embed.py

# Step 4: Launch the app
streamlit run app.py
```

### Running evaluation

```bash
python evaluate.py
# Results saved to evaluation_results.csv
```

---

## Key Learning Points

This project covers the full RAG engineering lifecycle:

- **Data acquisition** — programmatic PDF extraction with metadata preservation
- **Chunking strategy** — `RecursiveCharacterTextSplitter` with 500-char chunks and 50-char overlap to preserve sentence boundaries
- **Embedding model selection** — free local `all-MiniLM-L6-v2` vs paid OpenAI embeddings tradeoff
- **Prompt engineering** — constraining the LLM to only use retrieved context to prevent hallucination
- **RAG evaluation** — RAGAS metrics (faithfulness, answer relevancy, context recall) using LLM-as-judge
- **Production deployment** — secrets management, vectorstore rebuild on cold start, Streamlit Cloud

---

## What I'd Improve Next

- Filter table-of-contents chunks (< 200 chars) during ingestion to improve context recall
- Add a query rewriting step to handle ambiguous questions
- Implement hybrid search (BM25 + dense vectors) for better retrieval on financial terminology
- Add a comparison mode — side-by-side answers across multiple companies
- Connect `yfinance` for live stock price data in the ticker tape

---

## Author

Built by [@nydual](https://github.com/nydual) 

---

*Data sourced from SEDAR+ and company investor relations pages. For informational purposes only — not financial advice.*
