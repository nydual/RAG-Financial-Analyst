# src/chain.py

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.config import (
    GROQ_API_KEY,
    VECTORSTORE_DIR,
    LLM_MODEL,
    EMBEDDING_MODEL,
    TOP_K_RESULTS,
)


# in src/chain.py — replace PROMPT_TEMPLATE with this:

PROMPT_TEMPLATE = """You are an expert financial analyst specializing in \
Canadian public companies. Answer the user's question using the context \
provided below from company annual reports.

Important instructions:
- Use information from the context to give a detailed answer
- If multiple companies are mentioned, compare them using the context
- Cite your sources like this: [Company: RBC, Page: 45]
- Only if the context contains absolutely no relevant information at all, \
say you cannot find it
- Do NOT say you lack information if the context contains related content \
— use what is available

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore() -> FAISS:

    print("Loading embedding model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

    print("Loading FAISS vectorstore from disk...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    print(f"Vectorstore loaded successfully")
    return vectorstore


def build_chain(vectorstore: FAISS, company_filter: str = None):
  
    if company_filter:
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": TOP_K_RESULTS,
                "filter": {"company": company_filter},
            }
        )
    else:
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": TOP_K_RESULTS}
        )

    # Set up the LLM
    llm = ChatGroq(
    model=LLM_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
    )
    
    # Set up the prompt
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    def format_docs(docs):
      
        formatted = []
        for doc in docs:
            meta = doc.metadata
            chunk_text = (
                f"[Company: {meta['company']}, Page: {meta['page_number']}]\n"
                f"{doc.page_content}"
            )
            formatted.append(chunk_text)
        return "\n\n---\n\n".join(formatted)

    # Build the chain using LangChain's pipe operator
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def ask(chain, question: str) -> str:
    """Simple wrapper to invoke the chain and return the answer."""
    return chain.invoke(question)