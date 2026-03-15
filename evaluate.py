# evaluate.py

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextRecall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from src.chain import load_vectorstore, build_chain
from src.eval_dataset import EVAL_QUESTIONS
from src.config import GROQ_API_KEY


def run_evaluation():
    print("Loading vectorstore and chain...")
    vectorstore = load_vectorstore()
    chain = build_chain(vectorstore)

    print(f"Running {len(EVAL_QUESTIONS)} questions through RAG pipeline...")

    questions     = []
    answers       = []
    contexts      = []
    ground_truths = []

    for i, item in enumerate(EVAL_QUESTIONS):
        q  = item["question"]
        gt = item["ground_truth"]
        print(f"  [{i+1}/{len(EVAL_QUESTIONS)}] {q[:60]}...")

        answer = chain.invoke(q)

      
        if "don't have enough information" in answer.lower():
            print(f"    WARNING: No answer found for this question — skipping")
            continue

        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
        docs = retriever.invoke(q)
        context = [doc.page_content for doc in docs]

        questions.append(q)
        answers.append(answer)
        contexts.append(context)
        ground_truths.append(gt)

    if not questions:
        print("No answerable questions found — check your vectorstore")
        return

    print(f"\n{len(questions)}/{len(EVAL_QUESTIONS)} questions answered successfully")

    eval_dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts,
        "ground_truth": ground_truths,
    })

    # Wrap Groq as judge
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=GROQ_API_KEY,
            temperature=0,
        )
    )

    # Wrap local embeddings
    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
    )

    # Modern RAGAS v1.0 way — instantiate metrics as objects
    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextRecall(),
    ]

    # Assign judge to each metric
    for metric in metrics:
        metric.llm = judge_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = judge_embeddings

    print("\nRunning RAGAS evaluation (Groq as judge — free!)...")

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        raise_exceptions=False,
    )

    df = result.to_pandas()

    print("\nDataFrame columns:", df.columns.tolist())
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    # Safely read whichever columns exist
    score_map = {
        "faithfulness":     "Faithfulness",
        "answer_relevancy": "Answer relevancy",
        "context_recall":   "Context recall",
    }

    print("\nOverall scores:")
    found_cols = []
    for col, label in score_map.items():
        if col in df.columns:
            print(f"  {label}: {df[col].mean():.3f}")
            found_cols.append(col)

    if found_cols:
        print(f"\nPer-question breakdown:")
        print(df[found_cols].round(3).to_string(index=True))

    df.to_csv("evaluation_results.csv", index=False)
    print(f"\nFull results saved to evaluation_results.csv")

    return df


if __name__ == "__main__":
    run_evaluation()