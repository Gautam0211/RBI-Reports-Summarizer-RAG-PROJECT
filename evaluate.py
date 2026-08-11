import sys
import io

# Force UTF-8 stdout/stderr on Windows — without this, redirecting output to
# a file (or some terminals) falls back to cp1252, which can't encode emoji
# like 🚀 used in the print statements below, causing UnicodeEncodeError.
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
import types
import warnings

# --- STUB FIX FOR UPSTREAM RAGAS VERTEXAI IMPORT BUG ---
_dummy = types.ModuleType("langchain_community.chat_models.vertexai")
_dummy.ChatVertexAI = type("ChatVertexAI", (object,), {})
sys.modules["langchain_community.chat_models.vertexai"] = _dummy
# --------------------------------------------------------

# Suppress non-critical deprecation and sampling warnings for clean output
warnings.filterwarnings("ignore")

import os
import pandas as pd
from dotenv import load_dotenv

# Import from the plain module — NOT from app.py
from rag_core import hybrid_retrieve_candidates, rerank_chunks, generate_constrained_answer, gemini_client

from ragas import evaluate
from ragas.run_config import RunConfig  # Correct RAGAS concurrency controller
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# --- Medium-level test set ---
TEST_SET = [
    {
        "question": "How did the composition of India's Foreign Currency Assets (FCA) change between September 2025 and March 2026?",
        "ground_truth": "FCA declined from USD 579.181 billion to USD 552.283 billion. The share invested in securities decreased slightly from 84.52% to 84.31%, while deposits with other central banks and the BIS increased from 7.96% to 8.48%. Deposits with overseas commercial banks declined from 7.52% to 7.21%."
    },
    {
        "question": "What were the main sources of variation in India's foreign exchange reserves during April–December 2025, and what was their overall effect on reserves?",
        "ground_truth": "The current account balance contributed −USD 30.2 billion, the net capital account contributed −USD 0.6 billion, and valuation changes contributed +USD 50.2 billion. Overall, reserves increased by USD 19.4 billion during April–December 2025."
    },
    {
        "question": "What were the key factors supporting economic activity in India according to the April 2026 Monetary Policy Report?",
        "ground_truth": "The report highlights continued resilience in domestic economic activity, supported by factors such as strong domestic demand, improving investment activity, and favorable developments in key sectors of the economy."
    },
    {
        "question": "How did the April 2026 Monetary Policy Report assess the outlook for inflation, and what were the major risks to the inflation trajectory?",
        "ground_truth": "The report assessed the inflation outlook in the context of evolving domestic and global conditions. The major risks included movements in commodity prices, especially energy prices, global financial conditions, exchange-rate movements, and supply-side pressures."
    },
    {
        "question": "According to the Monetary Policy Report, what was the exact amount of RBI's foreign exchange market intervention in January 2026?",
        "ground_truth": "Insufficient information in the given context. The provided context does not specify the exact amount of RBI's foreign exchange market intervention in January 2026."
    }
]

def run_evaluation_benchmark():
    print("🚀 Starting End-to-End RAG Evaluation Benchmark...")

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for item in TEST_SET:
        query = item["question"]
        print(f"\n🔎 Querying: '{query}'")

        candidates = hybrid_retrieve_candidates(query, top_k=15)
        top_chunks = rerank_chunks(query, candidates, top_n=4)
        retrieved_contexts = [pt.payload.get("chunk_text_content", "") for pt in top_chunks]

        generated_answer = generate_constrained_answer(query, top_chunks)

        questions.append(query)
        answers.append(generated_answer)
        contexts_list.append(retrieved_contexts)
        ground_truths.append(item["ground_truth"])

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    })

    api_key = os.getenv("GOOGLE_API_KEY")

    # Use stable gemini-2.5-flash with retries
    evaluator_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-3.6-flash", 
            google_api_key=api_key,
            max_retries=5
        )
    )
    
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=api_key
        )
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    for metric in metrics:
        metric.llm = evaluator_llm
        if hasattr(metric, 'embeddings'):
            metric.embeddings = evaluator_embeddings

    # Enforce max_workers=1 via RunConfig to avoid API rate limit (429) errors that cause NaNs
    eval_run_config = RunConfig(max_workers=1, timeout=60, max_retries=5)

    print("\n📊 Computing Faithfulness, Relevancy, Precision, and Recall scores via Gemini Judge...")
    try:
        results = evaluate(
            dataset=dataset, 
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            run_config=eval_run_config
        )
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        return

    df_res = results.to_pandas()
    
    # Re-attach questions so printing df_res[["question", ...]] doesn't throw KeyError
    df_res.insert(0, "question", [item["question"] for item in TEST_SET])

    print("\n=================== EVALUATION RESULTS ===================")
    print(df_res[["question", "faithfulness", "answer_relevancy", "context_precision", "context_recall"]])
    print("==========================================================")

    df_res.to_csv("rag_evaluation_results.csv", index=False)
    print("\n✅ Evaluation complete! Results exported to 'rag_evaluation_results.csv'.")

if __name__ == "__main__":
    run_evaluation_benchmark()