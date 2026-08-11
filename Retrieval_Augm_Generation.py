import os
import time
from dotenv import load_dotenv

# Production Gemini Client (for Final Generation) and Qdrant Client
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector  # FIX: import SparseVector

# Local Embeddings & Cross-Encoder Re-ranker
from fastembed import TextEmbedding, SparseTextEmbedding
from sentence_transformers import CrossEncoder

load_dotenv()

# --- Configurations ---
COLLECTION_NAME = "rbi_hybrid_summarizer"
DENSE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DENSE_VECTOR_NAME = "dense"        # Matched to 2.py
SPARSE_VECTOR_NAME = "text-sparse" # Matched to 2.py
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

TOP_K_RETRIEVAL = 15   # Candidates fetched via Hybrid Search
TOP_K_RERANK = 4       # Chunks fed to Gemini after re-ranking

# FIX: retry transient 503s, and fall back to alternate models if the
# primary is persistently overloaded.
GEMINI_MODEL_CHAIN = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-pro-preview"]
GEMINI_MAX_RETRIES = 4
GEMINI_RETRY_BASE_DELAY = 3  # seconds, doubles each retry

# 1. Initialize Clients
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("❌ GOOGLE_API_KEY is missing from your .env file!")

gemini_client = genai.Client(api_key=google_api_key)

qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if qdrant_url and qdrant_api_key:
    print("🌐 Connecting to Qdrant Cloud Cluster...")
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
else:
    print("🏠 Connecting to local Qdrant database...")
    qdrant_client = QdrantClient(path="./qdrant_storage_db")

# 2. Initialize Local Embedding Models (Matching 2.py)
print(f"📦 Loading local dense query model: '{DENSE_MODEL_NAME}'...")
dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)

print("📦 Loading local sparse (BM25) query model...")
sparse_model = SparseTextEmbedding("Qdrant/bm25")

# 3. Load Local Re-ranker Model
print(f"🔄 Loading Cross-Encoder Re-ranker ('{RERANKER_MODEL_NAME}')...")
reranker = CrossEncoder(RERANKER_MODEL_NAME)


def hybrid_retrieve_candidates(query_text: str, top_k: int = TOP_K_RETRIEVAL):
    """Stage 1: Fetch candidates using local FastEmbed vectors & Qdrant RRF Fusion"""
    # Local Dense Query Vector (bge-base-en-v1.5)
    dense_query_vector = list(dense_model.embed([query_text]))[0].tolist()

    # Local Sparse Keyword Query Vector (BM25)
    sparse_emb = list(sparse_model.embed([query_text]))[0]

    # FIX: wrap the raw fastembed sparse embedding into a proper Qdrant SparseVector,
    # exactly as done during indexing in 2.py. Passing the raw fastembed object
    # directly will fail Qdrant's validation / produce incorrect query results.
    sparse_query_vector = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist()
    )

    response = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_query_vector,
                using=DENSE_VECTOR_NAME,  # "dense"
                limit=top_k
            ),
            Prefetch(
                query=sparse_query_vector,
                using=SPARSE_VECTOR_NAME,  # "text-sparse"
                limit=top_k
            )
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k
    )
    return response.points


def rerank_chunks(query_text: str, candidate_points, top_n: int = TOP_K_RERANK):
    """Stage 2: Cross-Encoder Scoring to isolate top-matching context"""
    if not candidate_points:
        return []

    pairs = [
        [query_text, point.payload.get("chunk_text_content", "")]
        for point in candidate_points
    ]

    scores = reranker.predict(pairs)

    scored_points = []
    for idx, point in enumerate(candidate_points):
        scored_points.append((scores[idx], point))

    scored_points.sort(key=lambda x: x[0], reverse=True)
    return [point for score, point in scored_points[:top_n]]


def generate_with_retry(prompt: str):
    """Call Gemini with retry/backoff on transient errors (e.g. 503 UNAVAILABLE),
    falling back through GEMINI_MODEL_CHAIN if a model stays overloaded."""
    last_err = None
    for model_name in GEMINI_MODEL_CHAIN:
        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if attempt > 1 or model_name != GEMINI_MODEL_CHAIN[0]:
                    print(f"    ✅ Succeeded with model '{model_name}' on attempt {attempt}.")
                return response
            except Exception as e:
                last_err = e
                status = getattr(e, "code", None) or getattr(e, "status_code", None)
                # Only retry on transient/server errors; fail fast on real client errors (400, 401, etc.)
                is_retryable = status in (429, 500, 502, 503, 504) or "UNAVAILABLE" in str(e) or "high demand" in str(e)
                if not is_retryable:
                    raise
                delay = GEMINI_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                print(f"    ⚠️ '{model_name}' attempt {attempt}/{GEMINI_MAX_RETRIES} failed "
                      f"({e.__class__.__name__}). Retrying in {delay}s...")
                time.sleep(delay)
        print(f"    ⏭️ '{model_name}' exhausted retries, falling back to next model in chain...")

    raise RuntimeError(f"All models in GEMINI_MODEL_CHAIN failed. Last error: {last_err}")


def generate_constrained_answer(query_text: str, top_chunks):
    """Stage 3: Constrained Answer Generation using Gemini 2.5 Flash"""
    if not top_chunks:
        # FIX: handle the no-candidates case explicitly instead of sending
        # an empty context block to Gemini.
        return ("Insufficient information in the provided RBI reports.\n\n"
                "### 📚 Sources Referenced:\n(none)")

    context_blocks = []
    sources = set()

    for i, point in enumerate(top_chunks, 1):
        payload = point.payload
        source_file = payload.get("source_file", "Unknown File")
        h1 = payload.get("heading_h1", "N/A")
        h2 = payload.get("heading_h2", "N/A")
        content = payload.get("chunk_text_content", "").strip()

        header_path = f"{h1} > {h2}" if h2 != "N/A" else h1
        sources.add(f"- **{source_file}** (Section: {header_path})")

        block = f"[CONTEXT BLOCK #{i}]\nSource File: {source_file}\nSection: {header_path}\nContent:\n{content}"
        context_blocks.append(block)

    full_context = "\n\n----------------------------------------\n\n".join(context_blocks)

    prompt = f"""You are a highly analytical macroeconomic and financial summarizer for Reserve Bank of India (RBI) reports.

### RETRIEVED CONTEXT FROM RBI REPORTS:
{full_context}

### USER QUESTION:
{query_text}

### CONSTRAINED GENERATION RULES:
1. Answer strictly using ONLY the provided RBI context above. If the context does not contain the answer, output: "Insufficient information in the provided RBI reports."
2. Output strictly in clear, crisp Markdown bullet points.
3. Limit each bullet point to a maximum of 20 words per bullet.
4. Prioritize quantitative data, percentages, basis points, and specific macroeconomic metrics.
5. Do NOT write long prose paragraphs or essays.
6. Never make up facts, names, dates, or numbers.
"""

    response = generate_with_retry(prompt)

    citation_block = "\n\n### 📚 Sources Referenced:\n" + "\n".join(sorted(list(sources)))
    return response.text + citation_block


def ask_rbi_rag(user_query: str):
    """Full End-to-End Execution Pipeline"""
    print("\n==================================================")
    print(f"❓ USER QUERY: '{user_query}'")
    print("==================================================")

    # Step 1: Hybrid Retrieval
    print(f"\n🔍 Step 1: Fetching top {TOP_K_RETRIEVAL} candidates via Hybrid RRF Search...")
    candidates = hybrid_retrieve_candidates(user_query, top_k=TOP_K_RETRIEVAL)
    print(f"  ↳ Retrieved {len(candidates)} candidate chunks from Qdrant.")

    # Step 2: Re-ranking
    print(f"\n🎯 Step 2: Re-ranking candidates using Cross-Encoder ('{RERANKER_MODEL_NAME}')...")
    top_chunks = rerank_chunks(user_query, candidates, top_n=TOP_K_RERANK)
    print(f"  ↳ Filtered down to top {len(top_chunks)} highest quality context blocks.")

    # Step 3: Constrained Generation
    print("\n⚡ Step 3: Generating constrained answer via Gemini 3.6 Flash...\n")
    final_answer = generate_constrained_answer(user_query, top_chunks)

    print("=" * 60)
    print(final_answer)
    print("=" * 60)


if __name__ == "__main__":
    # Test Query on Monetary Policy
    sample_query = "What is the projected real GDP growth rate and CPI inflation forecast for FY27?"
    ask_rbi_rag(sample_query)