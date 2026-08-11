import os
import time
from dotenv import load_dotenv

from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseVector,
    Prefetch, FusionQuery, Fusion, Filter, FieldCondition, MatchAny
)
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from fastembed import TextEmbedding, SparseTextEmbedding
from sentence_transformers import CrossEncoder

load_dotenv()

# --- Constants (kept identical to app4.py) ---
COLLECTION_NAME = "rbi_hybrid_summarizer"
DENSE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DENSE_DIM = 768
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "text-sparse"
SOURCE_FILE_FIELD = "source_file"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

GEMINI_MODEL_CHAIN = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-pro-preview"]
GEMINI_MAX_RETRIES = 4
GEMINI_RETRY_BASE_DELAY = 3

STARTUP_NET_MAX_RETRIES = 3
STARTUP_NET_RETRY_BASE_DELAY = 2

# Retry settings for hot-path Qdrant queries (retrieval, not just startup) —
# guards against transient "ReadTimeout"/ResponseHandlingException blips on
# Qdrant Cloud during longer-running sessions like evaluation runs.
QUERY_MAX_RETRIES = 3
QUERY_RETRY_BASE_DELAY = 2


def get_secret(key_name: str, default=None):
    """Plain os.getenv resolver — no st.secrets here since this module has
    no Streamlit dependency. app4.py has its own get_secret() that also
    checks st.secrets; that one is used for anything called from the UI."""
    return os.getenv(key_name, default)


def _with_retry(fn, *args, max_retries=STARTUP_NET_MAX_RETRIES, base_delay=STARTUP_NET_RETRY_BASE_DELAY, **kwargs):
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(base_delay * attempt)
    raise last_err


# --- Client & Model Initialization (module-level, runs once on first import) ---
_api_key = get_secret("GOOGLE_API_KEY")
if not _api_key:
    raise ValueError("❌ GOOGLE_API_KEY is missing from your .env file!")
gemini_client = genai.Client(api_key=_api_key)

_qdrant_url = get_secret("QDRANT_URL")
_qdrant_api_key = get_secret("QDRANT_API_KEY")
if _qdrant_url and _qdrant_api_key:
    # Timeout raised from 60s -> 120s: evaluation runs (RAGAS) can hit a
    # slower/cold Qdrant Cloud cluster after several back-to-back queries,
    # and a longer timeout gives it room to respond instead of hard-failing.
    qdrant_client = QdrantClient(url=_qdrant_url, api_key=_qdrant_api_key, timeout=120)
else:
    qdrant_client = QdrantClient(path="./qdrant_storage_db")

dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
sparse_model = SparseTextEmbedding("Qdrant/bm25")
reranker = CrossEncoder(RERANKER_MODEL_NAME)

# Ensure collection + payload index exist (retry-wrapped for DNS blips)
try:
    if not _with_retry(qdrant_client.collection_exists, COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={DENSE_VECTOR_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams()}
        )
except Exception as e:
    raise RuntimeError(f"❌ Could not reach Qdrant at startup: {e}")

try:
    _with_retry(
        qdrant_client.create_payload_index,
        collection_name=COLLECTION_NAME,
        field_name=SOURCE_FILE_FIELD,
        field_schema="keyword",
    )
except Exception as e:
    if "already exists" not in str(e).lower():
        print(f"⚠️ Could not verify payload index on '{SOURCE_FILE_FIELD}': {e}")


# --- Discover which files are actually indexed in Qdrant ---
def list_indexed_source_files():
    seen = set()
    next_page = None
    try:
        while True:
            points, next_page = qdrant_client.scroll(
                collection_name=COLLECTION_NAME,
                limit=200,
                offset=next_page,
                with_payload=[SOURCE_FILE_FIELD],
                with_vectors=False,
            )
            for p in points:
                sf = p.payload.get(SOURCE_FILE_FIELD)
                if sf:
                    seen.add(sf)
            if next_page is None:
                break
    except Exception as e:
        print(f"Could not list indexed files: {e}")
    return sorted(seen)


# --- Retrieval & Synthesis Pipeline ---
def hybrid_retrieve_candidates(query_text: str, top_k: int = 15, source_files=None, max_retries: int = QUERY_MAX_RETRIES):
    """source_files (list[str] | None) scopes retrieval to specific
    documents via a Qdrant payload filter. None / empty list = search
    everything indexed.

    Retries on transient Qdrant read timeouts (ResponseHandlingException /
    UnexpectedResponse) instead of letting one flaky network moment kill a
    whole evaluation run partway through."""
    dense_query_vector = list(dense_model.embed([query_text]))[0].tolist()
    sparse_emb = list(sparse_model.embed([query_text]))[0]
    sparse_query_vector = SparseVector(indices=sparse_emb.indices.tolist(), values=sparse_emb.values.tolist())

    query_filter = None
    if source_files:
        query_filter = Filter(must=[FieldCondition(key=SOURCE_FILE_FIELD, match=MatchAny(any=source_files))])

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            response = qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                prefetch=[
                    Prefetch(query=dense_query_vector, using=DENSE_VECTOR_NAME, limit=top_k, filter=query_filter),
                    Prefetch(query=sparse_query_vector, using=SPARSE_VECTOR_NAME, limit=top_k, filter=query_filter)
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                query_filter=query_filter,
                limit=top_k
            )
            return response.points
        except (ResponseHandlingException, UnexpectedResponse) as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(QUERY_RETRY_BASE_DELAY * attempt)  # 2s, 4s, ...
    raise last_err


def rerank_chunks(query_text: str, candidate_points, top_n: int = 4):
    if not candidate_points:
        return []
    pairs = [[query_text, point.payload.get("chunk_text_content", "")] for point in candidate_points]
    scores = reranker.predict(pairs)
    scored_points = list(zip(scores, candidate_points))
    scored_points.sort(key=lambda x: x[0], reverse=True)
    return [point for score, point in scored_points[:top_n]]


def generate_with_retry(prompt: str):
    last_err = None
    for model_name in GEMINI_MODEL_CHAIN:
        for attempt in range(1, GEMINI_MAX_RETRIES + 1):
            try:
                return gemini_client.models.generate_content(model=model_name, contents=prompt)
            except Exception as e:
                last_err = e
                err_msg = str(e)
                is_retryable = "UNAVAILABLE" in err_msg or "high demand" in err_msg or "503" in err_msg or "429" in err_msg
                if not is_retryable:
                    raise
                time.sleep(GEMINI_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    raise RuntimeError(f"All models in GEMINI_MODEL_CHAIN failed. Last error: {last_err}")


def generate_constrained_answer(query_text: str, top_chunks, chat_history=None):
    if not top_chunks:
        return "Insufficient information in the provided RBI reports.\n\n### 📚 Sources Referenced:\n(none)"

    context_blocks = []
    sources = set()
    for i, point in enumerate(top_chunks, 1):
        payload = point.payload
        source_file = payload.get(SOURCE_FILE_FIELD, "Unknown File")
        h1 = payload.get("heading_h1", "N/A")
        h2 = payload.get("heading_h2", "N/A")
        content = payload.get("chunk_text_content", "").strip()
        header_path = f"{h1} > {h2}" if h2 != "N/A" else h1
        sources.add(f"- **{source_file}** (`{header_path}`)")
        context_blocks.append(f"[CONTEXT BLOCK #{i}]\nSource File: {source_file}\nSection: {header_path}\nContent:\n{content}")

    full_context = "\n\n----------------------------------------\n\n".join(context_blocks)

    formatted_history = ""
    if chat_history:
        for msg in chat_history[-4:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history += f"{role}: {msg['content']}\n"

    prompt = f"""You are a highly analytical macroeconomic and financial summarizer for Reserve Bank of India (RBI) reports.

### RECENT CHAT HISTORY (FOR CONTEXT):
{formatted_history if formatted_history else "No previous conversation history."}

### RETRIEVED CONTEXT FROM RBI REPORTS:
{full_context}

### CURRENT USER QUESTION:
{query_text}

### CONSTRAINED GENERATION RULES:
1. Answer strictly using ONLY the provided RBI context above. Use the chat history to understand references like "it", "this ratio", or "the previous quarter".
2. If the context does not contain the answer, output: "Insufficient information in the provided RBI reports."
3. Output strictly in clear, crisp Markdown bullet points.
4. Limit each bullet point to a maximum of 20 words per bullet.
5. Prioritize quantitative data, percentages, basis points, and specific macroeconomic metrics.
6. Do NOT write long prose paragraphs or essays.
"""
    try:
        response = generate_with_retry(prompt)
    except Exception as e:
        return f"⚠️ Answer generation failed: {e}\n\n### 📚 Sources Referenced:\n" + "\n".join(sorted(sources))

    return response.text + "\n\n### 📚 Sources Referenced:\n" + "\n".join(sorted(sources))