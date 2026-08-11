import os
import time
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv

# NOTE: nest_asyncio.apply() used to be called here at import time. That
# patches the GLOBAL asyncio event loop machinery before Streamlit's own
# Starlette/uvicorn/anyio server loop starts, which corrupts anyio's ability
# to create/detect its own loop for serving static assets — causing
# `anyio.NoEventLoopError` on static file routes (this is also what broke
# the FileUploader.js dynamic import earlier). Import + apply it lazily,
# only inside the function that actually needs it, and only in that thread.

from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector,
    Prefetch, FusionQuery, Fusion, Filter, FieldCondition, MatchAny
)

from fastembed import TextEmbedding, SparseTextEmbedding
from sentence_transformers import CrossEncoder

load_dotenv()

st.set_page_config(
    page_title="RBI Intelligence RAG Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Constants ---
COLLECTION_NAME = "rbi_hybrid_summarizer"
DENSE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DENSE_DIM = 768
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "text-sparse"
SOURCE_FILE_FIELD = "source_file"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-base"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
EMBED_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 32
UPSERT_MAX_RETRIES = 5
UPSERT_RETRY_BASE_DELAY = 2

# gemini-2.5-flash / gemini-2.5-pro now return 404 "no longer available to
# new users" for many accounts, even though Google's lifecycle page still
# lists a later shutdown date. Don't rely on them as fallbacks.
GEMINI_MODEL_CHAIN = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-pro-preview"]
GEMINI_MAX_RETRIES = 4
GEMINI_RETRY_BASE_DELAY = 3

STARTUP_NET_MAX_RETRIES = 3
STARTUP_NET_RETRY_BASE_DELAY = 2

# --- Secret Resolver ---
def get_secret(key_name: str, default=None):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name, default)

# --- Small helper: retry transient network calls (e.g. DNS blips on Windows,
# "[Errno 11001] getaddrinfo failed") instead of crashing the whole app on
# the first hiccup. Used only for one-off startup calls, not hot-path queries.
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

# --- Cached Resource Loaders ---
@st.cache_resource(show_spinner=False)
def load_gemini_client():
    api_key = get_secret("GOOGLE_API_KEY")
    if not api_key:
        st.error("❌ GOOGLE_API_KEY is missing! Please set it in your .env file or Streamlit Secrets.")
        st.stop()
    return genai.Client(api_key=api_key)

@st.cache_resource(show_spinner=False)
def load_qdrant_client():
    qdrant_url = get_secret("QDRANT_URL")
    qdrant_api_key = get_secret("QDRANT_API_KEY")

    if qdrant_url and qdrant_api_key:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60)
    else:
        client = QdrantClient(path="./qdrant_storage_db")

    return client

@st.cache_resource(show_spinner="Loading local dense embedding model...")
def load_dense_model():
    return TextEmbedding(model_name=DENSE_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_sparse_model():
    return SparseTextEmbedding("Qdrant/bm25")

@st.cache_resource(show_spinner="Loading Cross-Encoder Re-ranker Model...")
def load_reranker_model():
    return CrossEncoder(RERANKER_MODEL_NAME)

# Initialize Resources
gemini_client = load_gemini_client()
qdrant_client = load_qdrant_client()
dense_model = load_dense_model()
sparse_model = load_sparse_model()
reranker = load_reranker_model()

# Ensure Qdrant collection exists on startup (retry-wrapped: survives a
# transient DNS/connection blip instead of crashing the Streamlit run)
try:
    collection_already_exists = _with_retry(qdrant_client.collection_exists, COLLECTION_NAME)
    if not collection_already_exists:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams()
            }
        )
except Exception as e:
    st.error(f"❌ Could not reach Qdrant at startup: {e}")
    st.stop()

# Ensure a payload index exists on "source_file" — Qdrant (esp. Cloud) requires
# an explicit index before a field can be used in a Filter/FieldCondition,
# otherwise query_points raises 400: "Index required but not found for
# source_file of one of the following types: [keyword]". Safe to call every
# startup; if it already exists we just swallow that specific message.
try:
    _with_retry(
        qdrant_client.create_payload_index,
        collection_name=COLLECTION_NAME,
        field_name=SOURCE_FILE_FIELD,
        field_schema="keyword",
    )
except Exception as e:
    if "already exists" not in str(e).lower():
        st.sidebar.warning(f"⚠️ Could not verify payload index on '{SOURCE_FILE_FIELD}': {e}")


# --- Discover which files are actually indexed in Qdrant ---
def list_indexed_source_files():
    """Scrolls the collection's payloads to build the distinct list of
    source_file values, so the sidebar selector always reflects reality
    instead of relying on session state alone (works across reruns/restarts)."""
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
        st.sidebar.warning(f"Could not list indexed files: {e}")
    return sorted(seen)


# --- LlamaParse PDF Ingestion Pipeline ---
def parse_pdf_with_llamaparse(file_name: str, pdf_bytes: bytes):
    """Parses uploaded PDF bytes into structured Markdown via LlamaParse."""
    llama_key = get_secret("LLAMA_CLOUD_API_KEY")
    if not llama_key:
        st.error("❌ LLAMA_CLOUD_API_KEY is missing from your .env file or Streamlit Secrets!")
        return None

    parser = LlamaParse(
        api_key=llama_key,
        result_type="markdown",
        verbose=True
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        # LlamaParse network/parsing failures no longer crash the app —
        # arbitrary user uploads are far more likely to hit malformed/scanned
        # PDFs or transient API errors than one pre-tested file.
        documents = parser.load_data(tmp_path)

        if not documents:
            st.error(f"❌ LlamaParse returned no pages for '{file_name}'. The PDF may be corrupted, "
                      f"password-protected, or entirely scanned images with no extractable layout.")
            return None

        markdown_text = "\n\n".join([doc.text for doc in documents if doc.text and doc.text.strip()])
    except Exception as e:
        st.error(f"❌ LlamaParse failed to parse '{file_name}': {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    return markdown_text

def chunk_markdown_text(markdown_text):
    headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_docs = markdown_splitter.split_text(markdown_text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n|", "\n", ". ", " ", ""]
    )
    return recursive_splitter.split_documents(header_docs)

def upsert_with_retry(points):
    """Reuses the same retry/backoff logic as the indexing script, so
    uploads from the Streamlit UI survive transient connection resets."""
    last_err = None
    for attempt in range(1, UPSERT_MAX_RETRIES + 1):
        try:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, wait=True, points=points)
            return True
        except Exception as e:
            last_err = e
            time.sleep(UPSERT_RETRY_BASE_DELAY * (2 ** (attempt - 1)))
    st.warning(f"⚠️ A batch of {len(points)} points failed to upload after {UPSERT_MAX_RETRIES} retries: {last_err}")
    return False

def process_and_index_pdf(file_name: str, pdf_bytes: bytes, status_container):
    """Full Pipeline: LlamaParse -> Splitters -> FastEmbed -> Qdrant Batch Upsert."""
    status_container.write("🦙 1/4 Extracting structured Markdown via LlamaParse...")
    markdown_content = parse_pdf_with_llamaparse(file_name, pdf_bytes)

    if not markdown_content or not markdown_content.strip():
        st.error("❌ LlamaParse produced empty content for this PDF.")
        return 0

    status_container.write("✂️ 2/4 Chunking Markdown via Structural & Recursive Splitters...")
    try:
        chunks = chunk_markdown_text(markdown_content)
    except Exception as e:
        st.error(f"❌ Chunking failed for '{file_name}': {e}")
        return 0

    # Filter empty/whitespace-only chunks (can crash BM25 embedding on "")
    filtered = [(c.page_content, c.metadata) for c in chunks if c.page_content and c.page_content.strip()]
    if not filtered:
        st.error(f"❌ '{file_name}' produced no usable chunks after filtering.")
        return 0

    texts = [t for t, _ in filtered]
    metadatas = [m for _, m in filtered]

    status_container.write(f"⚡ 3/4 Generating Local Dense ({DENSE_MODEL_NAME}) & BM25 Sparse Vectors for {len(texts)} chunks...")
    try:
        sparse_embeddings = list(sparse_model.embed(texts))
        dense_embeddings = list(dense_model.embed(texts, batch_size=EMBED_BATCH_SIZE))
        dense_vectors = [vec.tolist() for vec in dense_embeddings]
    except Exception as e:
        st.error(f"❌ Embedding generation failed for '{file_name}': {e}")
        return 0

    if len(dense_vectors) != len(texts) or len(sparse_embeddings) != len(texts):
        st.error(f"❌ Vector count mismatch for '{file_name}'; aborting to avoid misaligned upsert.")
        return 0

    status_container.write(f"☁️ 4/4 Upserting points to Qdrant Cloud in batches of {UPSERT_BATCH_SIZE}...")
    points_to_upsert = []
    for i, chunk_text in enumerate(texts):
        metadata_payload = {
            SOURCE_FILE_FIELD: file_name,
            "heading_h1": metadatas[i].get("Header_1", "N/A"),
            "heading_h2": metadatas[i].get("Header_2", "N/A"),
            "heading_h3": metadatas[i].get("Header_3", "N/A"),
            "chunk_text_content": chunk_text
        }

        sparse_emb = sparse_embeddings[i]
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector={
                DENSE_VECTOR_NAME: dense_vectors[i],
                SPARSE_VECTOR_NAME: SparseVector(
                    indices=sparse_emb.indices.tolist(),
                    values=sparse_emb.values.tolist()
                ),
            },
            payload=metadata_payload
        )
        points_to_upsert.append(point)

    indexed = 0
    for i in range(0, len(points_to_upsert), UPSERT_BATCH_SIZE):
        batch = points_to_upsert[i:i + UPSERT_BATCH_SIZE]
        if upsert_with_retry(batch):
            indexed += len(batch)

    return indexed

# --- Retrieval & Synthesis Pipeline ---
def hybrid_retrieve_candidates(query_text: str, top_k: int = 15, source_files=None):
    """source_files (list[str] | None) scopes retrieval to specific
    documents via a Qdrant payload filter. None / empty list = search
    everything indexed (matches this baseline's original behavior)."""
    dense_query_vector = list(dense_model.embed([query_text]))[0].tolist()
    sparse_emb = list(sparse_model.embed([query_text]))[0]

    sparse_query_vector = SparseVector(
        indices=sparse_emb.indices.tolist(),
        values=sparse_emb.values.tolist()
    )

    query_filter = None
    if source_files:
        query_filter = Filter(
            must=[FieldCondition(key=SOURCE_FILE_FIELD, match=MatchAny(any=source_files))]
        )

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

def rerank_chunks(query_text: str, candidate_points, top_n: int = 4):
    if not candidate_points:
        return []

    pairs = [[query_text, point.payload.get("chunk_text_content", "")] for point in candidate_points]
    scores = reranker.predict(pairs)

    scored_points = list(zip(scores, candidate_points))
    scored_points.sort(key=lambda x: x[0], reverse=True)

    return [point for score, point in scored_points[:top_n]]

def generate_with_retry(prompt: str):
    """Retry transient Gemini errors and fall back through GEMINI_MODEL_CHAIN.
    A genuine (non-retryable) error stops the whole attempt instead of
    silently falling through every model in the chain and masking the real
    cause behind a generic 'all models failed' message."""
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

        block = f"[CONTEXT BLOCK #{i}]\nSource File: {source_file}\nSection: {header_path}\nContent:\n{content}"
        context_blocks.append(block)

    full_context = "\n\n----------------------------------------\n\n".join(context_blocks)

    formatted_history = ""
    if chat_history:
        recent_turns = chat_history[-4:]
        for msg in recent_turns:
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
        return (f"⚠️ Answer generation failed: {e}\n\n"
                f"### 📚 Sources Referenced:\n" + "\n".join(sorted(list(sources))))

    formatted_sources = "\n\n### 📚 Sources Referenced:\n" + "\n".join(sorted(list(sources)))
    return response.text + formatted_sources

# --- Streamlit UI Layout ---
st.title("🏦 RBI Macroeconomic & Policy RAG Assistant")
st.markdown("Search, index, and synthesize insights across custom RBI PDFs or pre-indexed reports.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("📂 Upload Custom RBI Report")
    uploaded_file = st.file_uploader("Upload an RBI PDF to parse via LlamaParse", type=["pdf"])

    if uploaded_file is not None:
        if st.button("🚀 Parse & Index PDF", use_container_width=True):
            with st.status("🦙 Ingesting PDF with LlamaParse...", expanded=True) as status:
                num_points = process_and_index_pdf(
                    file_name=uploaded_file.name,
                    pdf_bytes=uploaded_file.getvalue(),
                    status_container=st
                )
                if num_points:
                    status.update(label=f"✅ Successfully indexed {num_points} chunks from '{uploaded_file.name}'!", state="complete", expanded=False)
                    st.success(f"'{uploaded_file.name}' is active in Qdrant Cloud!")
                    # Auto-scope future queries to the file just uploaded, and
                    # force the sidebar file list to refresh.
                    st.session_state["scope_selection"] = [uploaded_file.name]
                    st.session_state.pop("_indexed_files_cache", None)
                    st.rerun()
                else:
                    status.update(label=f"❌ Indexing failed for '{uploaded_file.name}'.", state="error", expanded=True)

    st.divider()

    st.header("📑 Search Scope")
    if "_indexed_files_cache" not in st.session_state:
        st.session_state["_indexed_files_cache"] = list_indexed_source_files()
    indexed_files = st.session_state["_indexed_files_cache"]

    if st.button("🔄 Refresh file list", use_container_width=True):
        st.session_state["_indexed_files_cache"] = list_indexed_source_files()
        indexed_files = st.session_state["_indexed_files_cache"]

    if indexed_files:
        default_scope = st.session_state.get("scope_selection", indexed_files)
        default_scope = [f for f in default_scope if f in indexed_files] or indexed_files

        scope_selection = st.multiselect(
            "Answer using only these file(s):",
            options=indexed_files,
            default=default_scope,
            help="Retrieval is restricted to the selected file(s). Uncheck all "
                 "to search across every indexed report."
        )
        st.session_state["scope_selection"] = scope_selection
    else:
        st.info("No documents indexed yet. Upload a PDF above.")
        st.session_state["scope_selection"] = []

    st.divider()
    st.header("⚙️ Search Settings")
    top_k_candidates = st.slider("Top Candidates (Hybrid Search)", min_value=5, max_value=30, value=15, step=5)
    top_n_rerank = st.slider("Final Context Chunks (Re-ranked)", min_value=2, max_value=8, value=4, step=1)

    st.divider()
    st.markdown("### 🏗️ Architecture Status")
    st.success("✅ Parser: LlamaParse (Markdown)")
    st.success("✅ Hybrid Search: BGE-base Dense + BM25 Sparse")
    st.success(f"✅ Re-ranker: `{RERANKER_MODEL_NAME}`")
    st.success("✅ Generator: Gemini 3.6 Flash")

    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --- Chat Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & Execution Loop ---
if user_query := st.chat_input("Ask a question about your uploaded or pre-indexed RBI reports..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.status("🔍 Searching and Analyzing RBI Reports...", expanded=True) as status:

            scope = st.session_state.get("scope_selection") or None
            scope_note = f" (scoped to {', '.join(scope)})" if scope else " (all indexed files)"
            st.write(f"Fetching candidates via Hybrid BM25 + Semantic Search{scope_note}...")
            candidates = hybrid_retrieve_candidates(user_query, top_k=top_k_candidates, source_files=scope)
            st.write(f"✓ Retrieved {len(candidates)} candidates from Qdrant.")

            st.write("Re-ranking context using Cross-Encoder...")
            top_chunks = rerank_chunks(user_query, candidates, top_n=top_n_rerank)
            st.write(f"✓ Selected top {len(top_chunks)} context blocks.")

            st.write("Synthesizing answer with Gemini...")
            final_answer = generate_constrained_answer(
                query_text=user_query,
                top_chunks=top_chunks,
                chat_history=st.session_state.messages[:-1]
            )

            status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})