import os
import time
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv

from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from qdrant_client.models import PointStruct, SparseVector

from rag_core import (
    COLLECTION_NAME, DENSE_MODEL_NAME, DENSE_DIM, DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME, SOURCE_FILE_FIELD, RERANKER_MODEL_NAME,
    gemini_client, qdrant_client, dense_model, sparse_model, reranker,
    hybrid_retrieve_candidates, rerank_chunks, generate_with_retry,
    generate_constrained_answer, list_indexed_source_files,
)

load_dotenv()

st.set_page_config(
    page_title="RBI Intelligence RAG Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
EMBED_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 32
UPSERT_MAX_RETRIES = 5
UPSERT_RETRY_BASE_DELAY = 2


# --- Secret Resolver (UI-specific: checks st.secrets, then falls back to .env) ---
def get_secret(key_name: str, default=None):
    try:
        if key_name in st.secrets:
            return st.secrets[key_name]
    except Exception:
        pass
    return os.getenv(key_name, default)


# --- LlamaParse PDF Ingestion Pipeline ---
def parse_pdf_with_llamaparse(file_name: str, pdf_bytes: bytes):
    llama_key = get_secret("LLAMA_CLOUD_API_KEY")
    if not llama_key:
        st.error("❌ LLAMA_CLOUD_API_KEY is missing from your .env file or Streamlit Secrets!")
        return None

    parser = LlamaParse(api_key=llama_key, result_type="markdown", verbose=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

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