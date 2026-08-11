import os
import glob
import time
import uuid
from dotenv import load_dotenv

# LangChain Structural Text Splitters
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Hybrid Qdrant Client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector

# FastEmbed for BOTH dense (local, open-source) and sparse (BM25) vectors.
from fastembed import TextEmbedding, SparseTextEmbedding

load_dotenv()

# --- Configurations ---
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
COLLECTION_NAME = "rbi_hybrid_summarizer"

# BAAI/bge-base-en-v1.5 natively outputs 768-dim vectors
DENSE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
DENSE_DIM = 768
DENSE_VECTOR_NAME = "dense"     # FIX: named vector instead of ""
SPARSE_VECTOR_NAME = "text-sparse"
EMBED_BATCH_SIZE = 32
UPSERT_BATCH_SIZE = 32   # FIX: smaller batches are far more reliable over REST/Windows
UPSERT_MAX_RETRIES = 5
UPSERT_RETRY_BASE_DELAY = 2  # seconds, doubles each retry

# 1. Initialize Qdrant Client (Cloud with Local Fallback)
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if qdrant_url and qdrant_api_key:
    print(f"🌐 Connecting to Qdrant Cloud Cluster: {qdrant_url[:30]}...")
    # FIX: explicit timeout so a stalled request fails fast/cleanly instead of
    # hanging until the OS forcibly resets the socket (WinError 10054).
    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key,
        timeout=60,
    )
else:
    print("🏠 Qdrant Cloud credentials not complete in .env. Falling back to local disk mode ('./qdrant_storage_db')...")
    qdrant_client = QdrantClient(path="./qdrant_storage_db")

# 2. Initialize local embedding models once, up front
print(f"📦 Loading local dense embedding model: '{DENSE_MODEL_NAME}'...")
dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)

print("📦 Loading local sparse (BM25) embedding model: 'Qdrant/bm25'...")
sparse_model = SparseTextEmbedding("Qdrant/bm25")

# 3. Setup Hybrid Search Engine (Recreate collection if vector space changes)
if qdrant_client.collection_exists(COLLECTION_NAME):
    print(f"🗑️ Re-creating collection '{COLLECTION_NAME}' to clear old embeddings...")
    qdrant_client.delete_collection(COLLECTION_NAME)

print(f"Creating fresh Hybrid Qdrant Collection: '{COLLECTION_NAME}'...")
qdrant_client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config={
        DENSE_VECTOR_NAME: VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        SPARSE_VECTOR_NAME: SparseVectorParams()
    }
)


def chunk_markdown_file(markdown_text, chunk_size, chunk_overlap):
    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    header_docs = markdown_splitter.split_text(markdown_text)

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n|", "\n", ". ", " ", ""]
    )
    return recursive_splitter.split_documents(header_docs)


def upsert_with_retry(client, collection_name, points):
    """Upsert a batch with exponential-backoff retries to survive transient
    connection resets (e.g. WinError 10054) against Qdrant Cloud."""
    last_err = None
    for attempt in range(1, UPSERT_MAX_RETRIES + 1):
        try:
            client.upsert(collection_name=collection_name, wait=True, points=points)
            return True
        except Exception as e:
            last_err = e
            delay = UPSERT_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"    ⚠️ Upsert attempt {attempt}/{UPSERT_MAX_RETRIES} failed ({e.__class__.__name__}: {e}). "
                  f"Retrying in {delay}s...")
            time.sleep(delay)
    print(f"    ❌ Giving up on this batch after {UPSERT_MAX_RETRIES} attempts: {last_err}")
    return False


def embed_dense_batch(texts):
    """Embed texts locally via fastembed — no network calls, no rate limits."""
    embeddings = list(dense_model.embed(texts, batch_size=EMBED_BATCH_SIZE))
    return [vec.tolist() for vec in embeddings]


def run_hybrid_pipeline():
    md_files = [
        f for f in glob.glob("parsed_data/*.md")
        if "monetary" in os.path.basename(f).lower()
    ]

    if not md_files:
        print("❌ Error: No Monetary Policy .md file found inside 'parsed_data/'!")
        return

    total_indexed = 0

    for file_path in md_files:
        file_name = os.path.basename(file_path)
        print(f"\n🚀 Indexing structures via Hybrid Search for: '{file_name}'...")

        with open(file_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        if not markdown_content.strip():
            print(f"⚠️ Warning: '{file_name}' is empty. Skipping.")
            continue

        chunks = chunk_markdown_file(markdown_content, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            print(f"⚠️ Warning: '{file_name}' produced no chunks. Skipping.")
            continue

        # FIX: drop empty/whitespace-only chunks before embedding
        filtered = [
            (c.page_content, c.metadata)
            for c in chunks
            if c.page_content and c.page_content.strip()
        ]
        if not filtered:
            print(f"⚠️ Warning: '{file_name}' had only empty chunks after filtering. Skipping.")
            continue

        texts = [t for t, _ in filtered]
        metadatas = [m for _, m in filtered]

        # FIX: wrap sparse embedding in try/except like dense
        print("  ↳ Computing keyword sparse matrix mappings (local BM25)...")
        try:
            sparse_embeddings = list(sparse_model.embed(texts))
        except Exception as e:
            print(f"❌ Failed to generate sparse embeddings for {file_name}: {e}")
            continue

        print(f"  ↳ Computing dense semantic embeddings for {len(texts)} chunks (local {DENSE_MODEL_NAME})...")
        try:
            dense_vectors = embed_dense_batch(texts)
        except Exception as e:
            print(f"❌ Failed to generate dense embeddings for {file_name}: {e}")
            continue

        if len(dense_vectors) != len(texts) or len(sparse_embeddings) != len(texts):
            print(f"❌ Vector count mismatch for {file_name}, skipping to avoid misaligned upsert.")
            continue

        points_to_upsert = []
        for i, chunk_text in enumerate(texts):
            metadata_payload = {
                "source_file": file_name,
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

        if points_to_upsert:
            print(f"  ↳ Uploading {len(points_to_upsert)} points to Qdrant in batches of {UPSERT_BATCH_SIZE}...")
            file_indexed = 0
            file_failed_batches = 0
            for i in range(0, len(points_to_upsert), UPSERT_BATCH_SIZE):
                batch_points = points_to_upsert[i:i + UPSERT_BATCH_SIZE]
                batch_num = i // UPSERT_BATCH_SIZE + 1
                ok = upsert_with_retry(qdrant_client, COLLECTION_NAME, batch_points)
                if ok:
                    file_indexed += len(batch_points)
                else:
                    file_failed_batches += 1
                    print(f"    ⏭️ Skipped batch {batch_num} ({len(batch_points)} points) after retries exhausted.")

            total_indexed += file_indexed
            if file_failed_batches:
                print(f"⚠️ Partial success: indexed {file_indexed}/{len(points_to_upsert)} points for "
                      f"'{file_name}' ({file_failed_batches} batch(es) failed).")
            else:
                print(f"✅ Success! Hybrid indexed {file_indexed} blocks for '{file_name}'.")

    if total_indexed == 0:
        print("\n⚠️ Pipeline finished but 0 points were indexed — check the errors above.")
    else:
        print(f"\n🎉 Core Pipeline Completed! {total_indexed} total points indexed across Monetary Policy report.")


if __name__ == "__main__":
    run_hybrid_pipeline()