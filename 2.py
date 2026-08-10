import os
import glob
import uuid
from dotenv import load_dotenv

# LangChain Structural Text Splitters
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# Production Gemini and Hybrid Qdrant Clients
from google import genai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, SparseVectorParams, SparseVector

# FastEmbed for sparse (BM25) vectors — used directly instead of the
# QdrantClient convenience wrappers, which don't support mixing an
# externally-computed dense vector with an auto-generated sparse one.
from fastembed import SparseTextEmbedding

load_dotenv()

# --- Configurations ---
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300
COLLECTION_NAME = "rbi_hybrid_summarizer"
DENSE_DIM = 768  # Native dimension for text-embedding-004
EMBED_BATCH_SIZE = 100  # Gemini embed_content has a per-call item limit

# 1. Initialize Gemini Client
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("❌ GOOGLE_API_KEY is missing from your .env file!")

gemini_client = genai.Client(api_key=google_api_key)

# 2. Initialize Qdrant Client (Cloud with Local Fallback)
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

if qdrant_url and qdrant_api_key:
    print(f"🌐 Connecting to Qdrant Cloud Cluster: {qdrant_url[:30]}...")
    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
else:
    print("🏠 Qdrant Cloud credentials not complete in .env. Falling back to local disk mode ('./qdrant_storage_db')...")
    qdrant_client = QdrantClient(path="./qdrant_storage_db")

# 3. Initialize the sparse (BM25) embedding model once, up front
sparse_model = SparseTextEmbedding("Qdrant/bm25")

# 4. Setup Hybrid Search Engine (Dense Vector Params + Sparse Keyword Params)
if not qdrant_client.collection_exists(COLLECTION_NAME):
    print(f"Creating fresh Hybrid Qdrant Collection: '{COLLECTION_NAME}'...")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        # Configuration for AI Semantic Meaning (Dense) — must be a dict
        # keyed by vector name ("") since points also carry a named
        # "text-sparse" vector; a bare VectorParams here can cause
        # ambiguity issues with some client versions.
        vectors_config={
            "": VectorParams(size=DENSE_DIM, distance=Distance.COSINE)
        },
        # Configuration for Exact Keyword Matching (Sparse)
        sparse_vectors_config={
            "text-sparse": SparseVectorParams()
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


def embed_dense_batch(texts):
    """Embed texts with Gemini in batches, returning a flat list of vectors."""
    all_vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        embed_response = gemini_client.models.embed_content(
            model="text-embedding-004",
            contents=batch
        )
        all_vectors.extend(emb.values for emb in embed_response.embeddings)
    return all_vectors


def run_hybrid_pipeline():
    md_files = glob.glob("parsed_data/*.md")
    if not md_files:
        print("❌ Error: No .md files found inside 'parsed_data/'!")
        return

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

        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        # Generate all keyword sparse vectors in one single batch call
        print("  ↳ Computing keyword sparse matrix mappings...")
        sparse_embeddings = list(sparse_model.embed(texts))

        # Compute all dense embeddings via Gemini (batched to respect API limits)
        print(f"  ↳ Computing dense semantic embeddings for {len(texts)} chunks via Gemini...")
        try:
            dense_vectors = embed_dense_batch(texts)
        except Exception as e:
            print(f"❌ Failed to generate Gemini embeddings for {file_name}: {e}")
            continue

        if len(dense_vectors) != len(texts) or len(sparse_embeddings) != len(texts):
            print(f"❌ Vector count mismatch for {file_name}, skipping to avoid misaligned upsert.")
            continue

        # Compile points matrix
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
                    "": dense_vectors[i],  # Primary Dense Slot
                    "text-sparse": SparseVector(
                        indices=sparse_emb.indices.tolist(),
                        values=sparse_emb.values.tolist()
                    ),
                },
                payload=metadata_payload
            )
            points_to_upsert.append(point)

        if points_to_upsert:
            qdrant_client.upsert(
                collection_name=COLLECTION_NAME,
                wait=True,
                points=points_to_upsert
            )
            print(f"✅ Success! Hybrid indexed {len(points_to_upsert)} blocks for '{file_name}'.")

    print("\n🎉 Core Pipeline Completed! Your documents are dual-indexed and active inside Qdrant.")


if __name__ == "__main__":
    run_hybrid_pipeline()