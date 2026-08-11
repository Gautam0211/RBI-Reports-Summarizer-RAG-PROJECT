# 🏦 RBI Macroeconomic & Policy RAG Assistant Chatbot

An enterprise-grade Hybrid Retrieval-Augmented Generation (RAG) system designed to parse, index, search, and synthesize insights from Reserve Bank of India (RBI) Monetary Policy and Financial Stability Reports. 

Built with a dual-vector hybrid search engine (Dense + Sparse BM25) in **Qdrant**, cross-encoder reranking, and **Google Gemini** for constrained synthesis, alongside an automated **RAGAS** benchmarking suite.

---

## 🌟 Key Features

* **📄 Complex PDF Parsing:** Uses **LlamaParse** to accurately extract structured text and complex tables/markdown matrices from RBI policy documents.
* **🧩 Structural & Recursive Chunking:** Combines **Markdown Header Text Chunker** with **Recursive Character Text Splitter** to preserve section hierarchies and semantic context across large reports.
* **🔎 Dual Hybrid Retrieval:** Combines **Dense Semantic Search** (`BAAI/bge-base-en-v1.5`) with **Sparse Keyword Search** (`BM25`) inside **Qdrant Vector DB** using Reciprocal Rank Fusion (RRF).
* **🎯 Cross-Encoder Reranking:** Filters and re-ranks initial candidates using `BAAI/bge-reranker-base` to feed only high-precision chunks to the LLM.
* **🛡️ Constrained Synthesis:** Employs **Google Gemini** with strict system guardrails to eliminate hallucinations and generate quantitative, bulleted summaries with source citations.
* **💻 Interactive Dashboard:** Streamlit UI featuring document context switching, multi-turn chat memory, and status tracking.

---

### 💡 Model Summary Matrix

| Stage | Model / Tool | Architecture Type | Main Purpose |
| :--- | :--- | :--- | :--- |
| **Parsing** | **LlamaParse** | Vision-Layout Parser | Extract tables & Markdown matrices from PDFs |
| **Dense Search** | **`bge-base-en-v1.5`** | Dense BERT Encoder (768d) | Capture conceptual & semantic meaning |
| **Sparse Search** | **BM25** | Lexical Term Model | Match exact terms, acronyms, and figures |
| **Reranking** | **`bge-reranker-base`** | Cross-Encoder | Score & rank query-context pairs for top precision |
| **Generation** | **Gemini 3.6 Flash** | Generative LLM | Synthesize guarded, cited answers |
| **Evaluation** | **Gemini + RAGAS** | LLM-as-a-Judge | Automated benchmark grading (0.0 to 1.0) |


## 🛠️ System Architecture

```text
┌─────────────────┐      ┌────────────────────┐      ┌───────────────────────────────────┐
│  RBI PDF Report │ ───> │     LlamaParse     │ ───> │ Markdown Header Text Chunker +    │
└─────────────────┘      └────────────────────┘      │ Recursive Character Text Splitter │
                                                             └─────────────────┬─────────────────┘
                                                                               │
                                                                               ▼
┌─────────────────┐      ┌────────────────────┐      ┌───────────────────────────────────┐
│ Gemini Synthesis│ <─── │ Cross-Encoder      │ <─── │ Qdrant Hybrid Search              │
│  (Streamlit UI) │      │ (bge-reranker-base)│      │ (Dense BGE-1.5 + Sparse BM25)     │
└─────────────────┘      └────────────────────┘      └───────────────────────────────────┘

## Repository Structure  

├── app.py                   # Main Streamlit Dashboard UI
├── rag_core.py              # Core Engine: Parsing, Indexing, Hybrid Search & Gemini Synthesis
├── evaluate.py              # RAGAS Automated Evaluation Benchmark Script
├── requirements.txt         # Project Dependencies
├── .env.example             # Environment Variables Template
├── DATA/                    # Local Raw RBI PDF Documents (Ignored by Git)
└── README.md                # Project Documentation
##  Running the Application
streamlit run app.py

