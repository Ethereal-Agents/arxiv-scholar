import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Arxiv-Scholar: End-to-End RAG Pipeline\n",
                "\n",
                "This notebook walks through the entire ingestion and retrieval pipeline of the **Arxiv-Scholar** project. We will explore each component in detail, highlighting their advantages and internal workings.\n",
                "\n",
                "The pipeline consists of the following phases:\n",
                "1. **Download & Ingestion:** Fetching scientific papers from arXiv and parsing them.\n",
                "2. **Chunking:** Splitting the parsed documents into layout-aware semantic chunks.\n",
                "3. **Embedding:** Generating both Dense (Semantic) and Sparse (Keyword) vectors using lightweight CPU models.\n",
                "4. **Vector Database Insertion:** Storing chunks and vectors in Qdrant.\n",
                "5. **Hybrid Retrieval:** Querying Qdrant using Reciprocal Rank Fusion (RRF) to combine Dense and Sparse search results."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import sys\n",
                "import logging\n",
                "import json\n",
                "import pprint\n",
                "\n",
                "sys.path.insert(0, os.path.abspath('..'))\n",
                "sys.path.insert(0, os.path.abspath('../src'))\n",
                "\n",
                "logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')\n",
                "\n",
                "# Ensure we have a download directory\n",
                "os.environ[\"DOWNLOAD_DIR\"] = \"notebook_trial\"\n",
                "os.environ[\"STATE_FILE\"] = \"notebook_trial_state.json\""
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Download & Ingestion\n",
                "\n",
                "**Components:** `ArxivUnifiedEngine` and `LocalDirectoryReader`\n",
                "\n",
                "**Concept:** Ingesting 1TB of arXiv PDFs requires careful batching to avoid running out of disk space and memory. The `ArxivUnifiedEngine` downloads PDFs in manageable batches based on state tracking. The `LocalDirectoryReader` then parses these unstructured PDF files into a structured `Document` schema.\n",
                "\n",
                "**Advantages:**\n",
                "- Resume-able ingestion: the state tracking ensures we can pause/resume.\n",
                "- Memory efficient: files are processed locally in small batches and then cleaned up."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from arxiv_scholar.download.arxiv_ingestion import ArxivUnifiedEngine\n",
                "from arxiv_scholar.ingestion.local import LocalDirectoryReader\n",
                "from configs import config\n",
                "\n",
                "config.DOWNLOAD_DIR = \"notebook_trial\"\n",
                "config.STATE_FILE = \"notebook_trial_state.json\"\n",
                "\n",
                "engine = ArxivUnifiedEngine()\n",
                "print(\"Fetching a small batch of PDFs (batch_size=1)...\")\n",
                "paths = engine.get_batch(batch_size=1)\n",
                "\n",
                "print(f\"\\nDownloaded paths: {paths}\")\n",
                "\n",
                "print(\"\\nReading documents from local directory...\")\n",
                "reader = LocalDirectoryReader(directory_path=config.DOWNLOAD_DIR)\n",
                "documents = list(reader.read())\n",
                "\n",
                "print(f\"\\nExtracted {len(documents)} document(s).\")\n",
                "if documents:\n",
                "    sample_doc = documents[0]\n",
                "    print(f\"Document ID: {sample_doc.id}\")\n",
                "    print(f\"Document Metadata: {sample_doc.metadata}\")\n",
                "    print(f\"Document Content (First 200 chars): {sample_doc.content[:200]}...\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Layout-Aware Chunking\n",
                "\n",
                "**Component:** `LayoutAwareChunker`\n",
                "\n",
                "**Concept:** Scientific papers are dense with information. Simply splitting by character count slices paragraphs and tables in half, destroying semantic meaning. Layout-aware chunking attempts to split text at natural boundaries (paragraphs, sections) up to a defined token limit.\n",
                "\n",
                "**Advantages:**\n",
                "- Preserves context boundaries.\n",
                "- Embeddings are more accurate because the chunk contains a complete thought."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from arxiv_scholar.chunking.layout import LayoutAwareChunker\n",
                "\n",
                "chunker = LayoutAwareChunker(max_chunk_size=1500)\n",
                "chunks = []\n",
                "\n",
                "if documents:\n",
                "    print(f\"Chunking document {sample_doc.id}...\")\n",
                "    chunks = list(chunker.chunk(sample_doc))\n",
                "    print(f\"Generated {len(chunks)} chunks.\")\n",
                "    if chunks:\n",
                "        print(f\"\\nSample Chunk [0] ID: {chunks[0].id}\")\n",
                "        print(f\"Sample Chunk [0] Length: {len(chunks[0].content)} chars\")\n",
                "        print(f\"Sample Chunk [0] Preview: {chunks[0].content[:200]}...\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Embedding (Dense + Sparse)\n",
                "\n",
                "**Components:** `FastEmbedEmbedder` and `SparseBM25Embedder`\n",
                "\n",
                "**Concept:** \n",
                "- **Dense Embeddings:** Capture the semantic meaning of text in a dense vector (e.g., 384 dimensions). Useful for \"conceptual\" searches where the exact words don't match but the meaning does.\n",
                "- **Sparse Embeddings (BM25):** Capture exact keyword frequencies. Useful for searching specific names, acronyms, or formulas.\n",
                "\n",
                "We use `fastembed` which uses ONNX Runtime. \n",
                "**Advantages:**\n",
                "- No heavy PyTorch dependency, much smaller footprint.\n",
                "- Highly optimized for CPU inference."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from arxiv_scholar.embedding.fastembed_embedder import FastEmbedEmbedder, SparseBM25Embedder\n",
                "\n",
                "print(\"Loading FastEmbed Dense Model (BAAI/bge-small-en-v1.5)...\")\n",
                "dense_embedder = FastEmbedEmbedder(\n",
                "    model_name=\"BAAI/bge-small-en-v1.5\",\n",
                "    batch_size=2\n",
                ")\n",
                "\n",
                "print(\"Loading FastEmbed Sparse Model (Qdrant/bm25)...\")\n",
                "sparse_embedder = SparseBM25Embedder(batch_size=2)\n",
                "\n",
                "if chunks:\n",
                "    # Take top 5 chunks for speed in this notebook\n",
                "    sample_chunks = chunks[:5]\n",
                "    texts = [c.content for c in sample_chunks]\n",
                "    \n",
                "    print(f\"\\nEmbedding {len(texts)} chunks (Dense)...\")\n",
                "    dense_vectors = dense_embedder.embed(texts)\n",
                "    print(f\"Dense Vector [0] length: {len(dense_vectors[0])} dimensions\")\n",
                "    \n",
                "    print(f\"\\nEmbedding {len(texts)} chunks (Sparse)...\")\n",
                "    sparse_vectors = sparse_embedder.embed(texts)\n",
                "    print(f\"Sparse Vector [0] Indices shape: {len(sparse_vectors[0].indices)}\")\n",
                "    print(f\"Sparse Vector [0] Values shape: {len(sparse_vectors[0].values)}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Vector Database Insertion\n",
                "\n",
                "**Component:** `QdrantVectorStore`\n",
                "\n",
                "**Concept:** We need a robust database to store and index our high-dimensional vectors and sparse vectors. Qdrant is chosen for its native support of multi-vector search (hybrid search) and its high performance.\n",
                "\n",
                "**Advantages:**\n",
                "- Supports `Prefetch` to run sparse and dense searches simultaneously.\n",
                "- Native Reciprocal Rank Fusion (RRF) on the server side, saving network bandwidth."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from arxiv_scholar.storage.qdrant_store import QdrantVectorStore\n",
                "\n",
                "collection_name = \"arxiv_notebook_test\"\n",
                "\n",
                "print(\"Connecting to Qdrant (ensure server is running on localhost:6333)...\")\n",
                "store = QdrantVectorStore(\n",
                "    collection_name=collection_name,\n",
                "    host=\"localhost\",\n",
                "    port=6333\n",
                ")\n",
                "\n",
                "print(\"Ensuring collection exists with correct dimensions...\")\n",
                "store.ensure_collection(dimension=dense_embedder.dimension)\n",
                "\n",
                "if chunks:\n",
                "    print(f\"\\nUpserting {len(sample_chunks)} points into Qdrant...\")\n",
                "    upserted_count = store.upsert(\n",
                "        chunks=sample_chunks,\n",
                "        vectors=dense_vectors,\n",
                "        sparse_vectors=sparse_vectors\n",
                "    )\n",
                "    print(f\"Successfully upserted {upserted_count} points.\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Hybrid Retrieval using RRF\n",
                "\n",
                "**Component:** `HybridRetriever`\n",
                "\n",
                "**Concept:** When a user asks a question, we want the best of both worlds: conceptual matching (Dense) and exact term matching (Sparse). We issue both queries to Qdrant using the `Prefetch` API. Qdrant runs both searches independently, and then fuses the ranked lists together using Reciprocal Rank Fusion (RRF).\n",
                "\n",
                "**Formula (RRF):** $Score = \\frac{1}{k + Rank_{dense}} + \\frac{1}{k + Rank_{sparse}}$"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from arxiv_scholar.retrieval.retrieval import HybridRetriever\n",
                "\n",
                "print(\"Initializing Hybrid Retriever...\")\n",
                "retriever = HybridRetriever(\n",
                "    collection_name=collection_name,\n",
                "    qdrant_host=\"localhost\",\n",
                "    qdrant_port=6333,\n",
                "    dense_model_name=\"BAAI/bge-small-en-v1.5\",  # Match model used in ingestion\n",
                "    sparse_model_name=\"Qdrant/bm25\"\n",
                ")\n",
                "\n",
                "query = \"What are the main contributions of this paper?\"\n",
                "print(f\"\\nExecuting Hybrid Search for: '{query}'\")\n",
                "results = retriever.retrieve(query_text=query, limit=3)\n",
                "\n",
                "print(\"\\n--- Top Results ---\")\n",
                "for i, res in enumerate(results):\n",
                "    print(f\"\\nRank {i+1} | Score: {res['score']:.4f}\")\n",
                "    print(f\"Chunk ID: {res['chunk_id']}\")\n",
                "    print(f\"Content preview: {res['text'][:250]}...\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Cleanup\n",
                "Clean up the downloaded batch to free disk space."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Cleaning up batch...\")\n",
                "engine.cleanup_batch(paths)\n",
                "print(\"Done!\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("/Users/tri/Projects/arxiv-scholar/notebooks/arxiv_scholar_e2e_detailed.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
print("Notebook created at notebooks/arxiv_scholar_e2e_detailed.ipynb")
