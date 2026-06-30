import json

code = """
import logging
from arxiv_scholar.schema import Chunk
from arxiv_scholar.embedding.fastembed_embedder import FastEmbedEmbedder, SparseBM25Embedder
from arxiv_scholar.storage.qdrant_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

print("Initializing embedders...")
dense_embedder = FastEmbedEmbedder(
    model_name="BAAI/bge-small-en-v1.5",
    batch_size=2
)
sparse_embedder = SparseBM25Embedder(batch_size=2)

print("Connecting to Qdrant...")
store = QdrantVectorStore(
    collection_name="test_hybrid_search_nb",
    host="localhost",
    port=6333
)

print("Ensuring collection exists...")
store.ensure_collection(dimension=dense_embedder.dimension)

print("Creating mock chunks...")
texts = [
    "Machine learning is a field of study in artificial intelligence.",
    "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics.",
    "Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers."
]

import hashlib
chunks = []
for i, text in enumerate(texts):
    chunk_id = hashlib.sha256(text.encode()).hexdigest()
    chunks.append(
        Chunk(
            id=chunk_id,
            document_id="doc_1",
            content=text,
            metadata={"index": i}
        )
    )

print("Embedding dense vectors...")
dense_vectors = dense_embedder.embed(texts)

print("Embedding sparse vectors...")
sparse_vectors = sparse_embedder.embed(texts)

print("Upserting to Qdrant...")
store.upsert(chunks, dense_vectors, sparse_vectors)
print("Upsert successful!")

# Test retrieval
query_text = "What is deep learning?"
print(f"\\nSearching for: '{query_text}'")
query_dense = dense_embedder.embed([query_text])[0]
query_sparse = sparse_embedder.embed([query_text])[0]

print("Performing hybrid search...")
results = store.hybrid_search(
    query_vector=query_dense,
    sparse_vector=query_sparse,
    top_k=2
)

print("\\nResults:")
for i, res in enumerate(results):
    print(f"Rank {i+1}: Score={res['score']} | Content: {res['content']}")
"""

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# E2E Flow Test: Qdrant Hybrid Search (Dense + Sparse + RRF)\n",
                "This notebook tests the E2E flow for insertion and retrieval from Qdrant using dense and sparse vectors, along with Reciprocal Rank Fusion (RRF)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": code.strip().splitlines(True)
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

with open("/Users/tri/Projects/arxiv-scholar/notebooks/e2e_hybrid_search.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
