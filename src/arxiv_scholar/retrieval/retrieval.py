"""Hybrid Search Retriever.

Implements a pure Python hybrid search retrieval function targeting a Qdrant vector database.
Executes dense and sparse embeddings via fastembed and triggers server-side Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client import models
from fastembed import TextEmbedding, SparseTextEmbedding

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Retriever for Qdrant using FastEmbed and server-side RRF fusion."""

    def __init__(
        self,
        collection_name: str,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        location: str = None,
        dense_model_name: str = "BAAI/bge-m3",
        sparse_model_name: str = "Qdrant/bm25",
    ) -> None:
        """Initializes the retriever and its global state (models and db client)."""
        self.collection_name = collection_name
        
        # 1. Initialize the Qdrant client
        if location:
            self.client = QdrantClient(location=location)
        else:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            
        logger.info(f"Initialized QdrantClient for collection '{collection_name}'")

        # 2. Initialize two fastembed models (Dense and Sparse).
        # These are loaded globally for the instance and cached.
        logger.info(f"Loading dense model: {dense_model_name}")
        self.dense_model = TextEmbedding(model_name=dense_model_name)
        
        logger.info(f"Loading sparse model: {sparse_model_name}")
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)

    def retrieve(self, query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Executes a hybrid search query with server-side RRF.
        
        Args:
            query_text: The raw natural language query.
            limit: The maximum number of top-K results to return (default 20).
            
        Returns:
            A list of dictionaries containing chunk_id, text, score, and metadata.
        """
        # 3. Generate the Dense vector for the query_text.
        # fastembed returns generators, so we consume it into a list and take the first item
        dense_vector = list(self.dense_model.embed([query_text]))[0].tolist()

        # 4. Generate the Sparse vector for the query_text.
        sparse_result = list(self.sparse_model.embed([query_text]))[0]
        # fastembed SparseEmbedding objects have .indices and .values properties
        sparse_vector = models.SparseVector(
            indices=sparse_result.indices,
            values=sparse_result.values,
        )

        # 5. The Prefetch Construction
        # Construct prefetch for the Dense search path
        # Assuming the dense vector is the unnamed default vector ("") in Qdrant
        prefetch_dense = models.Prefetch(
            query=dense_vector,
            using="",
            limit=limit,
        )

        # Construct prefetch for the Sparse search path
        # Assuming the sparse vector is named "bm25" or "sparse" in Qdrant (using "bm25" based on earlier ingestion)
        prefetch_sparse = models.Prefetch(
            query=sparse_vector,
            using="bm25",
            limit=limit,
        )

        # 6. Database Execution
        # Trigger server-side Reciprocal Rank Fusion by passing prefetches
        # and setting the query to models.FusionQuery
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[prefetch_dense, prefetch_sparse],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )

        # 7. Output Formatting
        # Extract the payload and the fused score from the returned Qdrant points
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append({
                "chunk_id": payload.get("chunk_id", str(point.id)),
                "text": payload.get("content", ""),
                "score": point.score,
                "metadata": payload.get("metadata", {}),
            })

        return results
