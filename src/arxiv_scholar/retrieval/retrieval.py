"""Hybrid Search Retriever.

Implements a pure Python hybrid search retrieval function targeting a Qdrant vector database.
Executes dense and sparse embeddings via fastembed and triggers server-side Reciprocal Rank Fusion (RRF).
"""

import logging
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client import models
from fastembed import TextEmbedding, SparseTextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from configs.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL,
    RERANKER_MODEL,
    RERANKER_TRUNCATION_LENGTH,
    RERANKER_FETCH_MULTIPLIER,
    DENSE_WEIGHT,
    SPARSE_WEIGHT,
    USE_RERANKER,
)

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Retriever for Qdrant using FastEmbed and server-side RRF fusion."""

    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        location: str = None,
        dense_model_name: str = EMBEDDING_MODEL,
        sparse_model_name: str = SPARSE_EMBEDDING_MODEL,
        reranker_model_name: str = RERANKER_MODEL,
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
        if "bge-m3" in dense_model_name.lower():
            from arxiv_scholar.embedding.st_embedder import SentenceTransformerEmbedder
            self.dense_model = SentenceTransformerEmbedder(model_name=dense_model_name)
            self._is_st = True
        else:
            self.dense_model = TextEmbedding(model_name=dense_model_name)
            self._is_st = False
        
        logger.info(f"Loading sparse model: {sparse_model_name}")
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        
        self.reranker_model = None
        if reranker_model_name:
            logger.info(f"Loading fastembed reranker model: {reranker_model_name}")
            self.reranker_model = TextCrossEncoder(model_name=reranker_model_name)

    def retrieve(self, query_text: str, limit: int = 20, use_reranker: bool = USE_RERANKER, dense_query_text: str = None, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Executes a hybrid search query with server-side RRF.
        
        Args:
            query_text: The raw natural language query.
            limit: The maximum number of top-K results to return (default 20).
            use_reranker: Whether to apply cross-encoder reranking.
            dense_query_text: Optional text to use for the dense embedding (useful for HyDE). Defaults to query_text.
            filters: Optional dictionary of metadata filters.
            
        Returns:
            A list of dictionaries containing chunk_id, text, score, and metadata.
        """
        # 3. Generate the Dense vector for the query_text.
        # fastembed returns generators, so we consume it into a list and take the first item
        dq = dense_query_text if dense_query_text else query_text
        if self._is_st:
            dense_vector = self.dense_model.embed([dq])[0]
        else:
            dense_vector = list(self.dense_model.embed([dq]))[0].tolist()

        # 4. Generate the Sparse vector for the query_text.
        sparse_result = list(self.sparse_model.embed([query_text]))[0]
        # fastembed SparseEmbedding objects have .indices and .values properties
        sparse_vector = models.SparseVector(
            indices=sparse_result.indices,
            values=sparse_result.values,
        )

        # 5. The Prefetch Construction
        qdrant_filter = None
        if filters:
            must_conditions = []
            for key, val_obj in filters.items():
                if isinstance(val_obj, dict) and "operator" in val_obj and "value" in val_obj:
                    op = val_obj["operator"]
                    v = val_obj["value"]
                    if op == "==":
                        must_conditions.append(models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=v)))
                    else:
                        range_args = {}
                        if op == ">=": range_args["gte"] = v
                        elif op == ">": range_args["gt"] = v
                        elif op == "<=": range_args["lte"] = v
                        elif op == "<": range_args["lt"] = v
                        must_conditions.append(models.FieldCondition(key=f"metadata.{key}", range=models.Range(**range_args)))
                else:
                    must_conditions.append(models.FieldCondition(key=f"metadata.{key}", match=models.MatchValue(value=val_obj)))
            
            if must_conditions:
                qdrant_filter = models.Filter(must=must_conditions)

        fetch_limit = limit * RERANKER_FETCH_MULTIPLIER if (use_reranker and self.reranker_model) else limit
        
        prefetch_dense = models.Prefetch(
            query=dense_vector,
            using="",
            limit=fetch_limit,
            filter=qdrant_filter,
        )

        prefetch_sparse = models.Prefetch(
            query=sparse_vector,
            using="bm25",
            limit=fetch_limit,
            filter=qdrant_filter,
        )

        # Trigger server-side Reciprocal Rank Fusion with custom weights
        # We assign DENSE_WEIGHT to prefetch_dense and SPARSE_WEIGHT to prefetch_sparse
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[prefetch_dense, prefetch_sparse],
            query=models.RrfQuery(rrf=models.Rrf(weights=[DENSE_WEIGHT, SPARSE_WEIGHT])),
            limit=fetch_limit,
        )

        # 7. Output Formatting
        # Extract the payload and the fused score from the returned Qdrant points
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append({
                "chunk_id": str(point.id),
                "text": payload.get("content", ""),
                "score": point.score,
                "metadata": payload.get("metadata", {}),
            })
            
        if use_reranker and self.reranker_model and results:
            results = self.rerank_results(query_text, results, limit)

        return results

    def rerank_results(self, query_text: str, results: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
        """Reranks a list of result chunks against a query using the cross-encoder."""
        if not self.reranker_model or not results:
            return results[:limit]
            
        # Predict cross-encoder scores
        documents = [res["text"][:RERANKER_TRUNCATION_LENGTH] for res in results]
        cross_scores = list(self.reranker_model.rerank(query_text, documents))
        
        # Update scores and sort descending
        for i, res in enumerate(results):
            res["score"] = float(cross_scores[i])
            
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:limit]
