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

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Retriever for Qdrant using FastEmbed and server-side RRF fusion."""

    def __init__(
        self,
        collection_name: str = "arxiv-scholar",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_url: str = "",
        qdrant_api_key: str = "",
        location: str = None,
        dense_model_name: str = "BAAI/bge-m3",
        sparse_model_name: str = "Qdrant/bm25",
        reranker_model_name: str = "jina-reranker-v1-tiny-en",
        use_reranker: bool = False,
        reranker_truncation_length: int = 8192,
        reranker_fetch_multiplier: int = 4
    ) -> None:
        """Initializes the retriever and its global state (models and db client)."""
        self.collection_name = collection_name
        
        # 1. Initialize the Qdrant client
        if location:
            self.client = QdrantClient(location=location, timeout=60.0)
        elif qdrant_url and qdrant_api_key:
            self.client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=60.0)
        else:
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=60.0)
            
        logger.info(f"Initialized QdrantClient for collection '{collection_name}'")

        # 2. Initialize two fastembed models (Dense and Sparse).
        # These are loaded globally for the instance and cached.
        # Initialize fastembed models FIRST to prevent ONNX/PyTorch deadlocks on Mac
        logger.info(f"Loading sparse model: {sparse_model_name}")
        self.sparse_model = SparseTextEmbedding(model_name=sparse_model_name)
        
        self.reranker_model = None
        self.use_reranker = use_reranker
        self.reranker_truncation_length = reranker_truncation_length
        self.reranker_fetch_multiplier = reranker_fetch_multiplier
        if use_reranker and reranker_model_name:
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder
                logger.info(f"Loading fastembed reranker model: {reranker_model_name}")
                self.reranker_model = TextCrossEncoder(model_name=reranker_model_name)
            except ImportError:
                logger.warning("fastembed not installed. Reranker disabled.")
                self.reranker_model = None

        logger.info(f"Loading dense model: {dense_model_name}")
        if "bge-m3" in dense_model_name.lower():
            from arxiv_scholar.embedding.st_embedder import SentenceTransformerEmbedder
            self.dense_model = SentenceTransformerEmbedder(model_name=dense_model_name)
            self._is_st = True
        else:
            self.dense_model = TextEmbedding(model_name=dense_model_name)
            self._is_st = False
            
        # Proactively validate embedding dimension against Qdrant collection
        try:
            collection_info = self.client.get_collection(self.collection_name)
            vec_params = collection_info.config.params.vectors
            if isinstance(vec_params, models.VectorParams):
                qdrant_dim = vec_params.size
            elif isinstance(vec_params, dict) and "" in vec_params:
                qdrant_dim = vec_params[""].size
            else:
                qdrant_dim = None

            if qdrant_dim is not None:
                if hasattr(self.dense_model, "dimension"):
                    model_dim = self.dense_model.dimension
                else:
                    if self._is_st:
                        test_vec = self.dense_model.embed(["test"])[0]
                    else:
                        test_vec = list(self.dense_model.embed(["test"]))[0]
                    model_dim = len(test_vec)
                    
                if model_dim != qdrant_dim:
                    raise ValueError(f"Vector dimension mismatch! Qdrant collection '{self.collection_name}' expects {qdrant_dim}D vectors, but embedder '{dense_model_name}' produces {model_dim}D. Please check your configurations.")
        except Exception as e:
            if isinstance(e, ValueError) and "Vector dimension mismatch" in str(e):
                raise
            logger.warning(f"Could not validate collection dimensions during init: {e}")

    def retrieve(self, query_text: str, limit: int = 20, use_reranker: bool = None, dense_query_text: str = None, filters: Dict[str, Any] = None, dense_weight: float = 1.0, sparse_weight: float = 0.3) -> List[Dict[str, Any]]:
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
        if use_reranker is None:
            use_reranker = self.use_reranker
            
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

        fetch_limit = limit * self.reranker_fetch_multiplier if (use_reranker and self.reranker_model) else limit
        
        # 6. Fetch independently and fuse manually with Min-Max normalization
        # We batch both queries into a single network round-trip to minimize latency
        try:
            batch_responses = self.client.query_batch_points(
                collection_name=self.collection_name,
                requests=[
                    models.QueryRequest(
                        query=dense_vector,
                        using="",
                        limit=fetch_limit,
                        filter=qdrant_filter,
                        with_payload=True,
                    ),
                    models.QueryRequest(
                        query=sparse_vector,
                        using="bm25",
                        limit=fetch_limit,
                        filter=qdrant_filter,
                        with_payload=True,
                    )
                ]
            )
        except Exception as e:
            if "Bad request" in str(e) or "Index required" in str(e):
                logger.warning(f"Qdrant rejected filters (likely LLM hallucination). Falling back to no filters. Error: {e}")
                batch_responses = self.client.query_batch_points(
                    collection_name=self.collection_name,
                    requests=[
                        models.QueryRequest(
                            query=dense_vector,
                            using="",
                            limit=fetch_limit,
                            filter=None,
                            with_payload=True,
                        ),
                        models.QueryRequest(
                            query=sparse_vector,
                            using="bm25",
                            limit=fetch_limit,
                            filter=None,
                            with_payload=True,
                        )
                    ]
                )
            else:
                raise
        
        dense_response = batch_responses[0]
        sparse_response = batch_responses[1]

        def normalize_scores(points):
            if not points:
                return {}
            scores = {str(p.id): p.score for p in points}
            min_val = min(scores.values())
            max_val = max(scores.values())
            if max_val == min_val:
                return {k: 0.0 for k in scores}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}

        norm_dense = normalize_scores(dense_response.points)
        norm_sparse = normalize_scores(sparse_response.points)

        all_points = {str(p.id): p for p in dense_response.points + sparse_response.points}
        
        # 7. Output Formatting & Fusion
        results_unsorted = []
        for chunk_id, point in all_points.items():
            d_score = norm_dense.get(chunk_id, 0.0)
            s_score = norm_sparse.get(chunk_id, 0.0)
            fused_score = (dense_weight * d_score) + (sparse_weight * s_score)
            
            payload = point.payload or {}
            results_unsorted.append({
                "chunk_id": chunk_id,
                "text": payload.get("content", ""),
                "score": fused_score,
                "metadata": payload.get("metadata", {}),
            })
            
        results = sorted(results_unsorted, key=lambda x: x["score"], reverse=True)[:fetch_limit]
            
        if use_reranker and self.reranker_model and results:
            results = self.rerank_results(query_text, results, limit)

        return results

    def rerank_results(self, query_text: str, results: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
        """Reranks a list of result chunks against a query using the cross-encoder."""
        if not self.reranker_model or not results:
            return results[:limit]
            
        # Predict cross-encoder scores
        documents = [res["text"][:self.reranker_truncation_length] for res in results]
        cross_scores = list(self.reranker_model.rerank(query_text, documents))
        
        # Update scores and sort descending
        for i, res in enumerate(results):
            res["score"] = float(cross_scores[i])
            
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:limit]
