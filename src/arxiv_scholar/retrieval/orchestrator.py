import os
import re
import json
import asyncio
import logging
from typing import Any, Dict, List
from arxiv_scholar.llm.service import LLMService

from arxiv_scholar.retrieval.retrieval import HybridRetriever
from arxiv_scholar.retrieval.router import Route, MLQueryRouter
from configs.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION,
    EMBEDDING_MODEL,
    SPARSE_EMBEDDING_MODEL,
    RERANKER_MODEL,
    RERANKER_FETCH_MULTIPLIER,
    USE_RERANKER,
)

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(
        self,
        collection_name: str = QDRANT_COLLECTION,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        dense_model_name: str = EMBEDDING_MODEL,
        sparse_model_name: str = SPARSE_EMBEDDING_MODEL,
        reranker_model_name: str = RERANKER_MODEL,
    ):
        self.collection_name = collection_name
        
        self.retriever = HybridRetriever(
            collection_name=collection_name,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            dense_model_name=dense_model_name,
            sparse_model_name=sparse_model_name,
            reranker_model_name=reranker_model_name,
        )
        
        # Initialize ML router
        self.router = MLQueryRouter()
        
        # Initialize LLM Service
        self.llm_service = LLMService()

    async def retrieve(self, query: str, limit: int = 20, use_reranker: bool = USE_RERANKER) -> List[Dict[str, Any]]:
        logger.info(f"Orchestrator.retrieve called with query: '{query}', limit={limit}, use_reranker={use_reranker}")
        # Compute dense embedding to feed to the ML router
        dense_out = list(self.retriever.dense_model.embed([query]))[0]
        dense_vec = dense_out.tolist() if hasattr(dense_out, "tolist") else dense_out
        route = self.router.route(query, query_vector=dense_vec)
        logger.info(f"Router selected route: {route.name}")
        # Note: In a real system, we'd return the route to log the prometheus path metric
        # We will attach the path to the returned payload for benchmarking scripts to read
        
        if route == Route.DIRECT:
            results = await self._execute_direct(query, limit, use_reranker)
        elif route == Route.DECOMPOSE:
            results = await self._execute_decompose(query, limit, use_reranker)
        elif route == Route.HYDE:
            if not self.llm_service.client:
                results = await self._execute_direct(query, limit, use_reranker)
            else:
                results = await self._execute_hyde(query, limit, use_reranker)
        else:
            results = await self._execute_direct(query, limit, use_reranker)
            
        # Attach route metadata so benchmark can track it
        for r in results:
            r["_query_path"] = route.value
            
        return results

    async def _execute_direct(self, query: str, limit: int = 20, use_reranker: bool = USE_RERANKER, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.retriever.retrieve, query, limit, use_reranker, None, filters)

    async def _execute_hyde(self, query: str, limit: int = 20, use_reranker: bool = USE_RERANKER) -> List[Dict[str, Any]]:
        # 1. Generate hypothetical abstract via API
        abstract = await self.llm_service.generate_hyde_abstract(query)
        logger.info(f"Generated HyDE abstract: '{abstract[:100]}...'")
        
        # 2. Hybrid search (Dense uses abstract, Sparse uses original query)
        return await asyncio.to_thread(self.retriever.retrieve, query, limit, use_reranker, abstract)

    async def _execute_decompose(self, query: str, limit: int = 20, use_reranker: bool = USE_RERANKER) -> List[Dict[str, Any]]:
        if not self.llm_service.client:
            logger.warning("No LLM client configured for DECOMPOSE. Falling back to DIRECT.")
            return await self._execute_direct(query, limit, use_reranker)
            
        # 1. Generate fully contextualized sub-queries via LLM
        decomp_result = await self.llm_service.decompose_query(query)
        
        if isinstance(decomp_result, dict):
            sub_queries = decomp_result.get("sub_queries", [query])
            filters = decomp_result.get("filters", None)
        else:
            sub_queries = decomp_result
            filters = None
        
        if not sub_queries:
            sub_queries = [query]
            
        logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries} with filters: {filters}")
            
        # 2. Dynamic Compute Budgeting
        # Allocate the global cross-encoder budget equally across sub-queries
        global_budget = limit * RERANKER_FETCH_MULTIPLIER if use_reranker else limit
        sub_limit = max(limit, global_budget // len(sub_queries))
            
        # 3. Fire concurrent searches for each sub-query (force use_reranker=False)
        tasks = [self._execute_direct(sq, sub_limit, use_reranker=False, filters=filters) for sq in sub_queries]
        results_lists = await asyncio.gather(*tasks)
        
        # 3. Merge and deduplicate interface
        all_results = []
        seen = set()
        for r_list in results_lists:
            for r in r_list:
                if r["chunk_id"] not in seen:
                    seen.add(r["chunk_id"])
                    all_results.append(r)
                    
        # 4. Global Late Re-ranking
        if use_reranker:
            all_results = await asyncio.to_thread(self.retriever.rerank_results, query, all_results, limit)
        else:
            all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)[:limit]
            
        return all_results
