# Implementation Plan: Metadata Filtering & Router Overrides

## Objective
To enable actual database-level metadata filtering (e.g., filtering papers by publication year) and ensure that the query router reliably defaults to these filters rather than being bypassed by the ML classifier.

## Phase 0: Update Ingestion Scripts (Extract Year)
**Files:** `src/arxiv_scholar/ingestion/gcs.py` and `src/arxiv_scholar/ingestion/local.py`

*   **Problem:** Currently, the system only stores `arxiv_id`, `title`, and file info in Qdrant. There is no mathematical `year` field to filter on.
*   **Action:** Update the ingestion parsing logic to extract the year from the `arxiv_id` (e.g., `2101.00001` -> `2021`). Store this as an integer field (`"year": 2021`) inside the `metadata` dictionary before it gets saved to Qdrant.

## Phase 1: Router Fallbacks as Hard Overrides & Cleanup
**File:** `src/arxiv_scholar/retrieval/router.py`

*   **Problem:** Currently, the ML classifier evaluates the query first and returns immediately. Heuristics are either unreachable or redundant since the ML model naturally handles comparison semantics well.
*   **Action:** Remove redundant heuristics and reorder the necessary ones in `MLQueryRouter.route()`:
    1.  **Remove spaCy and keyword logic:** The ML model handles "vs/compare" queries accurately. Delete the spaCy dependency to save ~15ms startup time.
    2.  **Short queries** -> `Route.HYDE`
    3.  **Metadata regex** (years) -> `Route.DECOMPOSE` (Hard override, since ML misses this)
    4.  **ML Classifier** -> Let it decide if none of the explicit rules matched.
    5.  **Default Fallback** -> `Route.DIRECT`

## Phase 2: Structured Metadata Extraction
**File:** `src/arxiv_scholar/llm/service.py`

*   **Problem:** The `decompose_query` prompt only asks the LLM to rewrite the query into sub-queries (strings) and doesn't extract metadata constraints.
*   **Action:** Update the prompt to ask for structured filters.
    *   **New JSON Output Format:**
        ```json
        {
            "sub_queries": ["diffusion models"],
            "filters": {
                "year": {
                    "operator": ">=",
                    "value": 2022
                }
            }
        }
        ```
    *   Update the function to parse and return a dictionary containing both `sub_queries` and `filters`.

## Phase 3: Plumb Filters through the Orchestrator
**File:** `src/arxiv_scholar/retrieval/orchestrator.py`

*   **Action:**
    *   Update `_execute_decompose` to unpack the `filters` from `llm_service.decompose_query(query)`.
    *   Pass the extracted `filters` into `_execute_direct`.
    *   Update `_execute_direct` signature to pass `filters` to `self.retriever.retrieve`.

## Phase 4: Enforce Filters in Qdrant
**File:** `src/arxiv_scholar/retrieval/retrieval.py`

*   **Action:**
    *   Update `HybridRetriever.retrieve` signature to accept an optional `filters: Dict = None` argument.
    *   If `filters` are provided (e.g., `year`), map them to `qdrant_client.models.Filter` and `qdrant_client.models.FieldCondition`.
    *   Apply the filter to the `filter` argument of the `Prefetch` objects for both the dense and sparse paths.

## Phase 5: Global Late Re-ranking for Decomposed Queries
**Files:** `src/arxiv_scholar/retrieval/orchestrator.py` & `src/arxiv_scholar/retrieval/retrieval.py`

*   **Problem:** Currently, decomposed sub-queries apply the cross-encoder independently against sub-queries, resulting in mathematically uncalibrated scores and potential loss of the user's original intent.
*   **Action:**
    *   Expose a `rerank_results(query, results)` method inside `HybridRetriever` to decouple re-ranking from retrieval.
    *   In `Orchestrator._execute_decompose`, force `use_reranker=False` for all sub-query retrievals.
    *   Pool and deduplicate all retrieved chunks from the sub-queries.
    *   Apply `self.retriever.rerank_results(original_parent_query, pooled_chunks)` globally *once* at the end.
    *   Sort the globally scored chunks and truncate to `limit`.

## Phase 6: Core Optimizations & Bug Fixes
**Files:** `pyproject.toml`, `src/arxiv_scholar/retrieval/retrieval.py`, `src/arxiv_scholar/storage/qdrant_store.py`

*   **1. Missing ML Router Dependencies (Showstopper):** Add `joblib` and `scikit-learn` to `pyproject.toml` so the ML Query Router stops silently failing.
*   **2. Reranker Context Starvation:** Remove the hardcoded `[:500]` truncation slice in `HybridRetriever.retrieve()` so the cross-encoder can evaluate the full context. Move any context limit configurations to `configs/config.py`.
*   **3. Qdrant Payload Limit Crash:** Replace `self._client.upsert(points=...)` with `self._client.upload_points()` in `QdrantVectorStore` to automatically batch vector uploads.
*   **4. Dependency Bloat Cleanup:** Remove `spacy` from `pyproject.toml` since we are deleting the keyword heuristics in Phase 1.

---

## Design Decision: Regex vs. ML for Metadata Routing
We have explicitly chosen to use **Regex (Hard Overrides)** rather than updating the ML classifier to detect metadata constraints like dates/years. The reasoning is:
1. **100% Reliability:** Regex acts as a deterministic gatekeeper. If a user asks for papers from 2023, the system guarantees it will catch it, whereas ML models can hallucinate or fail on rare phrasing.
2. **Embeddings & Math:** Dense embedding models (like BGE) map semantic topics well, but represent numbers as arbitrary vectors. They are structurally poor at learning strict mathematical ranges.
3. **Speed & Efficiency:** A compiled regex check takes `<0.1ms`, whereas generating embeddings and running an ML inference takes `~5ms`.
4. **Maintainability:** Regex requires zero synthetic training data, no `.joblib` artifact updates, and no retraining pipelines.
