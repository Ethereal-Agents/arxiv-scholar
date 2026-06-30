import os
import time
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import importlib.util
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.py"))
spec = importlib.util.spec_from_file_location("local_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

from arxiv_scholar.retrieval.orchestrator import Orchestrator
from arxiv_scholar.llm.service import LLMService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1.5, min=2, max=60),
    before_sleep=lambda retry_state: logger.warning(f"LLM rating error. Retrying in {retry_state.next_action.sleep:.1f}s...")
)
async def _call_llm_with_retry(llm_service, prompt):
    return await llm_service._call_llm(prompt, max_tokens=10, temperature=0.0)

async def rate_chunk(llm_service, query, chunk_text):
    prompt = f"""
    Rate the relevance of the following text chunk to the query on a scale of 0 to 2:
    2 = Directly and comprehensively answers the query.
    1 = Partially relevant or contains some related information, but doesn't fully answer it.
    0 = Irrelevant.
    
    Query: {query}
    Chunk: {chunk_text}
    
    Return ONLY a single integer (0, 1, or 2).
    """
    try:
        response = await _call_llm_with_retry(llm_service, prompt)
        # Attempt to extract just the integer if the model returns extra text
        import re
        match = re.search(r'\b[0-2]\b', response)
        if match:
            return int(match.group())
        return 0
    except Exception as e:
        logger.error(f"Error rating chunk after retries: {e}")
        return 0

async def rate_chunks_batch(llm_service, query, chunks_text):
    tasks = [rate_chunk(llm_service, query, t) for t in chunks_text]
    return await asyncio.gather(*tasks)

def calculate_ndcg_graded(grades, k=10):
    dcg = 0.0
    idcg = 0.0
    
    for i, grade in enumerate(grades[:k]):
        dcg += grade / np.log2((i + 1) + 1)
        
    sorted_grades = sorted(grades, reverse=True)
    for i, grade in enumerate(sorted_grades[:k]):
        idcg += grade / np.log2((i + 1) + 1)
        
    return dcg / idcg if idcg > 0 else 0.0

async def run_judged_evaluation(data_file: str, collection_name: str, log_file: str = None):
    logger.info(f"Loading eval dataset: {data_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing {data_file}.")
        
    with open(data_file, "r") as f:
        queries = [json.loads(line) for line in f]
        
    # Filter out adversarial queries
    queries = [q for q in queries if q.get("query_type") == "standard"]
        
    retriever = Orchestrator(
        collection_name=collection_name, 
        qdrant_host=config.AppConfig().qdrant_host, 
        qdrant_port=config.AppConfig().qdrant_port,
        qdrant_url=config.AppConfig().qdrant_url,
        qdrant_api_key=config.AppConfig().qdrant_api_key,
        qdrant_timeout=config.AppConfig().qdrant_timeout,
        reranker_model_name="jinaai/jina-reranker-v1-tiny-en",
        use_reranker=True
    )
    
    llm_service = LLMService(
        api_key=config.AppConfig().llm_api_key, 
        base_url=config.AppConfig().llm_base_url, 
        model=config.AppConfig().llm_model
    )
    
    base_point_recalls = []
    base_jr10 = []
    base_jr20 = []
    base_ndcgs10 = []
    base_ndcgs20 = []
    
    rr_point_recalls = []
    rr_jr10 = []
    rr_jr20 = []
    rr_ndcgs10 = []
    rr_ndcgs20 = []
    
    base_latencies = []
    rr_latencies = []
    
    if log_file:
        log_f = open(log_file, "w")
    else:
        log_f = None
    
    logger.info(f"Running LLM-judged benchmarking against {collection_name} for {len(queries)} queries...")
    for q in tqdm(queries):
        query_id = q.get("query_id", "unknown")
        query_text = q["query"]
        target_id = q["positive_chunk"]["chunk_id"]
        
        start_t = time.perf_counter()
        results = await retriever.retrieve(query_text, limit=20, use_reranker=False)
        latency = time.perf_counter() - start_t
        
        retrieved_ids = [str(res["chunk_id"]) for res in results]
        chunks_text = [res["text"] for res in results]
        
        # Base Metrics
        base_pr20 = 1.0 if str(target_id) in retrieved_ids[:20] else 0.0
        base_point_recalls.append(base_pr20)
        
        grades = await rate_chunks_batch(llm_service, query_text, chunks_text)
        
        b_jr10 = 1.0 if any(g >= 1 for g in grades[:10]) else 0.0
        b_jr20 = 1.0 if any(g >= 1 for g in grades[:20]) else 0.0
        b_ndcg10 = calculate_ndcg_graded(grades, k=10)
        b_ndcg20 = calculate_ndcg_graded(grades, k=20)
        
        base_jr10.append(b_jr10)
        base_jr20.append(b_jr20)
        base_ndcgs10.append(b_ndcg10)
        base_ndcgs20.append(b_ndcg20)
        
        # Reranked Metrics (ZERO extra LLM calls!)
        grade_map = {res["chunk_id"]: grades[i] for i, res in enumerate(results)}
        
        # Apply reranker locally to the retrieved chunks
        start_rr_t = time.perf_counter()
        reranked_results = retriever.retriever.rerank_results(query_text, results, limit=20)
        rr_latency = latency + (time.perf_counter() - start_rr_t)
        
        reranked_ids = [str(res["chunk_id"]) for res in reranked_results]
        
        # Map the cached grades to the newly sorted array
        reranked_grades = [grade_map.get(cid, 0) for cid in reranked_ids]
        
        rr_pr20 = 1.0 if str(target_id) in reranked_ids[:20] else 0.0
        r_jr10 = 1.0 if any(g >= 1 for g in reranked_grades[:10]) else 0.0
        r_jr20 = 1.0 if any(g >= 1 for g in reranked_grades[:20]) else 0.0
        r_ndcg10 = calculate_ndcg_graded(reranked_grades, k=10)
        r_ndcg20 = calculate_ndcg_graded(reranked_grades, k=20)
        
        rr_point_recalls.append(rr_pr20)
        rr_jr10.append(r_jr10)
        rr_jr20.append(r_jr20)
        rr_ndcgs10.append(r_ndcg10)
        rr_ndcgs20.append(r_ndcg20)
        
        base_latencies.append(latency)
        rr_latencies.append(rr_latency)
        
        if log_f:
            log_entry = {
                "query_id": query_id,
                "query": query_text,
                "target_chunk_id": target_id,
                "base_results": [
                    {"chunk_id": res["chunk_id"], "score": float(res["score"]), "grade": int(g)}
                    for res, g in zip(results, grades)
                ],
                "reranked_results": [
                    {"chunk_id": res["chunk_id"], "score": float(res["score"]), "grade": int(g)}
                    for res, g in zip(reranked_results, reranked_grades)
                ],
                "metrics": {
                    "base_pr20": base_pr20,
                    "base_jr10": b_jr10,
                    "base_ndcg10": b_ndcg10,
                    "rr_pr20": rr_pr20,
                    "rr_jr10": r_jr10,
                    "rr_ndcg10": r_ndcg10
                }
            }
            log_f.write(json.dumps(log_entry) + "\n")
            log_f.flush()
            
    if log_f:
        log_f.close()
        
    p95_base_lat = np.percentile(base_latencies, 95) * 1000
    p95_rr_lat = np.percentile(rr_latencies, 95) * 1000
    
    metrics_base = {
        "Collection": f"{collection_name} (Base)",
        "PR@20": np.mean(base_point_recalls),
        "JR@10": np.mean(base_jr10),
        "JR@20": np.mean(base_jr20),
        "nDCG@10": np.mean(base_ndcgs10),
        "nDCG@20": np.mean(base_ndcgs20),
        "Latency_p95": p95_base_lat
    }
    
    metrics_rr = {
        "Collection": f"{collection_name} (Reranked)",
        "PR@20": np.mean(rr_point_recalls),
        "JR@10": np.mean(rr_jr10),
        "JR@20": np.mean(rr_jr20),
        "nDCG@10": np.mean(rr_ndcgs10),
        "nDCG@20": np.mean(rr_ndcgs20),
        "Latency_p95": p95_rr_lat
    }
    
    return [metrics_base, metrics_rr]

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/eval_dataset.jsonl")
    parser.add_argument("--collection", default="arxiv_papers", help="Target collection name.")
    parser.add_argument("--log_file", default=None, help="Path to JSONL file where detailed metrics and logs will be saved.")
    args = parser.parse_args()
    
    results = await run_judged_evaluation(args.data, args.collection, args.log_file)
    
    print("\n" + "="*110)
    print("JUDGED BENCHMARK RESULTS")
    print("="*110)
    
    format_str = "{:<28} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}"
    print(format_str.format("Collection", "PR@20", "JR@10", "JR@20", "nDCG@10", "nDCG@20", "p95 (ms)"))
    print("-" * 110)
    
    for r in results:
        print(format_str.format(
            r["Collection"],
            f"{r['PR@20']:.3f}",
            f"{r['JR@10']:.3f}",
            f"{r['JR@20']:.3f}",
            f"{r['nDCG@10']:.3f}",
            f"{r['nDCG@20']:.3f}",
            f"{r['Latency_p95']:.1f}"
        ))
    print("="*110)

if __name__ == "__main__":
    asyncio.run(main_async())
