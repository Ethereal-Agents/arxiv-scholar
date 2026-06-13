import os
import time
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm
from prometheus_client import start_http_server, Summary, Histogram, Counter
import asyncio

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import importlib.util
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.py"))
spec = importlib.util.spec_from_file_location("local_config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

from arxiv_scholar.retrieval.orchestrator import Orchestrator
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Prometheus Metrics
RETRIEVAL_LATENCY = Summary('retrieval_latency_seconds', 'Time spent retrieving from Qdrant')
RETRIEVAL_LATENCY_HIST = Histogram('retrieval_latency_histogram_seconds', 'Retrieval latency histogram', buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0])
QUERY_PATH_COUNTER = Counter('query_path_total', 'Distribution of query paths taken', ['path'])

def calculate_ndcg(retrieved_ids, target_id, hard_negative_ids, k=10):
    dcg = 0.0
    idcg = 1.0 # Max possible gain is 1.0 at rank 1, plus 0.1 for up to K-1 hard negatives
    
    # Calculate IDCG dynamically based on available hard negatives
    for i in range(1, min(k, len(hard_negative_ids) + 1)):
        idcg += 0.1 / np.log2((i+1) + 1)
        
    for i, res_id in enumerate(retrieved_ids[:k]):
        rank = i + 1
        if str(res_id) == str(target_id):
            rel = 1.0
        elif str(res_id) in hard_negative_ids:
            rel = 0.1
        else:
            rel = 0.0
            
        dcg += rel / np.log2(rank + 1)
        
    return dcg / idcg if idcg > 0 else 0.0

async def run_evaluation(data_file: str, collection_name: str):
    logger.info(f"Loading eval dataset: {data_file}")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing {data_file}. Did you run generate_eval_dataset.py?")
        
    with open(data_file, "r") as f:
        queries = [json.loads(line) for line in f]
        
    # Filter out adversarial queries due to logical ground-truth mismatch
    queries = [q for q in queries if q.get("query_type") == "standard"]
        
    retriever = Orchestrator(
        collection_name=collection_name, 
        qdrant_host=config.AppConfig().qdrant_host, 
        qdrant_port=config.AppConfig().qdrant_port,
        reranker_model_name="jinaai/jina-reranker-v1-tiny-en"
    )
    
    results_list = []
    
    for use_reranker in [False]:
        mode_name = f"{collection_name} (Baseline)"
        
        recalls_5 = []
        recalls_10 = []
        recalls_20 = []
        ndcgs = []
        latencies = []
        
        logger.info(f"Running benchmarking against {mode_name} for {len(queries)} queries...")
        for q in tqdm(queries):
            query_text = q["query"]
            target_id = q["positive_chunk"]["chunk_id"]
            hard_neg_ids = [hn["chunk_id"] for hn in q["hard_negatives"]]
            
            with RETRIEVAL_LATENCY.time():
                start_t = time.perf_counter()
                results = await retriever.retrieve(query_text, limit=20, use_reranker=use_reranker)
                end_t = time.perf_counter()
                latency = end_t - start_t
                RETRIEVAL_LATENCY_HIST.observe(latency)
                
            if results:
                path = results[0].get("_query_path", "direct")
                QUERY_PATH_COUNTER.labels(path=path).inc()
                
            retrieved_ids = [str(res["chunk_id"]) for res in results]
            
            # Calculate Metrics
            r5 = 1.0 if target_id in retrieved_ids[:5] else 0.0
            r10 = 1.0 if target_id in retrieved_ids[:10] else 0.0
            r20 = 1.0 if target_id in retrieved_ids[:20] else 0.0
            
            ndcg_val = calculate_ndcg(retrieved_ids, target_id, hard_neg_ids, k=10)
            
            recalls_5.append(r5)
            recalls_10.append(r10)
            recalls_20.append(r20)
            ndcgs.append(ndcg_val)
            latencies.append(latency)
            
        metrics = {
            "Collection": mode_name,
            "Queries": len(queries),
            "Recall@5": np.mean(recalls_5),
            "Recall@10": np.mean(recalls_10),
            "Recall@20": np.mean(recalls_20),
            "nDCG@10": np.mean(ndcgs),
            "Latency_p50": np.percentile(latencies, 50),
            "Latency_p95": np.percentile(latencies, 95),
            "Latency_p99": np.percentile(latencies, 99),
            "Avg_Latency_ms": np.mean(latencies) * 1000
        }
        results_list.append(metrics)
        
    return results_list

def calculate_cost(latency_mean_ms):
    # AWS EC2 t3.xlarge (4 vCPUs, 16GB) is ~$0.166/hr -> ~$0.000046/sec
    # We estimate cost based on embedding + retrieval compute time per 1000 queries
    cost_per_sec = 0.166 / 3600
    sec_per_1k = (latency_mean_ms / 1000.0) * 1000
    return sec_per_1k * cost_per_sec

async def run_alpha_sweep(data_file: str, collection_name: str, retriever, iterations: int = 1):
    logger.info(f"Running Alpha Sweep on {data_file} for collection {collection_name} ({iterations} iterations)")
    
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Missing {data_file}.")
        
    with open(data_file, "r") as f:
        queries = [json.loads(line) for line in f]
        
    queries = [q for q in queries if q.get("query_type") == "standard"]
    
    alpha_metrics = {round(a, 1): {"recalls": [], "ndcgs": [], "latencies": [], "fusion_latencies": []} for a in np.arange(0.0, 1.1, 0.1)}
    
    from qdrant_client.http import models
    for iteration in range(iterations):
        for q in tqdm(queries, desc=f"Alpha Sweep (Iter {iteration+1}/{iterations})"):
            query_text = q["query"]
            target_id = str(q["positive_chunk"]["chunk_id"])
            hard_neg_ids = [str(hn["chunk_id"]) for hn in q.get("hard_negatives", [])]
            
            # 1. Embed once
            start_t = time.perf_counter()
            if retriever._is_st:
                dense_vector = retriever.dense_model.embed([query_text])[0]
            else:
                dense_vector = list(retriever.dense_model.embed([query_text]))[0]
            
            sparse_result = list(retriever.sparse_model.embed([query_text]))[0]
            sparse_vector = models.SparseVector(
                indices=sparse_result.indices,
                values=sparse_result.values,
            )
            
            # 2. Fetch once (limit=100)
            batch_responses = await asyncio.to_thread(
                retriever.client.query_batch_points,
                collection_name=retriever.collection_name,
                requests=[
                    models.QueryRequest(query=dense_vector, using="", limit=100, with_payload=True),
                    models.QueryRequest(query=sparse_vector, using="bm25", limit=100, with_payload=True)
                ]
            )
            
            dense_response = batch_responses[0]
            sparse_response = batch_responses[1]
            
            # 3. Normalize scores
            def normalize_scores(points):
                if not points: return {}
                scores = [p.score for p in points]
                min_s, max_s = min(scores), max(scores)
                if max_s - min_s == 0: return {str(p.id): 1.0 for p in points}
                return {str(p.id): (p.score - min_s) / (max_s - min_s) for p in points}

            norm_dense = normalize_scores(dense_response.points)
            norm_sparse = normalize_scores(sparse_response.points)
            all_points = {str(p.id): p for p in list(dense_response.points) + list(sparse_response.points)}
            
            base_latency = time.perf_counter() - start_t
            
            for alpha in np.arange(0.0, 1.1, 0.1):
                alpha = round(alpha, 1)
                
                # Combine and sort
                start_fuse_t = time.perf_counter()
                combined = []
                for cid in all_points.keys():
                    d_score = norm_dense.get(cid, 0.0)
                    s_score = norm_sparse.get(cid, 0.0)
                    final_score = (alpha * d_score) + ((1 - alpha) * s_score)
                    combined.append((cid, final_score))
                    
                combined.sort(key=lambda x: x[1], reverse=True)
                final_top_20 = [x[0] for x in combined[:20]]
                
                fusion_latency_ms = (time.perf_counter() - start_fuse_t) * 1000
                latency = base_latency + (time.perf_counter() - start_fuse_t)
                
                if target_id in final_top_20:
                    alpha_metrics[alpha]["recalls"].append(1.0)
                else:
                    alpha_metrics[alpha]["recalls"].append(0.0)
                    
                ndcg_val = calculate_ndcg(final_top_20, target_id, hard_neg_ids, k=10)
                alpha_metrics[alpha]["ndcgs"].append(ndcg_val)
                alpha_metrics[alpha]["latencies"].append(latency)
                alpha_metrics[alpha]["fusion_latencies"].append(fusion_latency_ms)
                
    print("\n" + "="*95)
    print("ALPHA SWEEP RESULTS")
    print("="*95)
    format_str = "{:<25} | {:<10} | {:<10} | {:<10} | {:<10} | {:<12}"
    print(format_str.format("Alpha (Dense Weight)", "Recall@20", "nDCG@10", "p95 (ms)", "p99 (ms)", "Fusion (ms)"))
    print("-" * 95)
    for alpha in sorted(alpha_metrics.keys()):
        m = alpha_metrics[alpha]
        avg_recall = np.mean(m["recalls"]) if m["recalls"] else 0.0
        avg_ndcg = np.mean(m["ndcgs"]) if m["ndcgs"] else 0.0
        p95 = np.percentile(m["latencies"], 95) * 1000 if m["latencies"] else 0.0
        p99 = np.percentile(m["latencies"], 99) * 1000 if m["latencies"] else 0.0
        avg_fusion = np.mean(m["fusion_latencies"]) if "fusion_latencies" in m and m["fusion_latencies"] else 0.0
        print(format_str.format(f"{alpha:.1f}", f"{avg_recall:.4f}", f"{avg_ndcg:.4f}", f"{p95:.1f}", f"{p99:.1f}", f"{avg_fusion:.4f}"))
    print("="*95)

async def main_async():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/eval_dataset.jsonl")
    parser.add_argument("--metrics-port", type=int, default=8000)
    parser.add_argument("--collection", default="arxiv_papers", help="Target collection name.")
    parser.add_argument("--alpha-sweep", action="store_true", help="Run the alpha sweep diagnostics.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of times to run the queries for robust latency measurement.")
    args = parser.parse_args()
    
    if args.alpha_sweep:
        logger.info(f"Initializing HybridRetriever for Alpha Sweep ({args.iterations} iterations)...")
        from arxiv_scholar.retrieval.retrieval import HybridRetriever
        retriever = HybridRetriever(
            collection_name=args.collection,
            qdrant_host=config.AppConfig().qdrant_host,
            qdrant_port=config.AppConfig().qdrant_port
        )
        
        await run_alpha_sweep(
            data_file=args.data,
            collection_name=args.collection,
            retriever=retriever,
            iterations=args.iterations
        )
        return

    # Start prometheus metrics server
    logger.info(f"Starting Prometheus endpoint on port {args.metrics_port}")
    start_http_server(args.metrics_port)
    
    collections = [args.collection]
    results = []
    
    for coll in collections:
        res_list = await run_evaluation(args.data, coll)
        for res in res_list:
            res["Cost_per_1k"] = calculate_cost(res["Avg_Latency_ms"])
            results.append(res)
        
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    
    format_str = "{:<25} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10} | {:<10}"
    print(format_str.format("Collection", "Recall@5", "Recall@10", "Recall@20", "nDCG@10", "p95 (ms)", "p99 (ms)", "Cost/1k"))
    print("-" * 110)
    
    for r in results:
        print(format_str.format(
            r["Collection"],
            f"{r['Recall@5']:.3f}",
            f"{r['Recall@10']:.3f}",
            f"{r['Recall@20']:.3f}",
            f"{r['nDCG@10']:.3f}",
            f"{r['Latency_p95']*1000:.1f}",
            f"{r['Latency_p99']*1000:.1f}",
            f"${r['Cost_per_1k']:.4f}"
        ))
    print("="*110)
    
    with open("data/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Sleep to allow prometheus scraper to hit the endpoint if desired
    logger.info("Benchmark complete. Serving metrics for 10 seconds...")
    await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main_async())
