import argparse
import json
import logging
import uuid
import glob
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    SparseVectorParams,
    SparseVector,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 500
DENSE_DIM = 1024
MAX_WORKERS = 8

def extract_year_from_arxiv_id(arxiv_id: str):
    if not arxiv_id:
        return None
        
    old_format_match = re.search(r'^[a-z\-]+(?:-[a-z]+)?/(\d{2})\d{5}(?:v\d+)?$', arxiv_id, re.IGNORECASE)
    if old_format_match:
        yy = int(old_format_match.group(1))
    else:
        new_format_match = re.search(r'^(\d{2})\d{2}\.\d+', arxiv_id)
        if new_format_match:
            yy = int(new_format_match.group(1))
        else:
            return None

    if yy >= 90:
        return 1900 + yy
    else:
        return 2000 + yy

def stable_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

def count_lines(filepaths: list[str]) -> int:
    count = 0
    for path in filepaths:
        with open(path, "rb") as f:
            for _ in f:
                count += 1
    return count

def upload_batch_with_retry(client: QdrantClient, collection_name: str, batch: list[PointStruct]) -> int:
    for attempt in range(10):
        try:
            client.upload_points(collection_name=collection_name, points=batch)
            return len(batch)
        except Exception as e:
            logger.warning(f"Upload failed (attempt {attempt+1}/10): {e}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Failed to upload batch after 10 attempts")

def main():
    parser = argparse.ArgumentParser(description="Ingest BGE-M3 embeddings from folder into Qdrant in parallel")
    parser.add_argument("--data-dir", default=os.path.expanduser("~/Downloads/arxiv_extracted_jsonl/arxiv_embeddings"), help="Path to folder with jsonl files")
    parser.add_argument("--url", default="https://7ef78171-6709-42fb-a4a9-8c4809afbdb0.eu-central-1-0.aws.cloud.qdrant.io", help="Qdrant URL")
    parser.add_argument("--api-key", required=True, help="Qdrant API Key")
    parser.add_argument("--collection", default="Arxiv-Scholar", help="Target collection name")
    args = parser.parse_args()

    client = QdrantClient(url=args.url, api_key=args.api_key, timeout=120.0)
    logger.info(f"Connected to Qdrant at {args.url}")

    existing = [c.name for c in client.get_collections().collections]
    existing_ids = set()
    if args.collection in existing:
        logger.info(f"Collection '{args.collection}' already exists. Fetching existing IDs to resume efficiently...")
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=args.collection,
                limit=10000,
                with_payload=False,
                with_vectors=False,
                offset=offset
            )
            existing_ids.update(r.id for r in records)
            if offset is None:
                break
        logger.info(f"Found {len(existing_ids)} existing points in Qdrant. They will be skipped.")
    else:
        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            sparse_vectors_config={"bm25": SparseVectorParams()},
        )
        logger.info(f"Created collection '{args.collection}' (dim={DENSE_DIM}, distance=COSINE)")

    files = sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl")))
    if not files:
        logger.error(f"No .jsonl files found in {args.data_dir}")
        return

    logger.info(f"Found {len(files)} JSONL files. Counting records...")
    total = count_lines(files)
    logger.info(f"Found {total:,} records across {len(files)} files. Starting parallel ingestion in batches of {BATCH_SIZE}...")

    batch = []
    upserted = 0
    seen_chunk_ids = set()
    duplicate_count = 0

    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    futures = set()

    with tqdm(total=total, desc="Ingesting") as pbar:
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    record = json.loads(line)
                    chunk_id = record["payload"]["chunk_id"]
                    
                    point_id = stable_uuid(chunk_id)
                    if chunk_id in seen_chunk_ids or point_id in existing_ids:
                        duplicate_count += 1
                        pbar.update(1)
                        continue
                        
                    seen_chunk_ids.add(chunk_id)

                    # Extract year and inject into payload
                    payload = record["payload"]
                    arxiv_id = payload.get("metadata", {}).get("arxiv_id")
                    year = extract_year_from_arxiv_id(arxiv_id)
                    if year:
                        payload["metadata"]["year"] = year

                    point_vector = {
                        "": record["dense_vector"],
                        "bm25": SparseVector(
                            indices=record["sparse_indices"],
                            values=record["sparse_values"],
                        ),
                    }

                    batch.append(
                        PointStruct(
                            id=point_id,
                            vector=point_vector,
                            payload=payload,
                        )
                    )

                    if len(batch) >= BATCH_SIZE:
                        # Submit to thread pool
                        future = executor.submit(upload_batch_with_retry, client, args.collection, batch)
                        futures.add(future)
                        batch = []
                        
                        # To prevent memory explosion, limit the number of active futures
                        while len(futures) >= MAX_WORKERS * 2:
                            done, futures = wait_for_some(futures)
                            for d in done:
                                upserted += d.result()
                    
                    pbar.update(1)

    # Flush remaining
    if batch:
        future = executor.submit(upload_batch_with_retry, client, args.collection, batch)
        futures.add(future)

    # Wait for all futures to complete
    for future in as_completed(futures):
        upserted += future.result()

    executor.shutdown(wait=True)

    logger.info(f"Done! Upserted {upserted:,} points into '{args.collection}'. Skipped {duplicate_count:,} duplicates.")
    info = client.get_collection(args.collection)
    logger.info(f"Collection '{args.collection}' now has {info.points_count:,} points.")

def wait_for_some(futures):
    from concurrent.futures import FIRST_COMPLETED, wait
    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
    return done, not_done

if __name__ == "__main__":
    main()
