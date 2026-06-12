import argparse
import json
import logging
import uuid
import glob
import os

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

BATCH_SIZE = 256
DENSE_DIM = 1024

def stable_uuid(chunk_id: str) -> str:
    """Deterministic UUID-v5 from chunk_id."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

def count_lines(filepaths: list[str]) -> int:
    """Fast line count for progress bar."""
    count = 0
    for path in filepaths:
        with open(path, "rb") as f:
            for _ in f:
                count += 1
    return count

def main():
    parser = argparse.ArgumentParser(description="Ingest BGE-M3 embeddings from folder into Qdrant")
    parser.add_argument("--data-dir", default="/Users/tri/Downloads/arxiv_extracted_jsonl/arxiv_embeddings", help="Path to folder with jsonl files")
    parser.add_argument("--url", default="https://7ef78171-6709-42fb-a4a9-8c4809afbdb0.eu-central-1-0.aws.cloud.qdrant.io", help="Qdrant URL")
    parser.add_argument("--api-key", required=True, help="Qdrant API Key")
    parser.add_argument("--collection", default="arxiv-scholar", help="Target collection name")
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
    logger.info(f"Found {total:,} records across {len(files)} files. Starting ingestion in batches of {BATCH_SIZE}...")

    batch = []
    upserted = 0
    seen_chunk_ids = set()
    duplicate_count = 0

    with tqdm(total=total, desc="Ingesting") as pbar:
        for file in files:
            with open(file, "r") as f:
                for line in f:
                    record = json.loads(line)
                    chunk_id = record["payload"]["chunk_id"]
                    
                    # Deduplication using chunkId and existing Qdrant IDs
                    point_id = stable_uuid(chunk_id)
                    if chunk_id in seen_chunk_ids or point_id in existing_ids:
                        duplicate_count += 1
                        pbar.update(1)
                        continue
                        
                    seen_chunk_ids.add(chunk_id)

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
                            payload=record["payload"],
                        )
                    )

                    if len(batch) >= BATCH_SIZE:
                        for attempt in range(10):
                            try:
                                client.upload_points(collection_name=args.collection, points=batch)
                                break
                            except Exception as e:
                                logger.warning(f"Upload failed (attempt {attempt+1}/10): {e}")
                                import time
                                time.sleep(5 * (attempt + 1))
                        else:
                            raise RuntimeError("Failed to upload batch after 10 attempts")
                            
                        upserted += len(batch)
                        batch = []
                    
                    pbar.update(1)

    # Flush remaining
    if batch:
        for attempt in range(10):
            try:
                client.upload_points(collection_name=args.collection, points=batch)
                break
            except Exception as e:
                logger.warning(f"Upload failed (attempt {attempt+1}/10): {e}")
                import time
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError("Failed to upload final batch after 10 attempts")
        upserted += len(batch)

    logger.info(f"Done! Upserted {upserted:,} points into '{args.collection}'. Skipped {duplicate_count:,} duplicates.")
    info = client.get_collection(args.collection)
    logger.info(f"Collection '{args.collection}' now has {info.points_count:,} points.")

if __name__ == "__main__":
    main()
