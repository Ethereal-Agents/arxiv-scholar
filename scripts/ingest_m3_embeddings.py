"""Ingest pre-computed BGE-M3 embeddings into Qdrant.

Usage:
    uv run scripts/ingest_m3_embeddings.py [--data DATA_PATH] [--host HOST] [--port PORT] [--collection COLLECTION]

This script reads the `embedded_dataset_m3.jsonl` file produced by the Colab
embedding pipeline and upserts all records into a Qdrant collection with:
  - A dense vector (1024-dim, unnamed "") for BGE-M3
  - A sparse vector ("bm25") for BM25 keyword matching

The collection is created automatically if it does not exist.
"""

import argparse
import json
import logging
import uuid

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    BinaryQuantization,
    BinaryQuantizationConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 256
DENSE_DIM = 1024


def stable_uuid(chunk_id: str) -> str:
    """Deterministic UUID-v5 from chunk_id (matches qdrant_store.py logic)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def count_lines(path: str) -> int:
    """Fast line count for progress bar."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Ingest BGE-M3 embeddings into Qdrant")
    parser.add_argument("--data", default="data/embedded_dataset_m3.jsonl", help="Path to embedded JSONL file")
    parser.add_argument("--host", default="localhost", help="Qdrant host")
    parser.add_argument("--port", type=int, default=6333, help="Qdrant port")
    parser.add_argument("--collection", default="arxiv_papers_m3", help="Target collection name")
    parser.add_argument("--quantization", choices=["none", "int8", "binary"], default="none", help="Quantization method")
    args = parser.parse_args()

    # 1. Connect to Qdrant
    client = QdrantClient(host=args.host, port=args.port)
    logger.info(f"Connected to Qdrant at {args.host}:{args.port}")

    # 2. Create collection if needed
    existing = [c.name for c in client.get_collections().collections]
    if args.collection in existing:
        logger.info(f"Collection '{args.collection}' already exists. Upserting into it.")
    else:
        # Determine quantization config
        quant_config = None
        if args.quantization == "int8":
            quant_config = ScalarQuantization(
                scalar=ScalarQuantizationConfig(
                    type=ScalarType.INT8,
                    always_ram=True
                )
            )
        elif args.quantization == "binary":
            quant_config = BinaryQuantization(
                binary=BinaryQuantizationConfig(
                    always_ram=True
                )
            )

        client.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE, on_disk=True),
            sparse_vectors_config={"bm25": SparseVectorParams()},
            quantization_config=quant_config,
        )
        logger.info(f"Created collection '{args.collection}' (dim={DENSE_DIM}, quantization={args.quantization})")

    # 3. Count total records for progress bar
    logger.info(f"Counting records in {args.data}...")
    total = count_lines(args.data)
    logger.info(f"Found {total:,} records. Starting ingestion in batches of {BATCH_SIZE}...")

    # 4. Read and upsert in batches
    batch = []
    upserted = 0

    with open(args.data, "r") as f:
        for line in tqdm(f, total=total, desc="Ingesting"):
            record = json.loads(line)

            # Build the point ID from chunk_id (same logic as qdrant_store.py)
            chunk_id = record["payload"]["chunk_id"]
            point_id = stable_uuid(chunk_id)

            # Build vector dict (dense + sparse)
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
                client.upload_points(collection_name=args.collection, points=batch)
                upserted += len(batch)
                batch = []

    # Flush remaining
    if batch:
        client.upload_points(collection_name=args.collection, points=batch)
        upserted += len(batch)

    logger.info(f"Done! Upserted {upserted:,} points into '{args.collection}'.")

    # 5. Verify
    info = client.get_collection(args.collection)
    logger.info(f"Collection '{args.collection}' now has {info.points_count:,} points.")


if __name__ == "__main__":
    main()
