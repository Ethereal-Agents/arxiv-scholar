import os
import time
import logging
from statistics import mean
from pathlib import Path
from arxiv_scholar.ingestion.local import LocalDirectoryReader
from arxiv_scholar.chunking.layout import LayoutAwareChunker
from arxiv_scholar.embedding.fastembed_embedder import FastEmbedEmbedder, SparseBM25Embedder
from arxiv_scholar.storage.qdrant_store import QdrantVectorStore
from configs import config

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("benchmark")
logger.setLevel(logging.INFO)

def run_benchmark():
    download_dir = "trial_batch"
    os.environ["DOWNLOAD_DIR"] = download_dir
    os.environ["STATE_FILE"] = "benchmark_state.json"
    config.DOWNLOAD_DIR = download_dir
    config.STATE_FILE = "benchmark_state.json"
    
    if not os.path.exists(download_dir) or not os.listdir(download_dir):
        logger.info("Downloading a test PDF for benchmarking...")
        from arxiv_scholar.download.arxiv_ingestion import ArxivUnifiedEngine
        engine = ArxivUnifiedEngine()
        engine.get_batch(batch_size=1)
        
    # Read documents
    reader = LocalDirectoryReader(directory_path=download_dir)
    documents = list(reader.read())
    
    if not documents:
        logger.error("No documents found to benchmark.")
        return
        
    doc = documents[0]
    logger.info(f"Benchmarking using document: {doc.metadata.get('title', doc.id)} (size: {doc.metadata.get('file_size_bytes')} bytes)")

    models_to_test = ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"]
    chunk_sizes_to_test = [500, 1000, 1500, 2500, 4000]
    
    # Initialize Qdrant
    collection_name = "benchmark_chunk_sizes"
    store = QdrantVectorStore(collection_name=collection_name, host="localhost", port=6333)
    
    # We will just test with the first model's dimension for upsert benchmarking since both create vectors
    # But Qdrant requires a fixed dimension per collection. We'll drop and recreate it for each model if needed,
    # or just use in-memory collection. Since we just want upsert time, we can recreate it.
    
    for model_name in models_to_test:
        print("\n" + "="*85)
        print(f" MODEL: {model_name}")
        print("="*85)
        print(f"{'Max Chunk Size':<15} | {'Num Chunks':<12} | {'Avg Len (chars)':<15} | {'Embed Time (s)':<15} | {'Upsert Time (s)':<15}")
        print("-" * 85)
        
        dense_embedder = FastEmbedEmbedder(model_name=model_name, batch_size=4)
        sparse_embedder = SparseBM25Embedder(batch_size=4)
        
        # Recreate collection for the correct dimension
        store._client.delete_collection(collection_name)
        store.ensure_collection(dimension=dense_embedder.dimension)

        for size in chunk_sizes_to_test:
            chunker = LayoutAwareChunker(max_chunk_size=size)
            
            # 1. Chunking
            chunks = list(chunker.chunk(doc))
            if not chunks:
                continue
                
            avg_len = mean(len(c.content) for c in chunks)
            texts = [c.content for c in chunks]
            
            # 2. Embedding Time
            start_time = time.time()
            dense_vecs = dense_embedder.embed(texts)
            sparse_vecs = sparse_embedder.embed(texts)
            embed_time = time.time() - start_time
            
            # 3. Upsert Time
            start_time = time.time()
            store.upsert(chunks, dense_vecs, sparse_vecs)
            upsert_time = time.time() - start_time
            
            print(f"{size:<15} | {len(chunks):<12} | {avg_len:<15.1f} | {embed_time:<15.3f} | {upsert_time:<15.3f}")

    print("="*85)

if __name__ == "__main__":
    run_benchmark()
