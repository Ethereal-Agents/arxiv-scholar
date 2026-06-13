import os
import argparse
import logging

from configs import config
from arxiv_scholar.ingestion.local import LocalDirectoryReader
from arxiv_scholar.chunking.layout import LayoutAwareChunker
from arxiv_scholar.chunking.sliding_window import SlidingWindowChunker
from arxiv_scholar.embedding.fastembed_embedder import FastEmbedEmbedder, SparseBM25Embedder
from arxiv_scholar.storage.qdrant_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def ingest_local_folder(input_dir: str, chunker_type: str, collection_name: str):
    logger.info(f"Starting local ingestion from: {input_dir}")
    
    # Check if folder exists
    if not os.path.isdir(input_dir):
        logger.error(f"Directory not found: {input_dir}")
        return
        
    # Setup components
    if chunker_type == "layout":
        chunker = LayoutAwareChunker(max_chunk_size=2000)
    elif chunker_type == "sliding_window":
        chunker = SlidingWindowChunker(chunk_size=2000, chunk_overlap=200)
    else:
        raise ValueError(f"Unknown chunker type: {chunker_type}")
    
    embedder = FastEmbedEmbedder(
        model_name=config.EMBEDDING_MODEL,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )
    
    sparse_embedder = SparseBM25Embedder(batch_size=config.EMBEDDING_BATCH_SIZE)
    
    store = QdrantVectorStore(
        collection_name=collection_name,
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
    )
    store.ensure_collection(dimension=embedder.dimension)
    
    # Read files
    logger.info("Discovering PDF files...")
    reader = LocalDirectoryReader(directory_path=input_dir)
    
    total_docs = 0
    total_chunks = 0
    total_embedded = 0
    
    # Process one document at a time to avoid memory overflow
    for document in reader.read():
        filename = document.metadata.get('filename', 'unknown')
        logger.info(f"Processing: {filename}")
        
        try:
            chunks = list(chunker.chunk(document))
            
            if not chunks:
                logger.warning(f"No text extracted from {filename}")
                continue
                
            texts = [chunk.content for chunk in chunks]
            
            # Embed
            dense_vectors = embedder.embed(texts)
            sparse_vectors = sparse_embedder.embed(texts)
            
            # Upsert to Qdrant
            upserted = store.upsert(chunks, dense_vectors, sparse_vectors=sparse_vectors)
            
            total_docs += 1
            total_chunks += len(chunks)
            total_embedded += len(dense_vectors)
            
            logger.info(f"  ✓ Inserted {upserted} chunks into Qdrant.")
            
        except Exception as e:
            logger.error(f"Failed processing {filename}: {e}")

    logger.info("=========================================")
    logger.info("Ingestion Complete!")
    logger.info(f"Total Documents: {total_docs}")
    logger.info(f"Total Chunks: {total_chunks}")
    logger.info("=========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest a local folder of PDFs into Qdrant.")
    parser.add_argument("--dir", type=str, default="./nlp_ml_gcs_pdfs", help="Path to local folder containing PDFs")
    parser.add_argument("--chunker", type=str, choices=["layout", "sliding_window"], default="layout", help="Chunking strategy to use")
    parser.add_argument("--collection", type=str, default="arxiv_papers", help="Qdrant collection name")
    args = parser.parse_args()
    
    ingest_local_folder(args.dir, args.chunker, args.collection)
