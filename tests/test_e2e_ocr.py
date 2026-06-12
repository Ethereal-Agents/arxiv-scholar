import os
import glob
import logging
from configs import config

from arxiv_scholar.schema import Document
from arxiv_scholar.chunking.layout import LayoutAwareChunker
from arxiv_scholar.embedding.st_embedder import SentenceTransformerEmbedder
from arxiv_scholar.embedding.fastembed_embedder import FastEmbedEmbedder, SparseBM25Embedder
from arxiv_scholar.storage.qdrant_store import QdrantVectorStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    pdf_files = glob.glob("/Users/tri/Projects/arxiv-scholar/nlp_ml_gcs_pdfs/*.pdf")
    test_pdf = pdf_files[0]
    filename = os.path.basename(test_pdf)

    doc = Document(
        id=filename,
        content="",
        metadata={"source_path": test_pdf, "filename": filename}
    )

    logger.info("Initializing LayoutAwareChunker (OCR is disabled in code)")
    chunker = LayoutAwareChunker(max_chunk_size=2000)

    logger.info(f"Initializing Embedders (Forcing FastEmbed)")
    embedder = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5", batch_size=config.EMBEDDING_BATCH_SIZE)

    sparse_embedder = SparseBM25Embedder(batch_size=config.EMBEDDING_BATCH_SIZE)

    logger.info("Initializing Qdrant storage with collection: layout_ocr_disabled_test")
    store = QdrantVectorStore(
        collection_name="layout_ocr_disabled_test",
        host=config.QDRANT_HOST,
        port=config.QDRANT_PORT,
    )
    store.ensure_collection(dimension=embedder.dimension)

    logger.info(f"--- Chunking {test_pdf} ---")
    chunks = list(chunker.chunk(doc))
    logger.info(f"Generated {len(chunks)} chunks.")

    if chunks:
        logger.info("--- Embedding ---")
        texts = [chunk.content for chunk in chunks]
        vectors = embedder.embed(texts)
        sparse_vectors = sparse_embedder.embed(texts)
        
        logger.info("--- Upserting ---")
        upserted = store.upsert(chunks, vectors, sparse_vectors=sparse_vectors)
        logger.info(f"Upserted {upserted} points.")
        
        # Verify count
        count = store.client.count(collection_name="layout_ocr_disabled_test").count
        logger.info(f"✅ Verified: Collection 'layout_ocr_disabled_test' now has {count} points.")

if __name__ == "__main__":
    main()
