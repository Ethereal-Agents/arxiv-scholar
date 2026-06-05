"""Generate embedded_dataset_m3.jsonl from a folder of PDFs.

Colab-ready script that runs the full pipeline:
  PDF folder → LayoutAwareChunker → BGE-M3 dense + BM25 sparse → JSONL

Usage (Colab):
    !git clone -b feat/colab-rechunk-pipeline https://github.com/dubeyaayush07/arxiv-scholar.git
    %cd arxiv-scholar
    !pip install -e .
    # Upload PDFs to a folder, e.g. /content/pdfs/
    !python colab/generate_embedded_dataset.py --pdf-dir /content/pdfs --output data/embedded_dataset_m3.jsonl

The output JSONL matches the format expected by scripts/ingest_m3_embeddings.py:
    {"id": "uuid", "payload": {...}, "dense_vector": [...], "sparse_indices": [...], "sparse_values": [...]}
"""

import argparse
import json
import logging
import uuid
import sys
import os

# Ensure project root and src directories are in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_dir = os.path.join(project_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def stable_uuid(chunk_id: str) -> str:
    """Deterministic UUID-v5 from chunk_id (matches qdrant_store.py logic)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def main():
    parser = argparse.ArgumentParser(description="Generate embedded dataset JSONL from PDFs")
    parser.add_argument("--pdf-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--output", default="data/embedded_dataset_m3.jsonl", help="Output JSONL path")
    parser.add_argument("--max-chunk-size", type=int, default=1500, help="Max chunk size in chars")
    parser.add_argument("--target-chunk-size", type=int, default=1000, help="Target chunk size in chars")
    parser.add_argument("--embedding-batch-size", type=int, default=32, help="Embedding batch size")
    parser.add_argument("--append", action="store_true", help="Append to existing file instead of overwriting")
    parser.add_argument("--colab-gpu", action="store_true", help="Optimize Docling for Colab GPU (uses CUDA + 4 threads for layout detection)")
    args = parser.parse_args()

    # --- Initialize pipeline components ---
    from arxiv_scholar.ingestion.local import LocalDirectoryReader
    from arxiv_scholar.chunking.layout import LayoutAwareChunker
    from arxiv_scholar.embedding.st_embedder import SentenceTransformerEmbedder
    from arxiv_scholar.embedding.fastembed_embedder import SparseBM25Embedder

    reader = LocalDirectoryReader(directory_path=args.pdf_dir)
    chunker = LayoutAwareChunker(
        max_chunk_size=args.max_chunk_size,
        target_chunk_size=args.target_chunk_size,
    )

    # Override Docling's converter for Colab GPU acceleration
    if args.colab_gpu and chunker._is_ready:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
        from docling.datamodel.base_models import InputFormat

        gpu_pipeline_options = PdfPipelineOptions()
        gpu_pipeline_options.do_ocr = False
        gpu_pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.AUTO
        )
        chunker._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=gpu_pipeline_options)}
        )
        logger.info("Docling overridden: using GPU acceleration with 4 threads")
    dense_embedder = SentenceTransformerEmbedder(
        model_name="BAAI/bge-m3",
        batch_size=args.embedding_batch_size,
    )
    sparse_embedder = SparseBM25Embedder(batch_size=args.embedding_batch_size)

    logger.info(f"Dense embedder: {dense_embedder.model_name} (dim={dense_embedder.dimension})")
    logger.info(f"Reading PDFs from: {args.pdf_dir}")
    logger.info(f"Output: {args.output}")

    # --- Process documents ---
    mode = "a" if args.append else "w"
    total_chunks = 0
    total_docs = 0

    with open(args.output, mode) as out_f:
        for document in reader.read():
            total_docs += 1
            filename = document.metadata.get("filename", document.id[:12])
            logger.info(f"[{total_docs}] Chunking: {filename}")

            chunks = list(chunker.chunk(document))
            if not chunks:
                logger.warning(f"  No chunks produced for {filename}, skipping.")
                continue

            logger.info(f"  {len(chunks)} chunks. Embedding...")

            # Embed in one batch per document
            texts = [c.content for c in chunks]
            dense_vectors = dense_embedder.embed(texts)
            sparse_vectors = sparse_embedder.embed(texts)

            # Write each chunk as a JSONL record
            for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
                record = {
                    "id": stable_uuid(chunk.id),
                    "payload": {
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                    },
                    "dense_vector": dense_vec,
                    "sparse_indices": sparse_vec.indices.tolist(),
                    "sparse_values": sparse_vec.values.tolist(),
                }
                out_f.write(json.dumps(record) + "\n")

            total_chunks += len(chunks)
            logger.info(f"  Written {len(chunks)} records. Running total: {total_chunks}")

    logger.info(f"Done! {total_docs} documents → {total_chunks} embedded chunks → {args.output}")


if __name__ == "__main__":
    main()
