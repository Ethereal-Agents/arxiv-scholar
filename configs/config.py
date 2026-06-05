import os
from pathlib import Path

# Base directory of the repository (arxiv-scholar)
BASE_DIR = Path(__file__).resolve().parent.parent

# Centralized configurations for the download/ingestion module
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", str(BASE_DIR / "arxiv_batch"))
STATE_FILE = os.getenv("STATE_FILE", "ingestion_state.json")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "arxiv-dataset")
GCS_BASE_PREFIX = os.getenv("GCS_BASE_PREFIX", "arxiv/arxiv/pdf/")

# Embedding configuration
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "sentence-transformers")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
SPARSE_EMBEDDING_MODEL = os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25")
USE_RERANKER = os.getenv("USE_RERANKER", "False").lower() in ("true", "1", "t")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
RERANKER_TRUNCATION_LENGTH = int(os.getenv("RERANKER_TRUNCATION_LENGTH", "2000"))
RERANKER_FETCH_MULTIPLIER = int(os.getenv("RERANKER_FETCH_MULTIPLIER", "5"))
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")

# Qdrant storage configuration
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "arxiv_papers_m3")

# Hybrid Search Fusion Weights
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.6"))
SPARSE_WEIGHT = float(os.getenv("SPARSE_WEIGHT", "0.4"))
