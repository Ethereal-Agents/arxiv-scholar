import os
from dataclasses import dataclass, field

@dataclass
class AppConfig:
    # LLM configurations
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "claude-haiku-4-5"))
    
    # Embedding configurations
    embedding_backend: str = field(default_factory=lambda: os.getenv("EMBEDDING_BACKEND", "fastembed"))
    embedding_model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    sparse_embedding_model: str = field(default_factory=lambda: os.getenv("SPARSE_EMBEDDING_MODEL", "Qdrant/bm25"))
    embedding_batch_size: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))
    embedding_device: str = field(default_factory=lambda: os.getenv("EMBEDDING_DEVICE", "auto"))

    # Qdrant storage configuration
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", "https://7ef78171-6709-42fb-a4a9-8c4809afbdb0.eu-central-1-0.aws.cloud.qdrant.io"))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))
    qdrant_host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "Arxiv-Scholar"))
    qdrant_timeout: float = field(default_factory=lambda: float(os.getenv("QDRANT_TIMEOUT", "60.0")))

    # Retrieval & Reranker Configuration
    use_reranker: bool = field(default_factory=lambda: os.getenv("USE_RERANKER", "False").lower() == "true")
    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "jina-reranker-v1-tiny-en"))
    reranker_truncation_length: int = field(default_factory=lambda: int(os.getenv("RERANKER_TRUNCATION_LENGTH", "8192")))
    reranker_fetch_multiplier: int = field(default_factory=lambda: int(os.getenv("RERANKER_FETCH_MULTIPLIER", "4")))
    
    # Hybrid fusion weights
    dense_weight: float = field(default_factory=lambda: float(os.getenv("DENSE_WEIGHT", "1.0")))
    sparse_weight: float = field(default_factory=lambda: float(os.getenv("SPARSE_WEIGHT", "0.3")))

    # Paths
    download_dir: str = field(default_factory=lambda: os.getenv("DOWNLOAD_DIR", "data/papers"))
    state_file: str = field(default_factory=lambda: os.getenv("STATE_FILE", "data/pipeline_state.json"))
