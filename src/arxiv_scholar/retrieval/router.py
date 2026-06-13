import os
import joblib
import re
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class Route(Enum):
    DIRECT = "direct"
    DECOMPOSE = "decompose"
    HYDE = "hyde"

class MLQueryRouter:
    def __init__(self, model_path: str = None):
        if model_path is None:
            # src/arxiv_scholar/retrieval/router.py -> up 3 levels to project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            model_path = os.path.join(project_root, "data", "router_dataset", "query_router_model.joblib")
            
        self.classifier = None
        if os.path.exists(model_path):
            try:
                self.classifier = joblib.load(model_path)
                logger.info(f"Successfully loaded ML router from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load ML router: {e}")
        else:
            logger.warning(f"ML router model not found at {model_path}. Falling back to heuristics.")

    def route(self, query: str, query_vector: list = None) -> Route:
        query_lower = query.lower().strip()
        words = query_lower.split()
        
        # Heuristic 1: Sparse/Short queries (<= 4 words)
        if len(words) <= 4:
            logger.info(f"Query length ({len(words)}) <= 4 words, defaulting to Route.HYDE")
            return Route.HYDE
            
        # Hard Override: Strict proximity regex for metadata extraction (Look for years)
        metadata_pattern = re.compile(r"(?:published|available|released|from|since|before|after|in)\s+(?:year\s+)?(19\d{2}|20\d{2})")
        if metadata_pattern.search(query_lower):
            logger.info("Metadata pattern matched, routing to Route.DECOMPOSE (Hard Override)")
            return Route.DECOMPOSE
            
        # ML Routing (if loaded and vector provided)
        if self.classifier is not None and query_vector is not None:
            try:
                pred = self.classifier.predict([query_vector])[0]
                if pred == 1:
                    logger.info(f"ML Router predicted class {pred} -> Route.DECOMPOSE")
                    return Route.DECOMPOSE
                else:
                    logger.info(f"ML Router predicted class {pred} -> Route.DIRECT")
                    return Route.DIRECT
            except ValueError as e:
                logger.warning(f"ML Router failed (dimension mismatch?): {e}")
            
        # Default fallback: Long, highly-descriptive queries already have enough semantic density
        logger.info("Default fallback reached, returning Route.DIRECT")
        return Route.DIRECT