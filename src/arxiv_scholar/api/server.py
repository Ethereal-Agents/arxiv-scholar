import time
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from arxiv_scholar.llm.service import LLMService

from arxiv_scholar.retrieval.orchestrator import Orchestrator
from arxiv_scholar.api.schema import (
    QueryRequest, 
    SourceNode, 
    StreamMetadataEvent, 
    StreamTokenEvent, 
    StreamDoneEvent
)
from configs.config import AppConfig

logger = logging.getLogger(__name__)

# Global state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Initializing Orchestrator with ML Router and BGE Re-ranker...")
    config = AppConfig()
    orchestrator = Orchestrator(
        collection_name=config.qdrant_collection,
        qdrant_host=config.qdrant_host,
        qdrant_port=config.qdrant_port,
        qdrant_url=config.qdrant_url,
        qdrant_api_key=config.qdrant_api_key,
        dense_model_name=config.embedding_model,
        sparse_model_name=config.sparse_embedding_model,
        reranker_model_name=config.reranker_model,
        use_reranker=config.use_reranker,
        reranker_truncation_length=config.reranker_truncation_length,
        reranker_fetch_multiplier=config.reranker_fetch_multiplier,
        llm_api_key=config.llm_api_key,
        llm_base_url=config.llm_base_url,
        llm_model=config.llm_model
    )
    
    app_state["orchestrator"] = orchestrator
    app_state["llm_service"] = orchestrator.llm_service
    
    yield
    
    # Shutdown
    app_state.clear()

app = FastAPI(title="Arxiv Scholar RAG API", lifespan=lifespan)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allows the search UI to call the API from any origin (GitHub Pages, file://, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/v1/query")
@limiter.limit("5/minute")
async def query_endpoint(request: Request, body: QueryRequest):
    logger.info(f"Received query request: query='{body.query}', limit={body.limit}, rerank={body.use_reranker}")
    start_time = time.perf_counter()
    
    orchestrator = app_state.get("orchestrator")
    llm_service = app_state.get("llm_service")
    
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Orchestrator not initialized")
        
    async def _stream_response():
        try:
            # 1. Retrieve & Re-rank
            # Orchestrator is natively async, so we await it directly
            logger.debug(f"Starting retrieval for query: '{body.query}'")
            chunks = await orchestrator.retrieve(
                body.query,
                limit=body.limit,
                use_reranker=body.use_reranker
            )
            logger.debug(f"Retrieval completed. Fetched {len(chunks)} chunks.")
            
            # 2. Contextualize
            context_blocks = []
            sources = []
            paper_urls_set = set()
            
            for i, chunk in enumerate(chunks):
                arxiv_id = chunk["metadata"].get("arxiv_id")
                url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"Unknown Source {i+1}"
                if arxiv_id:
                    paper_urls_set.add(url)
                    
                context_blocks.append(f"Context {i+1} (Source: {url}):\n{chunk['text']}")
                sources.append(
                    SourceNode(
                        chunk_id=chunk["chunk_id"],
                        text=chunk["text"],
                        score=chunk["score"],
                        metadata=chunk["metadata"]
                    )
                )
                
            context_str = "\n\n".join(context_blocks)
            paper_urls = list(paper_urls_set)
            
            # YIELD 1: Metadata Event (Sent instantly)
            meta_event = StreamMetadataEvent(sources=sources, paper_urls=paper_urls)
            yield f"data: {meta_event.model_dump_json()}\n\n"
            
            # 3. LLM Synthesis Streaming
            if llm_service and llm_service.client and context_str:
                logger.debug(f"Starting LLM stream synthesis for query: '{body.query}'")
                stream = llm_service.stream_synthesis(body.query, context_str)
                
                # YIELD 2: Token Events
                async for token in stream:
                    if token:
                        token_event = StreamTokenEvent(content=token)
                        yield f"data: {token_event.model_dump_json()}\n\n"
                        
                        
                logger.debug(f"LLM stream synthesis completed for query: '{body.query}'")
            else:
                fallback_event = StreamTokenEvent(content="I could not find any matching papers in the database for your query.")
                yield f"data: {fallback_event.model_dump_json()}\n\n"
                        
            # YIELD 3: Done Event
            latency = (time.perf_counter() - start_time) * 1000
            logger.info(f"Request completed successfully in {latency:.2f}ms")
            done_event = StreamDoneEvent(latency_ms=latency)
            yield f"data: {done_event.model_dump_json()}\n\n"
            
        except Exception as e:
            logger.error(f"Error during retrieval for query '{body.query}': {e}", exc_info=True)
            yield f"data: {{\"type\": \"error\", \"detail\": \"{str(e)}\"}}\n\n"
            
    return StreamingResponse(_stream_response(), media_type="text/event-stream")

# Static file mount MUST come after route definitions to avoid shadowing /api/* routes
_docs_dir = Path(__file__).resolve().parents[3] / "docs"
if _docs_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(_docs_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
