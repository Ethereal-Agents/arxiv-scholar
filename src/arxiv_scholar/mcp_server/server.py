from typing import Optional, List, Dict, Any
from mcp.server.fastmcp import FastMCP
import logging

logger = logging.getLogger(__name__)

mcp = FastMCP("ArxivScholar")

# Simple dependency injection mechanism to avoid circular imports
_orchestrator_getter = None

def set_orchestrator_getter(getter):
    global _orchestrator_getter
    _orchestrator_getter = getter

def get_orchestrator():
    if _orchestrator_getter is None:
        raise RuntimeError("Orchestrator not initialized")
    orchestrator = _orchestrator_getter()
    if orchestrator is None:
        raise RuntimeError("Orchestrator is None")
    return orchestrator

@mcp.tool()
async def search_papers(query: str, limit: int = 5, use_reranker: bool = False) -> str:
    """
    Performs a hybrid search (dense and sparse) over the arXiv corpus to find relevant academic research papers. 
    Use this tool when you need to discover papers, retrieve raw excerpts, or find literature on a specific topic.

    Args:
        query: The natural language search query or topic to look up.
        limit: The maximum number of relevant paper chunks to return (default: 5, max recommended: 20).
        use_reranker: Set to True for higher accuracy sorting using a cross-encoder, at the cost of higher latency (default: False).
        
    Returns:
        A formatted string containing the top relevant document chunks, their arXiv IDs, paper titles, and relevance scores.
    """
    try:
        orchestrator = get_orchestrator()
        chunks = await orchestrator.retrieve(query, limit=limit, use_reranker=use_reranker)
        
        if not chunks:
            return "No papers found matching the query."
            
        results = []
        for i, chunk in enumerate(chunks):
            arxiv_id = chunk["metadata"].get("arxiv_id", "Unknown")
            title = chunk["metadata"].get("title", "Unknown Title")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")
            
            results.append(f"Result {i+1} [Score: {score:.3f}]\nArXiv ID: {arxiv_id}\nTitle: {title}\nContent:\n{text}\n")
            
        return "\n---\n".join(results)
    except Exception as e:
        logger.error(f"Error in search_papers tool: {e}", exc_info=True)
        return f"Error executing search: {str(e)}"

@mcp.tool()
async def synthesize_answer(query: str, limit: int = 5) -> str:
    """
    End-to-end Retrieval-Augmented Generation (RAG) tool. 
    Use this tool when you want a direct, comprehensive, and synthesized answer to a specific academic question. 
    It automatically retrieves context from the vector database and uses an internal LLM to write a detailed answer with citations.

    Args:
        query: The specific question to be answered based on the arXiv literature.
        limit: The number of context chunks to retrieve to base the answer on (default: 5).
        
    Returns:
        A synthesized string answering the query, including inline citations to the source arXiv papers.
    """
    try:
        orchestrator = get_orchestrator()
        chunks = await orchestrator.retrieve(query, limit=limit)
        
        if not chunks:
            return "I could not find any matching papers in the database for your query."
            
        context_blocks = []
        for i, chunk in enumerate(chunks):
            arxiv_id = chunk["metadata"].get("arxiv_id")
            url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else f"Unknown Source {i+1}"
            context_blocks.append(f"Context {i+1} (Source: {url}):\n{chunk['text']}")
            
        context_str = "\n\n".join(context_blocks)
        
        llm_service = getattr(orchestrator, "llm_service", None)
        if not llm_service or not llm_service.client:
            return "LLM service is not configured. Cannot synthesize answer."
            
        # Use stream_synthesis and accumulate
        stream = llm_service.stream_synthesis(query, context_str)
        
        full_response = []
        async for token in stream:
            if token:
                full_response.append(token)
                
        return "".join(full_response)
    except Exception as e:
        logger.error(f"Error in synthesize_answer tool: {e}", exc_info=True)
        return f"Error synthesizing answer: {str(e)}"

@mcp.resource("arxiv://system-stats")
def get_system_stats() -> str:
    """Returns current status of the MCP server."""
    return "ArXiv Scholar MCP Server is running and connected to the main Orchestrator instance."

if __name__ == "__main__":
    from configs.config import AppConfig
    from arxiv_scholar.retrieval.orchestrator import Orchestrator

    # Initialize the orchestrator for local stdio mode (e.g. for MCP Inspector)
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
    set_orchestrator_getter(lambda: orchestrator)
    
    # Run the server using standard I/O
    mcp.run()
