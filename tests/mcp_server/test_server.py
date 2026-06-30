import pytest
from unittest.mock import AsyncMock, MagicMock

from arxiv_scholar.mcp_server.server import (
    set_orchestrator_getter,
    get_orchestrator,
    search_papers,
    synthesize_answer,
    get_system_stats
)

# --- Category 1: Dependency Injection ---

def test_get_orchestrator_not_set():
    set_orchestrator_getter(None)
    with pytest.raises(RuntimeError, match="Orchestrator not initialized"):
        get_orchestrator()

def test_get_orchestrator_returns_none():
    set_orchestrator_getter(lambda: None)
    with pytest.raises(RuntimeError, match="Orchestrator is None"):
        get_orchestrator()

def test_get_orchestrator_success():
    mock_orchestrator = MagicMock()
    set_orchestrator_getter(lambda: mock_orchestrator)
    assert get_orchestrator() is mock_orchestrator

# --- Category 2: search_papers Tool ---

@pytest.mark.asyncio
async def test_search_papers_success_with_results():
    mock_orchestrator = MagicMock()
    # retrieve is async
    mock_orchestrator.retrieve = AsyncMock(return_value=[
        {
            "metadata": {"arxiv_id": "1234.5678", "title": "Test Paper 1"},
            "score": 0.95,
            "text": "This is test paper 1."
        },
        {
            "metadata": {}, # missing some metadata to test defaults
            "score": 0.5,
            "text": "This is test paper 2."
        }
    ])
    set_orchestrator_getter(lambda: mock_orchestrator)
    
    # Unpack the underlying tool function logic (FastMCP wraps the function but we can test the unwrapped one or the wrapper depending on how FastMCP exposes it, wait, FastMCP tools might be decorators that wrap the func, but we imported `search_papers` which is the original function. No wait, `FastMCP.tool` replaces it with a wrapped one. Actually, if we just call the function normally, will it work? FastMCP allows direct invocation of tool methods sometimes, but we might need to access the underlying function. Actually, if it's an async function we can just await it.)
    # Let's call the original wrapped function directly if possible. Actually, @mcp.tool() modifies the function to register it but might return the original function. Let's assume it returns the original.
    
    result = await search_papers("test query")
    
    assert "Result 1 [Score: 0.950]" in result
    assert "ArXiv ID: 1234.5678" in result
    assert "Title: Test Paper 1" in result
    assert "This is test paper 1." in result
    
    assert "Result 2 [Score: 0.500]" in result
    assert "ArXiv ID: Unknown" in result
    assert "Title: Unknown Title" in result
    assert "This is test paper 2." in result
    assert "---" in result

@pytest.mark.asyncio
async def test_search_papers_success_no_results():
    mock_orchestrator = MagicMock()
    mock_orchestrator.retrieve = AsyncMock(return_value=[])
    set_orchestrator_getter(lambda: mock_orchestrator)
    
    result = await search_papers("test query")
    assert result == "No papers found matching the query."

@pytest.mark.asyncio
async def test_search_papers_exception():
    mock_orchestrator = MagicMock()
    mock_orchestrator.retrieve = AsyncMock(side_effect=Exception("Database error"))
    set_orchestrator_getter(lambda: mock_orchestrator)
    
    result = await search_papers("test query")
    assert "Error executing search: Database error" in result

# --- Category 3: synthesize_answer Tool ---

@pytest.mark.asyncio
async def test_synthesize_answer_success():
    mock_orchestrator = MagicMock()
    mock_orchestrator.retrieve = AsyncMock(return_value=[
        {"metadata": {"arxiv_id": "1111.2222"}, "text": "Chunk 1"},
        {"metadata": {}, "text": "Chunk 2"}
    ])
    
    mock_llm = MagicMock()
    mock_llm.client = True 
    
    async def mock_stream_synthesis(query, context):
        yield "Hello "
        yield "World!"
        yield "" 
        yield None 
        
    mock_llm.stream_synthesis = MagicMock(side_effect=mock_stream_synthesis)
    mock_orchestrator.llm_service = mock_llm
    
    set_orchestrator_getter(lambda: mock_orchestrator)
    
    result = await synthesize_answer("test query")
    assert result == "Hello World!"
    
    mock_llm.stream_synthesis.assert_called_once()
    args, kwargs = mock_llm.stream_synthesis.call_args
    assert args[0] == "test query"
    assert "Context 1 (Source: https://arxiv.org/abs/1111.2222):\nChunk 1" in args[1]
    assert "Context 2 (Source: Unknown Source 2):\nChunk 2" in args[1]

@pytest.mark.asyncio
async def test_synthesize_answer_no_results():
    mock_orchestrator = MagicMock()
    mock_orchestrator.retrieve = AsyncMock(return_value=[])
    set_orchestrator_getter(lambda: mock_orchestrator)
    
    result = await synthesize_answer("test query")
    assert result == "I could not find any matching papers in the database for your query."

@pytest.mark.asyncio
async def test_synthesize_answer_missing_llm_service():
    mock_orchestrator = MagicMock()
    mock_orchestrator.retrieve = AsyncMock(return_value=[{"metadata": {}, "text": "Chunk 1"}])
    mock_orchestrator.llm_service = None 
    set_orchestrator_getter(lambda: mock_orchestrator)
    
    result = await synthesize_answer("test query")
    assert result == "LLM service is not configured. Cannot synthesize answer."
    
    mock_llm = MagicMock()
    mock_llm.client = None
    mock_orchestrator.llm_service = mock_llm
    
    result = await synthesize_answer("test query")
    assert result == "LLM service is not configured. Cannot synthesize answer."

@pytest.mark.asyncio
async def test_synthesize_answer_exception():
    mock_orchestrator = MagicMock()
    mock_orchestrator.retrieve = AsyncMock(side_effect=Exception("LLM error"))
    set_orchestrator_getter(lambda: mock_orchestrator)
    
    result = await synthesize_answer("test query")
    assert "Error synthesizing answer: LLM error" in result

# --- Category 4: get_system_stats Resource ---

def test_get_system_stats():
    result = get_system_stats()
    assert result == "ArXiv Scholar MCP Server is running and connected to the main Orchestrator instance."
