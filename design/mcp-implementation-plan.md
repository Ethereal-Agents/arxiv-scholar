# ArXiv Scholar MCP Server Implementation Plan

## 1. Overview
The goal is to expose the capabilities of the **ArXiv Scholar** RAG pipeline via a Model Context Protocol (MCP) server. This will allow AI agents (like Claude Desktop, Cursor, or custom agents) to directly interact with the indexed AI Engineering research papers, retrieve academic sources, and execute complex queries.

## 2. Best Practices for Python MCP Servers
Based on current best practices for building MCP servers in Python:
- **Use `FastMCP`**: Leverage the `mcp` SDK's `FastMCP` abstraction. It provides FastAPI-like decorators (`@mcp.tool()`, `@mcp.resource()`) and handles the underlying JSON-RPC complexities.
- **Dependency Management**: Use `uv` (which this project already utilizes) to manage dependencies and script execution safely.
- **Transport Protocol (Streamable HTTP)**: For remote servers, use the modern **Streamable HTTP** specification rather than legacy SSE. Streamable HTTP unifies communication into a single endpoint, supports session IDs for better reliability against network drops, and plays nicer with CDNs, proxies, and firewalls.
- **Logging Safety (stdio)**: If you also support local `stdio` transport for desktop agents, **never use `print()`** or log to `stdout`, as this corrupts the JSON-RPC communication stream. All logging must be routed to `sys.stderr` or a dedicated log file.
- **Idempotency & Isolation**: Design tools to be stateless. Agents might retry operations or call them in parallel.
- **Explicit Inputs & Pydantic**: Require explicit inputs for tools. Use Pydantic models (already prevalent in this project) to strictly validate tool arguments.
- **Rich Documentation**: Use detailed docstrings for tools and resources. The LLM uses these descriptions to decide when and how to call the tool.
- **Pathing**: Ensure absolute paths are used when the server is spawned by a local host (e.g., Claude Desktop).

## 3. Proposed MCP Server Architecture

We will implement the MCP server using `FastMCP`.

### 3.1. New Dependencies
Add the official MCP Python SDK to `pyproject.toml`:
```toml
mcp>=1.0.0
```

### 3.2. Code Structure
Create a new entrypoint for the MCP server:
```
src/arxiv_scholar/mcp_server/server.py
```

### 3.3. Proposed Tools (`@mcp.tool()`)
These tools will map to the existing core logic in `retrieval/orchestrator.py` and `llm/service.py`.

1. **`search_papers(query: str, year_filter: Optional[int] = None)`**
   - **Description**: Performs a hybrid search over the arXiv corpus. Returns relevant document chunks, arXiv IDs, and relevance scores.
   - **Backend**: Calls `RetrievalOrchestrator.retrieve(query)`.

2. **`synthesize_answer(query: str)`**
   - **Description**: End-to-end RAG tool. Retrieves context from the vector database and synthesizes a comprehensive answer with citations.
   - **Backend**: Uses the same logic as the existing `/api/v1/query` FastAPI route, yielding the final synthesized string instead of SSE events.

3. **`get_paper_summary(arxiv_id: str)`**
   - **Description**: Fetches all chunks associated with a specific `arxiv_id` and uses the LLM to generate a standalone summary of the paper.

### 3.4. Proposed Resources (`@mcp.resource()`)
Resources expose read-only contextual data to the agent.
- **`arxiv://system-stats`**: Returns current stats of the index (e.g., number of indexed papers, corpus domains).

## 4. Implementation Steps

1. **Install Dependencies**: Run `uv add "mcp[cli]"` to update `pyproject.toml` and `uv.lock`.
2. **Setup Server Shell**: Create `src/arxiv_scholar/mcp_server/server.py`. Initialize `mcp = FastMCP("ArxivScholar")`.
3. **Configure Logging**: Setup a Python logger that writes exclusively to `sys.stderr`.
4. **Implement Tools**:
   - Wrap the asynchronous initialization of `RetrievalOrchestrator`. (Note: Manage the async lifecycle properly since the underlying codebase uses async Qdrant and LLM clients).
   - Implement the `search_papers` and `synthesize_answer` functions as tools.
5. **Testing**: 
   - Test locally using the MCP Inspector: `npx @modelcontextprotocol/inspector uv run src/arxiv_scholar/mcp_server/server.py`
6. **Documentation**: Update `README.md` with instructions on how to connect Claude Desktop or Cursor to the ArXiv Scholar MCP server.

## 5. Security & Configuration
- The MCP server will read the same environment variables (e.g., `QDRANT_URL`, `QDRANT_API_KEY`, `OPENAI_API_KEY`) as the main app.

### Local Agents (stdio transport)
When configuring for local agents like Claude Desktop, the config block should look like:
```json
{
  "mcpServers": {
    "arxiv-scholar": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/arxiv-scholar",
        "run",
        "src/arxiv_scholar/mcp_server/server.py"
      ],
      "env": {
        "QDRANT_URL": "...",
        "QDRANT_API_KEY": "...",
        "OPENAI_API_KEY": "..."
      }
    }
  }
}
```

### Remote Agents (Streamable HTTP)
For serving remote agents (e.g., exposing `arxiv-scholar` as an API), integrate `FastMCP` directly into your existing FastAPI application to use the unified Streamable HTTP endpoint:
```python
# src/arxiv_scholar/api/server.py
from fastapi import FastAPI
from arxiv_scholar.mcp_server.server import mcp

app = FastAPI()

# Mount the MCP server to use the modern Streamable HTTP protocol
app.mount("/mcp", mcp.get_starlette_app())
```
This creates a robust, firewall-friendly single `/mcp/messages` endpoint for remote clients to connect and stream interactions using session IDs!
