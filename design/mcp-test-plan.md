# MCP Server Test Plan

This test plan aims to achieve 100% coverage of the newly created `src/arxiv_scholar/mcp_server/server.py` module, ensuring correct behavior and minimizing redundancy.

## Target Module: `src/arxiv_scholar/mcp_server/server.py`

### Category 1: Dependency Injection (`get_orchestrator`, `set_orchestrator_getter`)
- [ ] **Test 1.1**: `get_orchestrator` raises `RuntimeError` when the getter is entirely unset (e.g., initialized as `None`).
- [ ] **Test 1.2**: `get_orchestrator` raises `RuntimeError` when the getter is set but returns `None` (e.g., orchestrator failed to initialize).
- [ ] **Test 1.3**: `get_orchestrator` returns the valid orchestrator mock when correctly set.

### Category 2: `search_papers` Tool
- [ ] **Test 2.1**: **Success with results** - Mock `orchestrator.retrieve` to return a list of dictionaries simulating chunks (with `arxiv_id`, `title`, `score`, and `text`). Verify the returned formatted string correctly concatenates these details.
- [ ] **Test 2.2**: **Success with no results** - Mock `orchestrator.retrieve` to return an empty list. Verify it returns the specific `"No papers found matching the query."` message.
- [ ] **Test 2.3**: **Exception handling** - Mock `orchestrator.retrieve` to raise a generic `Exception`. Verify the tool catches it safely, logs the error, and returns an error string formatted as `"Error executing search: <exception>"`.

### Category 3: `synthesize_answer` Tool
- [ ] **Test 3.1**: **Success** - Mock `orchestrator.retrieve` to return valid chunks. Mock `orchestrator.llm_service.client` to exist, and mock `stream_synthesis` as an asynchronous generator yielding tokens. Verify the final returned string perfectly joins the tokens.
- [ ] **Test 3.2**: **No chunks found** - Mock `orchestrator.retrieve` to return an empty list. Verify it returns `"I could not find any matching papers in the database for your query."` without calling the LLM.
- [ ] **Test 3.3**: **Missing LLM Service** - Mock `orchestrator.retrieve` to return chunks, but set `orchestrator.llm_service` (or `.client`) to `None`. Verify it returns `"LLM service is not configured. Cannot synthesize answer."`.
- [ ] **Test 3.4**: **Exception handling** - Mock `orchestrator.retrieve` to raise an `Exception`. Verify the tool catches it and returns `"Error synthesizing answer: <exception>"`.

### Category 4: `get_system_stats` Resource
- [ ] **Test 4.1**: Call `get_system_stats()` directly and assert that it returns the expected constant string representing the server stats.

### Execution Plan
1. Create `tests/mcp_server/test_server.py`.
2. Use `pytest` and `unittest.mock` (specifically `AsyncMock` and `MagicMock`) to inject test data.
3. Install `pytest-asyncio` and `pytest-cov` if needed.
4. Run `pytest tests/mcp_server/test_server.py --cov=src/arxiv_scholar/mcp_server/server` and verify 100% line/branch coverage is achieved.
