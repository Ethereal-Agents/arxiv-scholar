import pytest
import numpy as np
import os
import sys

# Add scripts directory to path to import from run_judged_benchmarks
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.run_judged_benchmarks import calculate_ndcg_graded, rate_chunk

# Dummy Mock LLM Service for testing chunk rating logic
class MockLLMService:
    def __init__(self, expected_output: str, should_fail: bool = False):
        self.expected_output = expected_output
        self.should_fail = should_fail
        
    async def _call_llm(self, prompt, max_tokens, temperature):
        if self.should_fail:
            raise Exception("Mocked API failure")
        return self.expected_output

def test_calculate_ndcg_graded_perfect_ranking():
    # Perfect ranking: [2, 2, 1, 1, 0, 0]
    grades = [2, 2, 1, 1, 0, 0]
    # The sorted ideal ranking is identical
    ndcg = calculate_ndcg_graded(grades, k=6)
    assert np.isclose(ndcg, 1.0), f"Expected perfect nDCG of 1.0, got {ndcg}"

def test_calculate_ndcg_graded_worst_ranking():
    # Worst ranking: best matches are at the bottom
    grades = [0, 0, 1, 1, 2, 2]
    ndcg = calculate_ndcg_graded(grades, k=6)
    assert ndcg < 1.0, "Expected nDCG < 1.0 for reversed ranking"
    
def test_calculate_ndcg_graded_all_zeros():
    grades = [0, 0, 0, 0, 0]
    ndcg = calculate_ndcg_graded(grades, k=5)
    assert ndcg == 0.0, f"Expected nDCG of 0.0 for all zeros, got {ndcg}"

def test_calculate_ndcg_graded_k_cutoff():
    # Grades: 10 retrieved items
    grades = [2, 0, 0, 0, 0, 2, 2, 2, 2, 2]
    # At k=2, actual retrieved top 2 are [2, 0]. DCG = 2.0
    # The ideal top 2 out of all 10 grades is [2, 2]. IDCG = 2.0 + (2 / log2(3)) = 3.2618
    # Expected nDCG = 2.0 / 3.2618 ≈ 0.613
    ndcg = calculate_ndcg_graded(grades, k=2)
    assert np.isclose(ndcg, 0.613147, atol=1e-5), f"Expected nDCG of ~0.613 at k=2, got {ndcg}"

@pytest.mark.asyncio
async def test_rate_chunk_exact_match():
    llm = MockLLMService("2")
    grade = await rate_chunk(llm, "test query", "test chunk")
    assert grade == 2

@pytest.mark.asyncio
async def test_rate_chunk_verbose_output():
    # Models sometimes return conversational filler
    llm = MockLLMService("Based on the provided chunk, I would rate the relevance as 1.")
    grade = await rate_chunk(llm, "test query", "test chunk")
    assert grade == 1

@pytest.mark.asyncio
async def test_rate_chunk_invalid_output():
    llm = MockLLMService("The chunk is very interesting but there are no numbers here.")
    grade = await rate_chunk(llm, "test query", "test chunk")
    assert grade == 0, "Expected fallback to 0 when parsing fails"

@pytest.mark.asyncio
async def test_rate_chunk_api_failure():
    llm = MockLLMService("", should_fail=True)
    grade = await rate_chunk(llm, "test query", "test chunk")
    assert grade == 0, "Expected fallback to 0 on API failure"
