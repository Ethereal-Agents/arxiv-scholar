# Alpha Sweep Experiment Results

## What is an Alpha Sweep?
In hybrid search, we retrieve two independent sets of candidates for a given query:
1. **Dense Vectors** (Semantic similarity, e.g., using `BAAI/bge-m3`)
2. **Sparse Vectors** (Keyword matching, e.g., using `BM25`)

Because these two scoring mechanisms have entirely different scales, they cannot be directly added together. To fuse them manually (bypassing algorithms like Reciprocal Rank Fusion), we:
1. Independently fetch the top `K` candidates from both the dense and sparse representations.
2. Apply **Min-Max Normalization** to bound the scores of both result sets between `0.0` and `1.0`.
3. Compute a fused score using a weighting factor, **Alpha ($\alpha$)**:
   $$\text{Final Score} = (\alpha \times \text{Dense Score}) + ((1 - \alpha) \times \text{Sparse Score})$$

An **Alpha Sweep** is an empirical diagnostic test where we iterate $\alpha$ from `0.0` (100% Sparse) to `1.0` (100% Dense) in `0.1` increments. We measure the retrieval accuracy (e.g., `Recall@20`) at each increment to find the optimal fusion weight for our specific dataset and model combination.

---

## Experiment 1: Baseline Sweep (Limit = 100)
- **Dense Model**: `BAAI/bge-m3`
- **Sparse Model**: `Qdrant/bm25`
- **Re-Ranker**: None
- **Fetch Limit**: 100

| Alpha (Dense Weight) | Recall@20 |
| :------------------- | :-------- |
| 0.0                  | 0.6463    |
| 0.1                  | 0.6585    |
| 0.2                  | 0.7439    |
| 0.3                  | 0.7683    |
| 0.4                  | 0.8049    |
| 0.5                  | 0.8171    |
| **0.6**              | **0.8293**|
| 0.7                  | 0.8293    |
| 0.8                  | 0.8171    |
| 0.9                  | 0.8171    |
| 1.0                  | 0.8171    |

**Observation**: The optimal balance is heavily weighted toward the dense model ($\alpha = 0.6$), peaking at a Recall@20 of **0.8293**.

---

## Experiment 2: Increased Fetch Limit (Limit = 200)
- **Fetch Limit**: 200
- **Re-Ranker**: None

| Alpha (Dense Weight) | Recall@20 |
| :------------------- | :-------- |
| 0.0                  | 0.6463    |
| 0.1                  | 0.6829    |
| 0.2                  | 0.7195    |
| 0.3                  | 0.7683    |
| 0.4                  | 0.8049    |
| 0.5                  | 0.8171    |
| **0.6**              | **0.8293**|
| 0.7                  | 0.8171    |
| 0.8                  | 0.8171    |
| 0.9                  | 0.8171    |
| 1.0                  | 0.8171    |

**Observation**: Doubling the candidate pool did not surface any new positive chunks into the top 20. The maximum recall remained exactly the same.

---

## Experiment 3: Extreme Fetch Limit (Limit = 300)
- **Fetch Limit**: 300
- **Re-Ranker**: None

| Alpha (Dense Weight) | Recall@20 |
| :------------------- | :-------- |
| 0.0                  | 0.6463    |
| 0.1                  | 0.6829    |
| 0.2                  | 0.7195    |
| 0.3                  | 0.7317    |
| 0.4                  | 0.7927    |
| 0.5                  | 0.8171    |
| **0.6**              | **0.8293**|
| 0.7                  | 0.8171    |
| 0.8                  | 0.8171    |
| 0.9                  | 0.8171    |
| 1.0                  | 0.8171    |

**Observation**: The ceiling remains at **0.8293**. In fact, lower alphas ($\alpha = 0.3$, $0.4$) saw degraded performance because fetching 300 documents introduced a long tail of low-scoring sparse results, which stretched the min-max normalization boundaries and slightly distorted the fusion scores.

---

## Experiment 4: Alpha Sweep + Cross-Encoder Reranking
- **Fetch Limit**: 100
- **Re-Ranker**: `jinaai/jina-reranker-v1-tiny-en` (Applied to Top 100 fusion candidates)

| Alpha (Dense Weight) | Recall@20 |
| :------------------- | :-------- |
| 0.0                  | 0.7195    |
| 0.1                  | 0.7927    |
| 0.2                  | 0.7927    |
| **0.3 - 0.9**        | **0.8171**|
| 1.0                  | 0.7927    |

**Observation**: 
Applying the `jina-reranker-v1-tiny-en` cross-encoder to the hybrid candidates actually **decreased** the peak Recall@20 from `0.8293` to `0.8171`. 

Because `BAAI/bge-m3` is a highly advanced 1024-dimensional model, and the chosen Jina reranker is a "tiny" lightweight model, the reranker acted as a bottleneck, downgrading the highly accurate candidates fetched by M3. To break the `0.85` Recall target, a much larger cross-encoder (e.g., `BAAI/bge-reranker-v2-m3`) or query decomposition techniques are recommended over tweaking limits or tiny rerankers.
