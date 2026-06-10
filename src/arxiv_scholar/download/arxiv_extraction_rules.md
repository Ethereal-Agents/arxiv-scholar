# ArXiv Subset Extraction Rules: AI Engineering & Systems Corpus

To build a high-signal, zero-noise corpus optimized specifically for AI Engineering, Agent Development, and Inference Infrastructure, we applied the following strict filtering pipeline to the 3M+ paper arXiv snapshot.

## 1. Baseline Requirements (Must meet ALL)
*   **Date Limit:** Published or updated on or after **Jan 1, 2022**.
*   **Categories:** Must belong to at least one of the following:
    *   `cs.AI` (Artificial Intelligence)
    *   `cs.CL` (Computation and Language / NLP)
    *   `cs.IR` (Information Retrieval)
    *   `cs.LG` (Machine Learning)
    *   `cs.SE` (Software Engineering)

## 2. The Anti-Noise Filter (Exclusion Domains)
Any paper whose title or abstract contains terms from pure theory, hard sciences, or unrelated physical domains is **instantly rejected**, regardless of keyword matches.
*   **Medical/Bio:** `biomedical`, `clinical`, `radiology`, `patient`, `genomic`, `epidemiology`
*   **Physical Sciences:** `quantum`, `physics`, `fluid dynamics`, `climate change`, `materials science`, `geology`, `petroleum`
*   **Physical Robotics/Civil:** `kinematics`, `manipulator arm`, `quadruped`, `autonomous driving`, `urban planning`
*   **Pure Math Theory:** `Lipschitz`, `Rademacher`, `asymptotic convergence`

## 3. Inclusion Criteria (Must meet ONE of the following)
If a paper survives the baseline and exclusion filters, its Title + Abstract is evaluated against two inclusion paths:

### Path A: The "Golden Term" Bypass (VIP Access)
The paper explicitly mentions highly specific, modern AI infrastructure tools. If ANY of these exact terms are found, the paper is auto-included:
> `vLLM`, `SWE-bench`, `FlashAttention`, `TensorRT-LLM`, `DSPy`, `LangChain`, `LlamaIndex`, `HumanEval`, `OpenHands`, `SWE-agent`, `DeepSpeed`

### Path B: Multi-Dimensional Keyword Density
The paper is mapped against 6 distinct AI Engineering keyword groups. It must successfully match terms in **at least 3 different groups** to be included. The 6 groups are:
1.  **RAG & Retrieval:** `hybrid search`, `vector database`, `reranking`, `dense retrieval`, etc.
2.  **Large Language Models:** `LLM`, `foundation model`, `GPT-4`, etc.
3.  **Agents & Reasoning:** `multi-agent`, `chain of thought`, `tool use`, `ReAct`, etc.
4.  **Training & Alignment:** `RLHF`, `DPO`, `instruction tuning`, `LoRA`, `PEFT`, etc.
5.  **Safety & Quality:** `hallucination`, `factuality`, `knowledge graph`, etc.
6.  **Inference & Systems:** `KV cache`, `speculative decoding`, `quantization`, `inference latency`, `FSDP`, etc.

## 4. Budget Capping & Sorting
To stay strictly within the **5,600 paper budget** for our vector DB tier, the resulting papers were perfectly sorted by value, and only the top 5,600 were kept:
1.  **Tier 1:** Golden Term matches
2.  **Tier 2:** 5-Group matches → 4-Group matches → 3-Group matches
3.  **Tie-breaker:** Most recently updated first.

---

*The final output is `arxiv_manifest_5600.json`, containing 5,600 pure AI systems and engineering papers, with exact GCS paths included for direct, auth-free download.*
