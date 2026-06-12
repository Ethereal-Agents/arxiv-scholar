# ArXiv Scholar Chunk Size Analysis

**Database Target:** `arxiv_papers` (Qdrant)  
**Total Chunks Scraped:** 117,469  
**Scraping Time:** 2.53 seconds (~46,345 chunks/sec)  

## Size Distribution (Words/Chunk)
*Note: BAAI/bge-m3 has a max sequence length of 8,192 tokens (~6,000 words).*

| Metric | Value |
|--------|-------|
| Average words/chunk | 80.0 |
| Median words/chunk | 52.0 |
| Max words/chunk | 970 |
| Min words/chunk | 1 |

### Percentiles
- 5th percentile: 3.0 words
- 10th percentile: 8.0 words
- 25th percentile: 22.0 words
- **50th percentile: 52.0 words**
- 75th percentile: 102.0 words
- 90th percentile: 191.0 words
- 95th percentile: 281.0 words

### Micro-Chunk Fragmentation
- **< 20 words:** 22.6% (26,522 chunks)
- **< 50 words:** 47.9% (56,261 chunks)
- **< 100 words:** 74.2% (87,117 chunks)
- **< 200 words:** 90.5% (106,363 chunks)

## Conclusion & Optimization Strategy
The `LayoutAwareChunker` is producing excessive micro-chunks (almost half are under 50 words). These are likely isolated headers, figure captions, or single formulas.

**Recommendation:** 
Merge adjacent micro-chunks during ingestion until they reach an optimal semantic size of **~200 - 300 words**. 
This will prevent vector "dilution" for dense retrieval, preserve local context, and safely compress a standard 10,000-word academic paper from ~168 chunks down to roughly ~40-80 chunks, essentially doubling the document capacity of the existing Qdrant deployment without any hardware upgrades.
