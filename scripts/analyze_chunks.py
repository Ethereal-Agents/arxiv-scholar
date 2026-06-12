import time
import numpy as np
from qdrant_client import QdrantClient

# Update these if your qdrant instance is remote
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "arxiv_papers"
TARGET_CHUNKS = float('inf')

def main():
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collection_info = client.get_collection(COLLECTION_NAME)
        total_points = collection_info.points_count
        print(f"Total chunks in '{COLLECTION_NAME}': {total_points:,}")
    except Exception as e:
        print(f"Failed to connect or retrieve collection: {e}")
        return

    print(f"Starting scroll to fetch all {total_points:,} chunks...")
    start_time = time.time()
    
    chunks_text = []
    next_offset = None
    batch_size = 5000

    while len(chunks_text) < TARGET_CHUNKS:
        records, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=batch_size,
            with_payload=True,
            with_vectors=False,
            offset=next_offset
        )
        
        for record in records:
            content = record.payload.get("content", "")
            chunks_text.append(content)
            if len(chunks_text) >= TARGET_CHUNKS:
                break
                
        if next_offset is None:
            print("Reached end of collection before hitting target.")
            break

    fetch_time = time.time() - start_time
    print(f"Successfully fetched {len(chunks_text):,} chunks in {fetch_time:.2f} seconds!")
    print(f"Scraping Speed: {len(chunks_text)/fetch_time:,.0f} chunks/sec")

    # Analysis
    print("\n--- Chunk Size Analysis (Words) ---")
    word_counts = [len(text.split()) for text in chunks_text]
    
    if not word_counts:
        print("No data to analyze.")
        return

    print(f"Average words/chunk: {np.mean(word_counts):.1f}")
    print(f"Median words/chunk:  {np.median(word_counts):.1f}")
    print(f"Max words/chunk:     {np.max(word_counts)}")
    print(f"Min words/chunk:     {np.min(word_counts)}")
    
    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    p_values = np.percentile(word_counts, percentiles)
    for p, v in zip(percentiles, p_values):
        print(f"{p}th percentile: {v:.1f} words")
        
    # Thresholds
    thresholds = [20, 50, 100, 200]
    for t in thresholds:
        count_below = sum(1 for w in word_counts if w < t)
        pct_below = (count_below / len(word_counts)) * 100
        print(f"% chunks < {t} words: {pct_below:.1f}% ({count_below:,} chunks)")

if __name__ == "__main__":
    main()
