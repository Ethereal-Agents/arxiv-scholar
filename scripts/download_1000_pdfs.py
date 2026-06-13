import os
import time
import arxiv
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # 1. Create a dedicated folder for your NLP/ML dataset
    output_dir = "./nlp_ml_1000_pdfs"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Build a combined query targeting ONLY NLP and Machine Learning
    # cat:cs.CL = Natural Language Processing
    # cat:cs.LG = Machine Learning
    combined_query = "cat:cs.CL OR cat:cs.LG"

    search = arxiv.Search(
        query=combined_query,
        max_results=1000,
        sort_by=arxiv.SortCriterion.SubmittedDate  # Fetches the most recent papers
    )

    client = arxiv.Client()

    logger.info("Initializing download of 1,000 NLP/ML PDFs. Please wait...")

    # 3. Safely loop through and download each file
    for index, result in enumerate(client.results(search), start=1):
        try:
            # Clean the title to prevent file system naming errors
            clean_title = "".join(c for c in result.title if c.isalnum() or c in (" ", "_", "-")).rstrip()
            # Create a clean filename: [ArXiv_ID]_[Shortened_Title].pdf
            arxiv_id = result.entry_id.split('/')[-1]
            filename = f"{arxiv_id}_{clean_title[:40]}.pdf"
            filepath = os.path.join(output_dir, filename)
            
            # Auto-resume check: skip if file already exists
            if os.path.exists(filepath):
                logger.info(f"[{index}/1000] Skipping already downloaded: {result.title[:50]}...")
                continue
                
            logger.info(f"[{index}/1000] Downloading: {result.title[:50]}...")
            
            # Resilience: retry loop for network errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    result.download_pdf(dirpath=output_dir, filename=filename)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Download failed for {arxiv_id}, retrying ({attempt+1}/{max_retries})... Error: {e}")
                        time.sleep(5)
                    else:
                        raise e
            
            # Mandatory polite rate-limiting pause to prevent IP bans
            time.sleep(1.5)
            
        except Exception as e:
            logger.error(f"Error downloading paper {index} ({result.entry_id}): {e}")
            continue

    logger.info(f"\nSuccess! Processing finished. PDFs are saved to '{output_dir}'.")

if __name__ == "__main__":
    main()
