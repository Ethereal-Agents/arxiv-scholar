import os
import subprocess
import time
import logging
import urllib.request
import xml.etree.ElementTree as ET

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    # 1. Setup local output directory
    output_dir = "./nlp_ml_gcs_pdfs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check what we already have to allow true resuming without re-fetching API pages
    existing_files = set(os.listdir(output_dir))
    start_offset = len([f for f in existing_files if f.endswith('.pdf')])
    logger.info(f"Found {start_offset} existing PDFs. Resuming from offset {start_offset}.")

    needed = 1000 - start_offset
    if needed <= 0:
        logger.info("Already downloaded 1000 papers. Exiting.")
        return

    # Check gsutil
    gsutil_available = True
    try:
        subprocess.run(["gsutil", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        gsutil_available = False
        logger.warning("gsutil not found on system. Script will use HTTP fallback for all downloads.")

    # Fetch metadata in pages of 100 using raw API to allow start_offset (bypassing arxiv library pagination limits)
    page_size = 100
    downloaded_count = start_offset

    while downloaded_count < 1000:
        # Use raw arXiv API to jump straight to the required offset
        url = f"https://export.arxiv.org/api/query?search_query=cat:cs.CL+OR+cat:cs.LG&sortBy=submittedDate&sortOrder=descending&start={downloaded_count}&max_results={page_size}"
        logger.info(f"Fetching metadata for offset {downloaded_count} to {downloaded_count + page_size}...")
        
        api_success = False
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    xml_data = response.read()
                api_success = True
                break
            except Exception as e:
                logger.warning(f"API fetch failed (attempt {attempt+1}/5): {e}")
                time.sleep(10)
                
        if not api_success:
            logger.error("Failed to fetch metadata from arXiv API.")
            break

        # Parse XML
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', ns)
        
        if not entries:
            logger.info("No more entries found.")
            break

        for entry in entries:
            if downloaded_count >= 1000:
                break
                
            id_url = entry.find('atom:id', ns).text
            arxiv_id = id_url.split('/')[-1]
            raw_id = arxiv_id.split('v')[0]
            year_month = raw_id.split('.')[0]
            
            expected_filename = f"{arxiv_id}.pdf"
            filepath = os.path.join(output_dir, expected_filename)
            
            pdf_url = ""
            for link in entry.findall('atom:link', ns):
                if link.get('title') == 'pdf':
                    pdf_url = link.get('href')
                    break
            if not pdf_url:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

            downloaded_count += 1
            
            if os.path.exists(filepath):
                logger.info(f"[{downloaded_count}/1000] Skipping already downloaded: {arxiv_id}")
                continue
                
            logger.info(f"[{downloaded_count}/1000] Pulling: {arxiv_id}")
            
            # Resilience: Retry loop for PDF download
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    success = False
                    if gsutil_available:
                        gcs_path = f"gs://arxiv-dataset/arxiv/arxiv/pdf/{year_month}/{arxiv_id}.pdf"
                        try:
                            subprocess.run(["gsutil", "cp", gcs_path, output_dir], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            success = True
                        except Exception as gs_err:
                            logger.debug(f"gsutil failed for {arxiv_id}: {gs_err}, falling back to HTTP.")
                    
                    if not success:
                        req = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0'})
                        with urllib.request.urlopen(req, timeout=60) as response, open(filepath, 'wb') as out_file:
                            out_file.write(response.read())
                    
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Download failed for {arxiv_id}, retrying ({attempt+1}/{max_retries})... Error: {e}")
                        time.sleep(5)
                    else:
                        logger.error(f"Error downloading paper {arxiv_id}: {e}")
            
            # Polite rate limiting
            time.sleep(3.5)

    logger.info(f"\nSuccessfully gathered {downloaded_count} filtered NLP/ML PDFs into '{output_dir}'!")

if __name__ == "__main__":
    main()
