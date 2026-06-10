import os
import sys
import json
import shutil
import logging
import argparse
import subprocess
from google.cloud import storage
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration Defaults
DEFAULT_MANIFEST = "arxiv_manifest_5600.json"
GCS_BUCKET_NAME = "arxiv-dataset"
TRIAL_PDF_DIR = "trial_pdfs"
OUTPUT_JSONL = "data/embedded_dataset_m3.jsonl"
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/arxiv_embeddings"

def run_batch(start_paper: int, batch_size: int, manifest_path: str):
    logger.info(f"Loading manifest from {manifest_path}")
    if not os.path.exists(manifest_path):
        logger.error(f"Manifest not found at {manifest_path}")
        sys.exit(1)
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    start_idx = start_paper
    end_idx = min(start_idx + batch_size, len(manifest))
    batch_papers = manifest[start_idx:end_idx]
    
    logger.info("="*50)
    logger.info(f"Starting Batch (Processing papers {start_idx} to {end_idx})")
    logger.info("="*50)
    
    if not batch_papers:
        logger.warning(f"No papers found starting at {start_paper}. Total manifest size: {len(manifest)}")
        return

    # Cleanup local PDF dir
    if os.path.exists(TRIAL_PDF_DIR):
        logger.info(f"Cleaning up existing {TRIAL_PDF_DIR} directory...")
        shutil.rmtree(TRIAL_PDF_DIR)
    os.makedirs(TRIAL_PDF_DIR, exist_ok=True)
    
    if os.path.exists(OUTPUT_JSONL):
        logger.info(f"Removing old output file {OUTPUT_JSONL}...")
        os.remove(OUTPUT_JSONL)
        
    os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
    
    # Download PDFs from GCS
    logger.info("Connecting to GCS anonymously...")
    try:
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(GCS_BUCKET_NAME)
    except Exception as e:
        logger.error(f"Failed to connect to GCS: {e}")
        sys.exit(1)
    
    success_count = 0
    logger.info(f"Downloading {len(batch_papers)} PDFs...")
    for paper in tqdm(batch_papers, desc="Downloading PDFs"):
        gcs_path = paper.get("gcs_path")
        if not gcs_path: 
            logger.warning(f"Paper {paper.get('id')} has no gcs_path. Skipping.")
            continue
        
        filename = gcs_path.split("/")[-1]
        
        # Manifest paths might be missing the version number (e.g., v1, v2).
        # We search by prefix to find the exact file name in the bucket.
        prefix = gcs_path.replace('.pdf', '')
        blobs = list(bucket.list_blobs(prefix=prefix))
        
        if not blobs:
            logger.error(f"Failed to find any version of {filename} in GCS.")
            continue
            
        # Grab the latest version if multiple exist
        blob = sorted(blobs, key=lambda b: b.name)[-1]
        
        local_filename = blob.name.split("/")[-1]
        local_path = os.path.join(TRIAL_PDF_DIR, local_filename)
        
        try:
            blob.download_to_filename(local_path)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")
            
    logger.info(f"Successfully downloaded {success_count}/{len(batch_papers)} PDFs.")
    if success_count == 0:
        logger.error("No PDFs downloaded. Aborting pipeline for this batch.")
        sys.exit(1)
        
    # Run embedding script
    logger.info("Executing Document Embedding Pipeline...")
    drive_file = os.path.join(DRIVE_OUTPUT_DIR, f"embedded_dataset_m3_start_{start_paper}.jsonl")
    cmd = [
        sys.executable, "colab/generate_embedded_dataset.py",
        "--pdf-dir", TRIAL_PDF_DIR,
        "--output", OUTPUT_JSONL,
        "--embedding-batch-size", "128",
        "--colab-gpu",
        "--checkpoint-path", drive_file,
        "--checkpoint-interval", "50"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Embedding pipeline failed with exit code {e.returncode}")
        sys.exit(1)
    
    # Move to Google Drive
    drive_file = os.path.join(DRIVE_OUTPUT_DIR, f"embedded_dataset_m3_start_{start_paper}.jsonl")
    logger.info(f"Copying JSONL output to Google Drive: {drive_file}")
    try:
        shutil.copy2(OUTPUT_JSONL, drive_file)
    except Exception as e:
        logger.error(f"Failed to copy to Google Drive: {e}")
        sys.exit(1)
    
    # Final Cleanup
    logger.info("Cleaning up temporary local files...")
    shutil.rmtree(TRIAL_PDF_DIR)
    if os.path.exists(OUTPUT_JSONL):
        os.remove(OUTPUT_JSONL)
        
    logger.info(f"Batch from {start_paper} Complete! File saved to {drive_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GCS download and embedding pipeline for a batch of PDFs.")
    parser.add_argument("--start-paper", type=int, required=True, help="Index of the starting paper (e.g., 0, 400, 800...)")
    parser.add_argument("--batch-size", type=int, default=400, help="Number of papers per batch (default 400)")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help="Path to manifest JSON file")
    
    args = parser.parse_args()
    run_batch(args.start_paper, args.batch_size, args.manifest)
