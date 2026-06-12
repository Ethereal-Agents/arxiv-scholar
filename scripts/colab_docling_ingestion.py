import os
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import pandas as pd

import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

def process_single_pdf(pdf_path: Path, output_dir: Path, converter: DocumentConverter):
    file_size_kb = pdf_path.stat().st_size / 1024
    start_time = time.perf_counter()
    status = "SUCCESS"
    error_msg = ""
    
    try:
        # Convert using the shared global model
        result = converter.convert(pdf_path)
        
        # Extract dict
        doc_dict = result.document.export_to_dict()
        out_path = output_dir / f"{pdf_path.stem}.json"
        
        # Write to Google Drive
        with open(out_path, "w") as f:
            json.dump(doc_dict, f)
            
    except Exception as e:
        status = "FAILED"
        error_msg = str(e)
        
    end_time = time.perf_counter()
    processing_time = end_time - start_time
    
    return {
        "filename": pdf_path.name,
        "file_size_kb": file_size_kb,
        "processing_time_sec": processing_time,
        "status": status,
        "error_msg": error_msg
    }

def main():
    parser = argparse.ArgumentParser(description="Docling Parallel Ingestion for Colab")
    parser.add_argument("--threads", type=int, default=2, help="Number of concurrent threads (workers)")
    parser.add_argument("--max-files", type=int, default=0, help="Maximum number of files to process. 0 means all.")
    parser.add_argument("--disable-ocr", action="store_true", help="Disable OCR for faster processing.")
    
    args = parser.parse_args()

    # Fixed paths for Google Drive
    SOURCE_DIR = Path("/content/drive/MyDrive/Projects/nlp_ml_gcs_pdfs")
    OUTPUT_DIR = Path("/content/drive/MyDrive/Projects/output")
    
    print(f"Reading PDFs from: {SOURCE_DIR}")
    print(f"Writing JSONs to:  {OUTPUT_DIR}")
    print(f"Threads: {args.threads}")
    print(f"Max Files: {'All' if args.max_files == 0 else args.max_files}")
    print(f"OCR Enabled: {not args.disable_ocr}")
    print("-" * 40)

    if not SOURCE_DIR.exists():
        print(f"ERROR: Source directory {SOURCE_DIR} does not exist.")
        print("Please ensure your Google Drive is mounted and the path is correct!")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Initializing Docling Model (Downloading weights if first time)...")
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = not args.disable_ocr
    pipeline_options.do_table_structure = True

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    print("Model Initialized successfully!")

    pdf_files = list(SOURCE_DIR.glob("*.pdf"))
    if args.max_files > 0:
        pdf_files = pdf_files[:args.max_files]
        
    print(f"Found {len(pdf_files)} PDFs to process.")
    if len(pdf_files) == 0:
        return

    metrics = []
    start_total = time.perf_counter()
    
    # Process the files in parallel using multithreading
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_single_pdf, pdf, OUTPUT_DIR, converter): pdf for pdf in pdf_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parsing PDFs"):
            metrics.append(future.result())
            
    end_total = time.perf_counter()
    total_time = end_total - start_total
    
    # Analytics & Summary
    df_metrics = pd.DataFrame(metrics)
    print(f"\nPipeline completed in {total_time:.2f} seconds.")
    
    success_rate = (df_metrics['status'] == 'SUCCESS').mean() * 100
    
    # Safely calculate average time (avoid NaN if all failed)
    success_mask = df_metrics['status'] == 'SUCCESS'
    if success_mask.any():
        avg_time = df_metrics.loc[success_mask, 'processing_time_sec'].mean()
    else:
        avg_time = 0.0
        
    throughput = len(df_metrics) / total_time

    print(f"\n=== INGESTION SUMMARY ===")
    print(f"Total PDFs Processed: {len(df_metrics)}")
    print(f"Success Rate:         {success_rate:.2f}%")
    print(f"Total Wall Time:      {total_time:.2f} s")
    print(f"Throughput:           {throughput:.2f} PDFs / sec")
    print(f"Avg Time per PDF:     {avg_time:.2f} s")

    failures = df_metrics[df_metrics['status'] == 'FAILED']
    if not failures.empty:
        print(f"\nEncountered {len(failures)} failures:")
        print(failures[['filename', 'error_msg']].head())
    
    # Save the metrics to CSV as a bonus!
    metrics_path = OUTPUT_DIR / "ingestion_metrics.csv"
    df_metrics.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
