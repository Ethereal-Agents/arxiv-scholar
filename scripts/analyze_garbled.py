import os
import sys
import glob
import re
import time

sys.path.append(os.path.abspath('src'))
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat

pdf_dir = "nlp_ml_gcs_pdfs"
pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
total_pdfs = len(pdf_files)
print(f"Total PDFs found: {total_pdfs}", flush=True)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
pipeline_options.accelerator_options = AcceleratorOptions(num_threads=1, device=AcceleratorDevice.CPU)
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

garbled_count = 0
clean_count = 0
errors = 0

start_time = time.time()
for i, pdf_path in enumerate(pdf_files):
    pdf_start = time.time()
    try:
        dl_doc = converter.convert(pdf_path).document
        sample_text = dl_doc.export_to_markdown()[:5000]
        if len(re.findall(r'/[A-Z0-9]{2}', sample_text)) > 20:
            garbled_count += 1
        else:
            clean_count += 1
    except Exception as e:
        errors += 1
        print(f"Error processing {pdf_path}: {e}", flush=True)
        
    pdf_time = time.time() - pdf_start
    # Warn if a document took an unusually long time
    if pdf_time > 10:
        print(f"⚠️ [SLOW PDF] {pdf_path} took {pdf_time:.1f}s to parse!", flush=True)
        
    # Flush progress every 5 documents
    if (i + 1) % 5 == 0 or (i + 1) == total_pdfs:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        print(f"[{i + 1}/{total_pdfs}] Garbled: {garbled_count} | Clean: {clean_count} | Errors: {errors} | Rate: {rate:.2f} doc/s | Elapsed: {elapsed:.0f}s", flush=True)

end_time = time.time()
print(f"\n--- Final Results ---", flush=True)
print(f"Total PDFs Analyzed: {total_pdfs}", flush=True)
print(f"Garbled PDFs: {garbled_count} ({garbled_count/total_pdfs*100:.1f}%)", flush=True)
print(f"Clean PDFs: {clean_count} ({clean_count/total_pdfs*100:.1f}%)", flush=True)
print(f"Errors: {errors}", flush=True)
print(f"Time taken: {end_time - start_time:.2f} seconds", flush=True)
