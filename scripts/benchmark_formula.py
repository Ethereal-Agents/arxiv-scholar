import sys
import os
import time
sys.path.append(os.path.abspath('src'))
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.datamodel.base_models import InputFormat

pdf_path = "./notebooks/notebook_trial/0704.0001v2.pdf"

def run_conversion(enrich_formula):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_formula_enrichment = enrich_formula
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=1, device=AcceleratorDevice.CPU)
    
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    
    start = time.time()
    dl_doc = converter.convert(pdf_path).document
    end = time.time()
    
    # Just to verify formulas are found
    text = dl_doc.export_to_markdown()
    formula_count = text.count("$$") // 2 + text.count("<!-- formula-not-decoded -->")
    
    return end - start, formula_count

print("Warming up models...")
run_conversion(False) # Warmup

print("Running WITHOUT formula enrichment...")
time_without, count_without = run_conversion(False)

print("Running WITH formula enrichment...")
time_with, count_with = run_conversion(True)

print(f"Time WITHOUT formula enrichment: {time_without:.2f} seconds")
print(f"Time WITH formula enrichment: {time_with:.2f} seconds")
print(f"Extra compute overhead: {time_with - time_without:.2f} seconds")
