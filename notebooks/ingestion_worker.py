import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
import json

_converter = None
_io_pool = None

def init_worker():
    """Initializes the Docling model and an I/O thread pool once per worker process."""
    global _converter, _io_pool
    if _converter is None:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False 
        pipeline_options.do_table_structure = True 
        
        _converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF],
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        # Create a single background thread per worker for disk writes
        _io_pool = ThreadPoolExecutor(max_workers=1)

def _write_to_disk(doc_dict, out_path):
    with open(out_path, "w") as f:
        json.dump(doc_dict, f)

def process_single_pdf(pdf_path: Path, output_dir: Path):
    """
    Parses a single PDF and offloads the file write to a background thread.
    """
    file_size_kb = pdf_path.stat().st_size / 1024
    start_time = time.perf_counter()
    status = "SUCCESS"
    error_msg = ""
    
    try:
        # 1. CPU Bound: Convert using the process-local model
        result = _converter.convert(pdf_path)
        
        # 2. Extract dict
        doc_dict = result.document.export_to_dict()
        out_path = output_dir / f"{pdf_path.stem}.json"
        
        # 3. I/O Bound: Submit write task to background thread (Async)
        _io_pool.submit(_write_to_disk, doc_dict, out_path)
            
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
