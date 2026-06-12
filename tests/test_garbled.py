import re
text = "/D7/D4/CP\r/CT /CX/D7 /D7/D4/CT\r/CX/AS/CT/CS /CX/D2 /DB/CW/CX\r/CW /D8/CW/CT /CP/D0\r/D9/D0/CP/D8/CX/D3/D2 /CX/D7 /D1/D3/D7/D8 /D6/CT/D0/CX/CP/CQ"
matches = re.findall(r'/[A-Z0-9]{2}', text)
print(f"Matches: {len(matches)}")
print(matches)

# Let's run it on the actual PDF to be sure
import sys
import os
sys.path.append(os.path.abspath('src'))
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions

pdf_path = "./notebooks/notebook_trial/0704.0001v2.pdf"
if not os.path.exists(pdf_path):
    print("PDF not found")
    sys.exit(0)

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False
converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)
from docling.datamodel.base_models import InputFormat
doc = converter.convert(pdf_path).document
full_text = doc.export_to_markdown()
print(f"Length of text: {len(full_text)}")
matches = re.findall(r'/[A-Z0-9]{2}', full_text)
print(f"Matches in PDF: {len(matches)}")

