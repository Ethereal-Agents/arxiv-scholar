import json
import re
import os

def extract_year_from_arxiv_id(arxiv_id: str):
    """
    Extracts the publication year from an arxiv_id.
    Handles both new format (YYMM.NNNN) and old format (archive/YYMMNNN).
    Returns the year as a 4-digit integer.
    """
    if not arxiv_id:
        return None
        
    # Check for old format (e.g., math/9901001, astro-ph/0102030)
    old_format_match = re.search(r'^[a-z\-]+(?:-[a-z]+)?/(\d{2})\d{5}(?:v\d+)?$', arxiv_id, re.IGNORECASE)
    if old_format_match:
        yy = int(old_format_match.group(1))
    else:
        # Check for new format (e.g., 2103.12345, 0901.1234, 1502.1234v2)
        new_format_match = re.search(r'^(\d{2})\d{2}\.\d+', arxiv_id)
        if new_format_match:
            yy = int(new_format_match.group(1))
        else:
            return None

    # Assuming YY >= 90 means 19YY, else 20YY. ArXiv started in 1991.
    if yy >= 90:
        return 1900 + yy
    else:
        return 2000 + yy

def run_tests():
    print("--- Running Test Cases ---")
    test_cases = {
        "2103.12345": 2021,
        "0901.1234": 2009,
        "1502.1234v2": 2015,
        "math/9901001": 1999,
        "astro-ph/0102030": 2001,
        "cond-mat/9812345": 1998,
        "invalid_id": None
    }
    
    all_passed = True
    for arxiv_id, expected_year in test_cases.items():
        year = extract_year_from_arxiv_id(arxiv_id)
        status = "PASS" if year == expected_year else "FAIL"
        print(f"[{status}] ID: {arxiv_id:15} -> Extracted: {year} | Expected: {expected_year}")
        if status == "FAIL":
            all_passed = False
            
    print(f"All tests passed? {all_passed}\n")

def process_sample():
    print("--- Processing Sample Record ---")
    folder_path = os.path.expanduser("~/Downloads/arxiv_extracted_jsonl/arxiv_embeddings/")
    
    # Get the first JSONL file
    files = [f for f in os.listdir(folder_path) if f.endswith(".jsonl")]
    if not files:
        print("No JSONL files found.")
        return
        
    sample_file = os.path.join(folder_path, files[0])
    
    with open(sample_file, "r") as f:
        line = f.readline()
        record = json.loads(line)
        
    payload = record.get("payload", {})
    arxiv_id = payload.get("metadata", {}).get("arxiv_id")
    
    print(f"Original arxiv_id: {arxiv_id}")
    
    # Add year to metadata
    year = extract_year_from_arxiv_id(arxiv_id)
    if year:
        payload["metadata"]["year"] = year
    
    # Print the modified payload
    print("\nModified Payload (Preview):")
    print(json.dumps(payload, indent=2))

if __name__ == "__main__":
    run_tests()
    process_sample()
