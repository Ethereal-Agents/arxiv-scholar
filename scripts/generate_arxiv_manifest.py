"""Generate arxiv_manifest.json — arXiv IDs for high-signal AI engineering papers.

Criteria:
1. Golden Term Bypass: Contains ultra-high-signal terms (vLLM, SWE-bench, etc.) -> Auto-include.
2. Regular: Matches >= 3 keyword groups (high-signal multidimensional papers)
3. Rescued: Matches exactly 2 groups + containing >= 2 AI engineering meta-terms.

Excludes domain-specific application papers (medical, physics, quantum, petroleum, robotics kinematics, pure math theory, etc.).

Output format:
[
  {
    "id": "2305.14314",
    ...
    "inclusion_reason": "regular | rescued_2group | golden_term"
  },
  ...
]
"""

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

METADATA_FILE = "/Users/ayushdubey/Downloads/arxiv-metadata-oai-snapshot.json"
OUTPUT_FILE   = "arxiv_manifest.json"
MIN_DATE      = "2022-01-01"

# Added cs.SE (Software Engineering) for coding assistants and SWE benchmarks
TARGET_CATEGORIES = {"cs.CL", "cs.AI", "cs.IR", "cs.LG", "cs.SE"} 

KEYWORD_GROUPS = {
    "RAG & Retrieval": [
        r"\bretrieval[- ]augmented\b", r"\bRAG\b", r"\bdense retrieval\b",
        r"\bpassage retrieval\b", r"\bdocument retrieval\b", r"\bsemantic search\b",
        r"\bhybrid search\b", r"\bvector database\b", r"\bvector store\b",
        r"\bembedding model\b", r"\bsentence embedding\b", r"\breranking\b",
        r"\bre-ranking\b", r"\bquestion answering\b",
    ],
    "Large Language Models": [
        r"\blarge language model\b", r"\bLLM\b", r"\bLLMs\b", r"\bGPT-4\b",
        r"\bChatGPT\b", r"\bfoundation model\b", r"\btransformer architecture\b",
    ],
    "Agents & Reasoning": [
        r"\bAI agent\b", r"\bLLM agent\b", r"\bautonomous agent\b",
        r"\blanguage agent\b", r"\bmulti-agent\b", r"\btool use\b",
        r"\bfunction calling\b", r"\bchain[- ]of[- ]thought\b", r"\bReAct\b",
        r"\bin-context learning\b", r"\bagentic\b", r"\bcode generation\b",
    ],
    "Training & Alignment": [
        r"\binstruction tuning\b", r"\binstruction following\b", r"\bRLHF\b",
        r"\breinforcement learning from human feedback\b", r"\bfine[- ]?tuning\b",
        r"\bprompt engineering\b", r"\bLoRA\b", r"\bparameter[- ]efficient\b",
        r"\bPEFT\b", r"\bDPO\b", r"\bdirect preference optimization\b",
    ],
    "Safety & Quality": [
        r"\bhallucination\b", r"\bgrounding\b", r"\bknowledge graph\b",
        r"\bdocument understanding\b", r"\bfactuality\b",
    ],
    "Inference & Systems": [
        r"\bspeculative decoding\b", r"\bKV cache\b", r"\bFlashAttention\b",
        r"\bquantization\b", r"\bvLLM\b", r"\bTensorRT\b", r"\bDeepSpeed\b",
        r"\bFSDP\b", r"\bMegatron\b", r"\bscaling law\b", r"\bmodel parallel",
        r"\bpipeline parallel", r"\btensor parallel", r"\binference latency\b",
        r"\bAWQ\b", r"\bGPTQ\b",
    ],
    "AI Developer Tools": [
        r"\bcode synthesis\b", r"\brepository-level\b", r"\bcoding assistant\b",
        r"\bAI programmer\b", r"\bSWE-bench\b", r"\bHumanEval\b",
        r"\bDSPy\b", r"\bLangChain\b", r"\bLlamaIndex\b", r"\bOpenHands\b", r"\bSWE-agent\b",
    ]
}

# Golden terms bypass typical group requirements because they are pure AI Engineering
GOLDEN_TERMS = [
    r"\bvLLM\b", r"\bFlashAttention\b", r"\bSWE-bench\b", r"\bTensorRT-LLM\b", 
    r"\bHumanEval\b", r"\bDSPy\b", r"\bLangChain\b", r"\bLlamaIndex\b", 
    r"\bOpenHands\b", r"\bSWE-agent\b", r"\bDeepSpeed\b"
]

# Domain exclusion (expanded to remove pure theory, hard sciences, physical robotics)
DOMAIN_EXCLUSION_PATTERNS = [
    # Medical/Biology
    r"\bbiomedical\b", r"\bclinical\b", r"\bradiology\b", r"\bradiolog",
    r"\bmedical\b", r"\bhealthcare\b", r"\bpatient\b", r"\bdiagnos",
    r"\bgenomic\b", r"\bdrug\b", r"\bchemist", r"\bbiolog", r"\bEHR\b", 
    r"\bclinical trial\b", r"\bepidemiolog\b",
    # Physics/Hard Science
    r"\bquantum\b", r"\bastronom", r"\bastrophysic\b", r"\bphysics\b", 
    r"\bmaterials science\b", r"\bfluid dynamics\b", r"\bclimate change\b",
    r"\bgeolog\b", r"\bpetroleum\b", r"\breservoir\b", r"\bagricultur", r"\bcrop\b",
    # Civil/Vehicular/Physical Robotics
    r"\burban planning\b", r"\bautonomous driving\b", r"\bself-driving\b", 
    r"\bvehicular\b", r"\bkinematics\b", r"\bmanipulator arm\b", r"\bquadruped\b",
    # Pure Math/Theory
    r"\bRademacher\b", r"\bLipschitz\b", r"\basymptotic convergence\b"
]

# AI engineering meta-terms for rescuing 2-group papers
META_TERMS = [
    r"\bpipeline\b", r"\bframework\b", r"\barchitecture\b", r"\bbenchmark\b",
    r"\bablation\b", r"\btoolkit\b", r"\bindexing\b", r"\bchunking\b",
    r"\bembedding space\b", r"\bvector search\b", r"\bretrieval pipeline\b",
    r"\bend-to-end\b", r"\bsystem design\b", r"\bopen-source\b",
    r"\bscalab", r"\blatency\b", r"\bthroughput\b", r"\btoken\b",
]

# Precompile
COMPILED_GROUPS = {
    name: [re.compile(p, re.IGNORECASE) for p in patterns]
    for name, patterns in KEYWORD_GROUPS.items()
}
COMPILED_EXCLUSIONS = [re.compile(p, re.IGNORECASE) for p in DOMAIN_EXCLUSION_PATTERNS]
COMPILED_META = [re.compile(p, re.IGNORECASE) for p in META_TERMS]
COMPILED_GOLDEN = [re.compile(p, re.IGNORECASE) for p in GOLDEN_TERMS]

def get_matched_groups(text: str) -> list[str]:
    matched = []
    for group_name, patterns in COMPILED_GROUPS.items():
        for pat in patterns:
            if pat.search(text):
                matched.append(group_name)
                break
    return matched

def is_excluded_domain(text: str) -> bool:
    return any(pat.search(text) for pat in COMPILED_EXCLUSIONS)

def get_meta_term_count(text: str) -> int:
    return sum(1 for p in COMPILED_META if p.search(text))

def has_golden_term(text: str) -> bool:
    return any(pat.search(text) for pat in COMPILED_GOLDEN)

def arxiv_id_to_gcs_path(arxiv_id: str) -> str:
    if "." in arxiv_id:
        month = arxiv_id.split(".")[0]
        return f"arxiv/arxiv/pdf/{month}/{arxiv_id}.pdf"
    elif "/" in arxiv_id:
        category, number = arxiv_id.split("/")
        month = number[:4]
        return f"arxiv/arxiv/pdf/{month}/{category}{number}.pdf"
    else:
        return f"arxiv/arxiv/pdf/{arxiv_id[:4]}/{arxiv_id}.pdf"

def main():
    start = time.time()
    print(f"Reading: {METADATA_FILE}")
    print(f"Criteria: Golden Term OR >=3 groups OR (2 groups + >=2 meta terms)")
    print()

    manifest = []
    total = 0
    passed_cat = 0
    passed_date = 0
    excluded_domain = 0
    
    count_golden = 0
    count_regular = 0
    count_rescued = 0

    with open(METADATA_FILE, "r") as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 500_000 == 0:
                elapsed = time.time() - start
                print(f"  ...{line_num:,} papers scanned ({elapsed:.0f}s), {len(manifest):,} collected so far", file=sys.stderr)

            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1

            cats = set(paper.get("categories", "").split())
            if not cats & TARGET_CATEGORIES:
                continue
            passed_cat += 1

            update_date = paper.get("update_date", "")
            if update_date < MIN_DATE:
                continue
            passed_date += 1

            title    = paper.get("title", "").replace("\n", " ").strip()
            abstract = paper.get("abstract", "").replace("\n", " ").strip()
            text     = f"{title} {abstract}"

            if is_excluded_domain(text):
                excluded_domain += 1
                continue

            matched_groups = get_matched_groups(text)
            
            # Logic for inclusion
            inclusion_reason = None
            if has_golden_term(text):
                inclusion_reason = "golden_term"
                count_golden += 1
            elif len(matched_groups) >= 3:
                inclusion_reason = "regular"
                count_regular += 1
            elif len(matched_groups) == 2 and get_meta_term_count(text) >= 2:
                inclusion_reason = "rescued_2group"
                count_rescued += 1
            else:
                continue

            manifest.append({
                "id":             paper["id"],
                "title":          title,
                "abstract":       abstract,
                "categories":     paper.get("categories", ""),
                "update_date":    update_date,
                "groups_matched": len(matched_groups),
                "matched_groups": matched_groups,
                "gcs_path":       arxiv_id_to_gcs_path(paper["id"]),
                "inclusion_reason": inclusion_reason
            })

    # Sort
    # We sort by: whether it's golden (1) or not (0), then groups matched, then date.
    manifest.sort(key=lambda x: (
        1 if x["inclusion_reason"] == "golden_term" else 0,
        x["groups_matched"], 
        x["update_date"]
    ), reverse=True)

    elapsed = time.time() - start

    depth_counts = Counter(p["groups_matched"] for p in manifest)
    
    print()
    print(f"{'='*60}")
    print(f"  MANIFEST GENERATION COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Total scanned:          {total:>10,}")
    print(f"  Passed category filter: {passed_cat:>10,}")
    print(f"  Passed date filter:     {passed_date:>10,}")
    print(f"  Excluded (domain):      {excluded_domain:>10,}")
    print()
    print(f"  Included via Golden:    {count_golden:>10,}")
    print(f"  Included via Regular:   {count_regular:>10,}")
    print(f"  Included via Rescued:   {count_rescued:>10,}")
    print(f"  Final manifest size:    {len(manifest):>10,}")
    print()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(manifest, f, indent=2)

    size_mb = Path(OUTPUT_FILE).stat().st_size / 1_000_000
    print(f"  Saved → {OUTPUT_FILE}  ({size_mb:.1f} MB)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
