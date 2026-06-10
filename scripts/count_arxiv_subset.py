"""Analyze the arXiv metadata snapshot to count papers matching our subset criteria.

Streams through the file line-by-line to avoid loading 4GB into memory.
Revised keyword groups: covers AI engineering + general AI research.
"""

import json
import re
import sys
import time
from collections import defaultdict

METADATA_FILE = "/Users/ayushdubey/Downloads/arxiv-metadata-oai-snapshot.json"

# --- FILTER CRITERIA ---

TARGET_CATEGORIES = {"cs.CL", "cs.AI", "cs.IR"}
MIN_DATE = "2022-01-01"

# Keyword groups — each represents a distinct dimension of relevance.
# Papers matching more groups rank higher for selection.
KEYWORD_GROUPS = {
    "RAG & Retrieval": [
        r"\bretrieval[- ]augmented\b",
        r"\bRAG\b",
        r"\bdense retrieval\b",
        r"\bpassage retrieval\b",
        r"\bdocument retrieval\b",
        r"\bsemantic search\b",
        r"\bhybrid search\b",
        r"\bvector database\b",
        r"\bvector store\b",
        r"\bembedding model\b",
        r"\bsentence embedding\b",
        r"\breranking\b",
        r"\bre-ranking\b",
        r"\bquestion answering\b",
    ],
    "Large Language Models": [
        r"\blarge language model\b",
        r"\bLLM\b",
        r"\bLLMs\b",
        r"\bGPT-4\b",
        r"\bChatGPT\b",
        r"\bfoundation model\b",
        r"\btransformer architecture\b",
    ],
    "Agents & Reasoning": [
        r"\bAI agent\b",
        r"\bLLM agent\b",
        r"\bautonomous agent\b",
        r"\blanguage agent\b",
        r"\bmulti-agent\b",
        r"\btool use\b",
        r"\bfunction calling\b",
        r"\bchain[- ]of[- ]thought\b",
        r"\bReAct\b",
        r"\bin-context learning\b",
        r"\bagentic\b",
        r"\bcode generation\b",
    ],
    "Training & Alignment": [
        r"\binstruction tuning\b",
        r"\binstruction following\b",
        r"\bRLHF\b",
        r"\breinforcement learning from human feedback\b",
        r"\bfine[- ]?tuning\b",
        r"\bprompt engineering\b",
        r"\bLoRA\b",
        r"\bparameter[- ]efficient\b",
        r"\bPEFT\b",
        r"\bDPO\b",
        r"\bdirect preference optimization\b",
    ],
    "Safety & Quality": [
        r"\bhallucination\b",
        r"\bgrounding\b",
        r"\bknowledge graph\b",
        r"\bdocument understanding\b",
        r"\bfactuality\b",
    ],
}

# Precompile all keyword patterns
COMPILED_GROUPS = {}
for group_name, patterns in KEYWORD_GROUPS.items():
    COMPILED_GROUPS[group_name] = [re.compile(p, re.IGNORECASE) for p in patterns]


def matches_categories(categories_str: str) -> bool:
    cats = set(categories_str.split())
    return bool(cats & TARGET_CATEGORIES)


def matches_date(update_date: str) -> bool:
    return update_date >= MIN_DATE


def get_matching_groups(text: str) -> list:
    """Return list of keyword group names that match the text."""
    matched = []
    for group_name, patterns in COMPILED_GROUPS.items():
        for pat in patterns:
            if pat.search(text):
                matched.append(group_name)
                break
    return matched


def get_matched_keywords(text: str) -> dict:
    """Return dict of group_name -> list of matched keyword patterns."""
    result = {}
    for group_name, patterns in COMPILED_GROUPS.items():
        matched_kws = []
        for pat in patterns:
            if pat.search(text):
                matched_kws.append(pat.pattern)
        if matched_kws:
            result[group_name] = matched_kws
    return result


def main():
    print(f"Scanning: {METADATA_FILE}")
    print(f"Filters: categories={TARGET_CATEGORIES}, date>={MIN_DATE}")
    print(f"Keyword groups: {len(KEYWORD_GROUPS)}")
    print("=" * 70)
    print()

    start = time.time()

    # Counters
    total_papers = 0
    cat_match = 0
    cat_and_date_match = 0
    fully_matched = 0

    # Breakdowns
    by_year = defaultdict(int)
    by_group = defaultdict(int)
    by_num_groups = defaultdict(int)
    by_year_and_group = defaultdict(lambda: defaultdict(int))
    by_category = defaultdict(int)
    by_individual_keyword = defaultdict(int)  # track which specific keywords hit

    with open(METADATA_FILE, "r") as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 500_000 == 0:
                elapsed = time.time() - start
                print(f"  ...processed {line_num:,} papers ({elapsed:.0f}s)", file=sys.stderr)

            try:
                paper = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_papers += 1
            categories_str = paper.get("categories", "")
            update_date = paper.get("update_date", "")

            if not matches_categories(categories_str):
                continue
            cat_match += 1

            if not matches_date(update_date):
                continue
            cat_and_date_match += 1

            abstract = paper.get("abstract", "")
            title = paper.get("title", "")
            text = f"{title} {abstract}"

            matched_groups = get_matching_groups(text)
            if not matched_groups:
                continue

            fully_matched += 1
            year = update_date[:4] if update_date else "unknown"
            by_year[year] += 1
            by_num_groups[len(matched_groups)] += 1

            for g in matched_groups:
                by_group[g] += 1
                by_year_and_group[year][g] += 1

            for cat in set(categories_str.split()) & TARGET_CATEGORIES:
                by_category[cat] += 1

            # Track individual keyword hits
            kw_matches = get_matched_keywords(text)
            for group_name, kw_list in kw_matches.items():
                for kw in kw_list:
                    by_individual_keyword[f"{group_name}: {kw}"] += 1

    elapsed = time.time() - start

    # --- RESULTS ---
    print()
    print("=" * 70)
    print(f"  ARXIV SUBSET ANALYSIS v2 — COMPLETE ({elapsed:.0f}s)")
    print("=" * 70)

    print(f"\n--- FILTERING FUNNEL ---")
    print(f"  Total papers in dataset:             {total_papers:>10,}")
    print(f"  → Match categories (cs.CL/AI/IR):    {cat_match:>10,}")
    print(f"  → + Date >= 2022-01-01:              {cat_and_date_match:>10,}")
    print(f"  → + Keyword match (ANY group):       {fully_matched:>10,}")

    print(f"\n--- BY YEAR ---")
    for year in sorted(by_year.keys()):
        bar = "█" * (by_year[year] // 250)
        print(f"  {year}:  {by_year[year]:>6,}  {bar}")
    print(f"  {'TOTAL':>5}: {fully_matched:>6,}")

    print(f"\n--- BY KEYWORD GROUP ---")
    for group in sorted(by_group, key=by_group.get, reverse=True):
        pct = (by_group[group] / fully_matched) * 100
        bar = "█" * int(pct / 2)
        print(f"  {group:25s}  {by_group[group]:>6,}  ({pct:5.1f}%)  {bar}")

    print(f"\n--- KEYWORD DEPTH (groups matched per paper) ---")
    for n in sorted(by_num_groups.keys()):
        pct = (by_num_groups[n] / fully_matched) * 100
        cum = sum(by_num_groups[k] for k in by_num_groups if k >= n)
        print(f"  {n} group(s):  {by_num_groups[n]:>6,}  ({pct:5.1f}%)   ≥{n}: {cum:>6,}")

    print(f"\n--- BY CATEGORY (papers can appear in multiple) ---")
    for cat in sorted(by_category, key=by_category.get, reverse=True):
        print(f"  {cat}:  {by_category[cat]:>6,}")

    print(f"\n--- TOP 20 INDIVIDUAL KEYWORDS BY HIT COUNT ---")
    sorted_kws = sorted(by_individual_keyword.items(), key=lambda x: x[1], reverse=True)[:20]
    for kw, count in sorted_kws:
        pct = (count / fully_matched) * 100
        print(f"  {count:>6,}  ({pct:5.1f}%)  {kw}")

    print(f"\n--- YEAR × KEYWORD GROUP ---")
    groups_sorted = sorted(by_group, key=by_group.get, reverse=True)
    header = f"  {'Year':>6}  " + "  ".join(f"{g[:14]:>14}" for g in groups_sorted)
    print(header)
    print("  " + "-" * (8 + 16 * len(groups_sorted)))
    for year in sorted(by_year_and_group.keys()):
        row = f"  {year:>6}  "
        row += "  ".join(f"{by_year_and_group[year].get(g, 0):>14,}" for g in groups_sorted)
        print(row)

    print(f"\n{'=' * 70}")
    print(f"  RESULT: {fully_matched:,} papers match all criteria.")
    print(f"  Budget: ~3,700 – 6,000 papers (Qdrant free tier).")
    if fully_matched > 6000:
        cum_3plus = sum(by_num_groups[k] for k in by_num_groups if k >= 3)
        cum_2plus = sum(by_num_groups[k] for k in by_num_groups if k >= 2)
        print(f"  Papers matching ≥3 groups: {cum_3plus:,} (highest signal)")
        print(f"  Papers matching ≥2 groups: {cum_2plus:,}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
