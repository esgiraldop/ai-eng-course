"""
augment_ground_truth.py
------------------------
Retroactively augments an existing ground truth benchmark matrix (ground_truth_matrix.json)
with arbitrary candidate metadata attributes extracted from the parsed CV dataset 
(cv_extracted_info_eng.json / cv_extracted_info.json) using 'file_name' matching.

Why this script was created:
To avoid re-running expensive LLM API evaluation calls (and spending LLM tokens)
whenever new candidate metadata fields need to be added to an existing benchmark matrix.

Validation Rules & Safeguards:
1. Reserved Parameters Protection: You cannot pass core benchmark evaluation fields 
   ('cv_id', 'file_name', 'score', 'reasoning') as fields to augment.
2. Attribute Existence Check: Checks that all requested attributes exist in the 
   CV dataset before proceeding.

Usage:
1. Command Line:
   # Default fields ('name', 'email')
   python benchmark/augment_ground_truth.py

   # Custom attributes (e.g. 'profession', 'seniority_level', 'skills')
   python benchmark/augment_ground_truth.py --fields name email profession seniority_level

2. Python Module Import:
   from benchmark.augment_ground_truth import augment_ground_truth
   augment_ground_truth(
       matrix_file="data/ground_truth_matrix.json",
       cv_file="data/cv_extracted_info_eng.json",
       fields=["name", "email", "profession"]
   )
"""

import json
import argparse
import os
from typing import List, Optional

RESERVED_FIELDS = {"cv_id", "file_name", "score", "reasoning"}


def augment_ground_truth(
    matrix_file: str = "data/ground_truth_matrix.json",
    cv_file: str = "data/cv_extracted_info_eng.json",
    fields: Optional[List[str]] = None,
    fallback_cv_file: Optional[str] = "data/cv_extracted_info.json"
):
    if fields is None:
        fields = ["name", "email"]

    # 1. Validate against reserved fields
    forbidden_requested = set(fields).intersection(RESERVED_FIELDS)
    if forbidden_requested:
        raise ValueError(
            f"Cannot pass reserved evaluation parameter(s) {sorted(list(forbidden_requested))} to augment. "
            f"Reserved parameters are: {sorted(list(RESERVED_FIELDS))}."
        )

    # 2. File existence checks
    if not os.path.exists(matrix_file):
        raise FileNotFoundError(f"Ground truth matrix file '{matrix_file}' does not exist.")

    if not os.path.exists(cv_file):
        if fallback_cv_file and os.path.exists(fallback_cv_file):
            print(f"Primary CV file '{cv_file}' not found. Using fallback '{fallback_cv_file}'.")
            cv_file = fallback_cv_file
        else:
            raise FileNotFoundError(f"CV extracted info file '{cv_file}' does not exist.")

    # 3. Load CV dataset & check attribute existence
    with open(cv_file, "r", encoding="utf-8") as f:
        cv_dataset = json.load(f)

    if not cv_dataset:
        raise ValueError(f"CV dataset file '{cv_file}' is empty.")

    available_keys = set()
    for cv in cv_dataset:
        available_keys.update(cv.keys())

    missing_fields = set(fields) - available_keys
    if missing_fields:
        raise ValueError(
            f"Requested attribute(s) {sorted(list(missing_fields))} do not exist in the CV dataset. "
            f"Available attributes in CV data are: {sorted(list(available_keys))}."
        )

    # Build lookup map by file_name
    cv_lookup = {}
    for cv in cv_dataset:
        file_name = cv.get("file_name")
        if file_name:
            cv_lookup[file_name] = {field: cv.get(field, "") for field in fields}

    # Check fallback file for missing file_names if available
    if fallback_cv_file and os.path.exists(fallback_cv_file) and cv_file != fallback_cv_file:
        with open(fallback_cv_file, "r", encoding="utf-8") as f:
            fallback_dataset = json.load(f)
        for cv in fallback_dataset:
            file_name = cv.get("file_name")
            if file_name and file_name not in cv_lookup:
                cv_lookup[file_name] = {field: cv.get(field, "") for field in fields}

    print(f"Loaded {len(cv_lookup)} CV metadata records for lookup.")
    print(f"Augmenting attributes: {fields}")

    # 4. Load existing ground_truth_matrix.json
    with open(matrix_file, "r", encoding="utf-8") as f:
        ground_truth_matrix = json.load(f)

    total_evaluations = 0
    augmented_count = 0

    # 5. Augment each evaluation entry
    for jd_id, jd_data in ground_truth_matrix.items():
        evaluations = jd_data.get("evaluations", [])
        updated_evaluations = []

        for item in evaluations:
            total_evaluations += 1
            file_name = item.get("file_name", "")
            meta = cv_lookup.get(file_name, {})

            new_item = {"cv_id": item.get("cv_id", "")}
            
            for field in fields:
                new_item[field] = item.get(field) or meta.get(field, "")

            new_item["file_name"] = file_name
            new_item["score"] = item.get("score", 0)
            new_item["reasoning"] = item.get("reasoning", "")

            if any(meta.get(f) for f in fields):
                augmented_count += 1

            updated_evaluations.append(new_item)

        jd_data["evaluations"] = updated_evaluations

    # 6. Save updated ground truth matrix with unescaped UTF-8 formatting
    with open(matrix_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth_matrix, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully augmented '{matrix_file}' with fields {fields}!")
    print(f"Total evaluations processed: {total_evaluations}")
    print(f"Total candidate records updated: {augmented_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment existing ground_truth_matrix.json with custom candidate attributes from cv_extracted_info."
    )
    parser.add_argument(
        "--matrix-file",
        type=str,
        default="data/ground_truth_matrix.json",
        help="Path to ground_truth_matrix.json file"
    )
    parser.add_argument(
        "--cv-file",
        type=str,
        default="data/cv_extracted_info_eng.json",
        help="Path to cv_extracted_info_eng.json file"
    )
    parser.add_argument(
        "--fields",
        "-f",
        nargs="+",
        default=["name", "email"],
        help="Attributes from CV data to add to ground_truth_matrix.json (default: name email)"
    )

    args = parser.parse_args()
    augment_ground_truth(matrix_file=args.matrix_file, cv_file=args.cv_file, fields=args.fields)
