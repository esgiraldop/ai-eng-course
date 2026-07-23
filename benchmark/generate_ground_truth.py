"""
generate_ground_truth.py
------------------------
Generates an unbiased ground-truth benchmark matrix for vector search evaluation
by scoring candidate CVs against Job Descriptions using Gemini 2.5 Flash.

Why Batch Processing (10 CVs/call) vs. Loading All 100 CVs at Once:
1. Eliminates "Lost in the Middle" Bias: Prevents the LLM from over-indexing 
   on candidates at the start/end while ignoring those in positions 20–80.
2. Prevents Reasoning Degradation: Avoids superficial keyword-matching shortcuts 
   that happen when forcing an LLM to reason across 100k+ tokens at once.
3. Enables Graded Pointwise Scoring (0–3): Generates absolute relevance scores 
   required for NDCG@K and MRR@K metrics instead of brittle relative rankings.
4. Guarantees Output Integrity: Keeps response payloads small to prevent JSON 
   truncation and API parsing errors when paired with Pydantic schemas.
"""

import json
import time
import argparse
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from tqdm import tqdm
from google import genai
from google.genai import types
import env_config  # Load environment variables (.env)


# ------------------------------------------------------------------
# 1. Pydantic Schemas for Structured JSON Output
# ------------------------------------------------------------------
class CVScore(BaseModel):
    cv_id: str = Field(description="The unique identifier or index of the CV being evaluated")
    score: int = Field(description="Relevance score: 0 = Irrelevant, 1 = Low Match, 2 = Good Match, 3 = Perfect Match")
    reasoning: str = Field(description="1-sentence justification for the assigned score")

class CVScoreBatch(BaseModel):
    evaluations: List[CVScore]


# ------------------------------------------------------------------
# 2. Batch Generator Helper
# ------------------------------------------------------------------
def chunk_list(data: List[Any], chunk_size: int):
    """Yield successive chunks from data."""
    for i in range(0, len(data), chunk_size):
        yield data[i : i + chunk_size]


# ------------------------------------------------------------------
# 3. Main Evaluation Engine
# ------------------------------------------------------------------
def generate_ground_truth(
    cv_file_path: str = "data/cv_extracted_info.json",
    jd_file_path: str = "data/job_descriptions_train_batch.json",
    job_descriptions: Optional[List[Dict[str, Any]]] = None,
    output_file: str = "data/ground_truth_matrix.json",
    max_cvs: Optional[int] = None,
    max_jds: Optional[int] = None,
    batch_size: int = 10,
    model_name: str = "gemini-2.5-flash"
):
    # Initialize the GenAI client (picks up GEMINI_API_KEY env var)
    client = genai.Client()

    # Load the CV dataset JSON array
    with open(cv_file_path, "r", encoding="utf-8") as f:
        cv_dataset = json.load(f)

    # Filter/Limit CV dataset if max_cvs is specified
    if max_cvs is not None and max_cvs > 0:
        cv_dataset = cv_dataset[:max_cvs]
        print(f"Loaded {len(cv_dataset)} CVs from '{cv_file_path}' (limited by max_cvs={max_cvs}).")
    else:
        print(f"Loaded {len(cv_dataset)} CVs from '{cv_file_path}'.")

    # Ensure every CV has a unique cv_id identifier
    for idx, cv in enumerate(cv_dataset):
        if "id" not in cv and "cv_id" not in cv:
            cv["cv_id"] = f"cv_{idx+1:03d}"

    # Load or use provided Job Descriptions
    if job_descriptions is None:
        with open(jd_file_path, "r", encoding="utf-8") as f:
            job_descriptions = json.load(f)
        print(f"Loaded {len(job_descriptions)} Job Descriptions from '{jd_file_path}'.")

    # Filter/Limit Job Descriptions if max_jds is specified
    if max_jds is not None and max_jds > 0:
        job_descriptions = job_descriptions[:max_jds]
        print(f"Processing top {len(job_descriptions)} Job Descriptions (limited by max_jds={max_jds}).")

    ground_truth_matrix = {}

    for jd in job_descriptions:
        jd_id = jd.get("id") or jd.get("jd_id")
        jd_title = jd.get("title") or jd.get("job_title", "")
        jd_description = jd.get("description") or jd.get("role_description", "")

        print(f"\n==================================================")
        print(f"Evaluating Job Description: [{jd_id}] {jd_title}")
        print(f"==================================================")

        jd_scores = []
        cv_batches = list(chunk_list(cv_dataset, batch_size))

        for batch in tqdm(cv_batches, desc=f"Scoring batches for {jd_id}"):
            # Prepare minimal CV payload to save tokens
            batch_payload = []
            for cv in batch:
                cv_id_val = str(cv.get("id") or cv.get("cv_id"))
                batch_payload.append({
                    "cv_id": cv_id_val,
                    "candidate_data": cv
                })

            prompt = f"""
            You are an expert technical recruiter creating an objective benchmark dataset.
            Evaluate each candidate in the provided batch against the target Job Description.

            ### TARGET JOB DESCRIPTION:
            Title: {jd_title}
            Description & Requirements:
            {jd_description}

            ### CANDIDATE BATCH ({len(batch)} candidates):
            {json.dumps(batch_payload, indent=2)}

            ### SCORING GUIDELINES:
            - Score 0: Irrelevant (wrong field or missing core mandatory requirements)
            - Score 1: Low Match (adjacent background, missing key tech stack or experience)
            - Score 2: Good Match (solid core skills match, reasonable experience)
            - Score 3: Perfect Match (strong direct technical overlap, years of experience, and role fit)

            Output evaluations for ALL candidates in this batch.
            """

            max_retries = 3
            batch_results = None
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=CVScoreBatch,
                            temperature=0.0  # Deterministic scoring
                        )
                    )

                    # SDK automatically parses into the Pydantic class
                    batch_results = response.parsed
                    if batch_results and batch_results.evaluations:
                        break
                except Exception as e:
                    print(f"\n[Warning] API error on batch (attempt {attempt+1}/{max_retries}): {e}.")
                    if attempt < max_retries - 1:
                        time.sleep(3)

            if batch_results and batch_results.evaluations:
                for eval_item in batch_results.evaluations:
                    jd_scores.append(eval_item.model_dump())
            else:
                print(f"[Error] Failed to get evaluations for current batch after {max_retries} attempts.")

            # Brief pause to respect rate limits
            time.sleep(0.5)

        ground_truth_matrix[jd_id] = {
            "title": jd_title,
            "job_description": jd_description,
            "evaluations": jd_scores
        }

    # Save the aggregated benchmark matrix
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth_matrix, f, indent=2)

    print(f"\nSuccessfully generated ground truth matrix! Saved to '{output_file}'.")


# ------------------------------------------------------------------
# 4. Example Usage & CLI Entrypoint
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ground-truth benchmark matrix by scoring candidate CVs against Job Descriptions using Gemini."
    )
    parser.add_argument("--cv-file", type=str, default="data/cv_extracted_info.json", help="Path to CV extracted info JSON file")
    parser.add_argument("--jd-file", type=str, default="data/job_descriptions_train_batch.json", help="Path to Job Descriptions JSON file")
    parser.add_argument("--output-file", type=str, default="data/ground_truth_matrix.json", help="Path to save ground truth matrix JSON output")
    parser.add_argument("--max-cvs", "-c", type=int, default=None, help="Maximum number of CVs to evaluate (for testing/cost control)")
    parser.add_argument("--max-jds", "-j", type=int, default=None, help="Maximum number of Job Descriptions to process (for testing/cost control)")
    parser.add_argument("--batch-size", "-b", type=int, default=10, help="Number of CVs per LLM evaluation call")
    parser.add_argument("--model-name", "-m", type=str, default="gemini-2.5-flash", help="Gemini model name to use for scoring")

    args = parser.parse_args()

    generate_ground_truth(
        cv_file_path=args.cv_file,
        jd_file_path=args.jd_file,
        output_file=args.output_file,
        max_cvs=args.max_cvs,
        max_jds=args.max_jds,
        batch_size=args.batch_size,
        model_name=args.model_name
    )