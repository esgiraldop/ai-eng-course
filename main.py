import argparse
from db.db import DB
from db.ingest_db import ingest_db
import env_config  # noqa: F401
from preprocessing.parse_docs import parse_docs
import json
from utils import str2bool

def run_offline_ingestion(args):
    model_name = "microsoft/harrier-oss-v1-0.6b"
    collection_name = "cvs"
    cv_filepath: str = "data/cv_extracted_info_eng.json"

    if args.preprocess_docs: (cvs_eng, job_pos_eng) = parse_docs()

    # Initializing database
    cvs_db = DB(model_name)

    if args.ingest_db:
        if not args.preprocess_docs:
            with open(cv_filepath, "r", encoding="utf-8") as f:
                cvs_eng = json.load(f)

        ingest_db(
            db = cvs_db,
            docs = cvs_eng,
            collection_name = collection_name
        )

    if args.query_db:
        job_desc = "Senior hybrid role in Medellín ($5M–7M COP/month) with 5+ years experience. Responsible for designing, deploying, and maintaining autonomous AI agents and low-code digital workflows using n8n, LangGraph, CrewAI, MCP, RAG, Python, and LLMs (GPT-4, Claude 3.5, Gemini). Connects APIs with legacy CRMs (Bitrix24, Zoho) and establishes productized SaaS formulas. Requires fluent technical English."
        # Querying db
        cvs_db.query_db(
            query= job_desc, collection_name = collection_name
            )

        print(f"cvs_db: {cvs_db}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses cv and job descriptions documents, embeds them in uploads them in a vector database. A sample query to the database is made at the end."
    )
    parser.add_argument(
        "--preprocess-docs",
        type=str2bool,
        default=True,
        help="Whether to preprocess cvs and job descriptions or not (True/False).",
    )
    parser.add_argument(
        "--ingest-db",
        type=str2bool,
        default=True,
        help="Whether to ingest data to the database or not (True/False).",
    )
    parser.add_argument(
        "--query-db",
        type=str2bool,
        default=True,
        help="Whether to make the example query to the database or not (True/False).",
    )

    args = parser.parse_args()

    run_offline_ingestion(args)