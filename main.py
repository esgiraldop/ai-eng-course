from db.db import DB
from db.ingest_db import ingest_db
import env_config  # noqa: F401
from preprocessing.parse_docs import parse_docs

if __name__ == "__main__":

    model_name = "microsoft/harrier-oss-v1-0.6b",
    collection_name = "cvs"

    # Initializing database
    cvs_db = DB(model_name)

    (cvs_eng, job_pos_eng) = parse_docs()

    ingest_db(
        db = cvs_db,
        docs = cvs_eng,
        collection_name = collection_name
        )

    job_desc = "Senior hybrid role in Medellín ($5M–7M COP/month) with 5+ years experience. Responsible for designing, deploying, and maintaining autonomous AI agents and low-code digital workflows using n8n, LangGraph, CrewAI, MCP, RAG, Python, and LLMs (GPT-4, Claude 3.5, Gemini). Connects APIs with legacy CRMs (Bitrix24, Zoho) and establishes productized SaaS formulas. Requires fluent technical English."
    # Querying db
    cvs_db.query_db(
        query= job_desc, collection_name = collection_name
        )