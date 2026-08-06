from custom_types import Applicant
from db.db import DB

def ingest_db(db: str, docs: list[Applicant], collection_name: str):
    embeddings = db.embed_docs(docs)
    db.upload_db_points(docs, embeddings, collection_name)

