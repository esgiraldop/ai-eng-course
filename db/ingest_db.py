from custom_types import Applicant

def ingest_db(db: str, docs: list[Applicant], collection_name: str):
    data_points, embeddings = db.embed_docs(docs)
    db.upload_db_points(data_points, embeddings, collection_name)
