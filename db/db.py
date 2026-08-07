from custom_types import Applicant
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import os

class DB:

    HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

    def __init__(self, model_name: str = "avsolatorio/GIST-all-MiniLM-L6-v2"):
        self.client = QdrantClient(":memory:")
        self.model = SentenceTransformer(model_name, token=self.HUGGING_FACE_API_KEY, trust_remote_code=True)

    def serialize_cv_for_embedding(self, cv: dict) -> str:
        """Formats CV text specifically for optimal embedding model vectorization."""
        
        certifications = ", ".join(cv.get("certifications", [])) or "None"
        skills = ", ".join(cv.get("skills", [])) or "None"
        languages = ", ".join(cv.get("languages", [])) or "None"
        
        experience = "\n".join([f"- {exp}" for exp in cv.get("experience", [])])
        education = "\n".join([f"- {edu}" for edu in cv.get("education", [])])

        text = f"""Candidate Name: {cv.get("name")}
        Profession: {cv.get("profession")}
        Seniority Level: {cv.get("seniority_level")}
        Location: {cv.get("location")}
        Years of Experience: {cv.get("experience_years")}
        Languages: {languages}

        About Me:
        {cv.get("about_me", "")}

        Skills:
        {skills}

        Certifications:
        {certifications}

        Work Experience:
        {experience}

        Education:
        {education}
        """.strip()
        
        return text

    def embed_docs(self, docs: list[Applicant]) -> list[list[float]]:

        data_points = []

        # No chunking applied since according to task_3/script_2.ipynb
        #   no chunking is the strategy with the best metrics
        for cv in docs:
            cv_text = self.serialize_cv_for_embedding(cv)
            data_points.append({
                "chunk": cv_text,
                "cv": cv
            })

        return (data_points, self.model.encode([dp["chunk"] for dp in data_points]).tolist())

    def upload_db_points(self, data_points: list[Applicant], embeddings: list[list[float]], collection_name: str):
        
        self.client.upload_points(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=idx,
                    vector=embeddings[idx],
                    # Store the original dictionary (to keep queryable fields) and the chunked text (useful for LLM context later)
                    payload={
                        **data_point['cv'],
                        "chunk": data_point['chunk']
                    }
                )
                for idx, data_point in enumerate(data_points)
            ],
        )

    def query_db(self, query: str, collection_name: str, k:int = 10):
        
        result = self.client.query_points_groups(
            collection_name=collection_name,
            query=self.model.encode(query).tolist(),
            limit=k,
            # The two parameters below are used when the data points are chunked
            #   and the search groups by a field, in this case the email.
            # group_by="email",
            # group_size=1
        )

        return [group["hits"][0] for group in result.model_dump()["groups"]]