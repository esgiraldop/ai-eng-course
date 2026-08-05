from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
import os

class DbIngester:

    HUGGING_FACE_API_KEY = os.getenv("HUGGING_FACE_API_KEY")

    def __init__(self):
        print(self)

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

    def embed_cvs(self, model_name: str):

        model = SentenceTransformer(model_name, token=self.HUGGING_FACE_API_KEY)