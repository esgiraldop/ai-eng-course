import env_config  # noqa: F401
from task_1.doc_parser import DocsParser
from custom_types import Applicant, Position

llm_models = [
    "google/gemini-2.5-flash",
    "google/gemini-3-flash",
    "google/gemini-3.5-flash",
    "groq/llama-3.1-8b-instant",
    "groq/llama-3.3-70b-versatile"
]

if __name__ == "__main__":
    # Parsing CVs
    cvs_parser = DocsParser(
        response_model= Applicant,
        llm_models=llm_models,
        path="data/cv_extracted_info.json"
        )
    cvs_data = cvs_parser.gather_docs_info(
        dir = "data/CVs",
        doc_output = list(Applicant.model_fields.keys()),
        prompt = "Extract the complete applicant information from the CV file.",
        max_docs = -1
        )
    cvs_parser.save_extracted_data(
        data = cvs_data
        )

    # Parsing "Vacantes"
    positions_parser = DocsParser(
        response_model= Position,
        llm_models=llm_models,
        path="data/positions_extracted_info.json",
        )
    positions_data = positions_parser.gather_docs_info(
        dir = "data/Vacantes",
        doc_output = list(Position.model_fields.keys()),
        prompt = "Extract the complete job position information from the file.",
        max_docs = -1
        )
    positions_parser.save_extracted_data(
        data = positions_data
        )