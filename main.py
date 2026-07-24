import env_config  # noqa: F401
from task_1.doc_parser import DocsParser
from custom_types import Applicant, Position

llm_models = [
    "google/gemini-2.5-flash",
    "google/gemini-3.5-flash",
    "google/gemini-2.5-pro"
]

if __name__ == "__main__":
    # Parsing CVs
    output_filename = "data/cv_extracted_info.json"
    cvs_parser = DocsParser(
        response_model= Applicant,
        llm_models=llm_models,
        path=output_filename
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
    # Loading cvs and translating to English
    cvs_parser.process_and_translate_docs(
        output_filename = "data/cv_extracted_info_eng.json",
        max_docs = 2,
        file_path = output_filename
    )

    # Parsing "Vacantes"
    positions_parser = DocsParser(
        response_model= Position,
        llm_models=llm_models,
        path="data/job_descriptions_extracted_info.json",
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