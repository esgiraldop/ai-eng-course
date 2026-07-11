import env_config  # noqa: F401
from task_1.doc_parser import DocsParser
from custom_types import Applicant, Position

if __name__ == "__main__":
    # # Parsing CVs
    # cvs_parser = DocsParser(
    #     response_model= Applicant,
    #     # llm_model="google/gemini-2.5-flash",
    #     llm_model="google/gemini-3.5-flash"
    #     )
    # cvs_data = cvs_parser.gather_docs_info(
    #     dir = "data/CVs",
    #     doc_output = ["name", "email", "phone", "skills", "experience", "education"],
    #     prompt = "Extract the complete applicant information from the CV file."
    #     )
    # cvs_parser.save_extracted_data(
    #     path="data/cv_extracted_info.json",
    #     data = cvs_data
    #     )

    # Parsing "Vacantes"
    positions_parser = DocsParser(
        response_model= Position,
        llm_model="google/gemini-2.5-flash",
        # llm_model="google/gemini-3.5-flash"
        )
    positions_data = positions_parser.gather_docs_info(
        dir = "data/Vacantes",
        doc_output = ["job_title", "seniority_level", "work_arrangement", "salary", "experience_years", "required_skills", "required_education"],
        prompt = "Extract the complete job position information from the file."
        )
    positions_parser.save_extracted_data(
        path="data/positions_extracted_info.json",
        data = positions_data
        )