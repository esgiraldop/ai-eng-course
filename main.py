import env_config  # noqa: F401
from task_1.doc_parser import DocsParser
from custom_types import Applicant

if __name__ == "__main__":
    DocsParser = DocsParser(
        response_model= Applicant,
        # llm_model="google/gemini-2.5-flash",
        llm_model="google/gemini-3.5-flash"
        )
    cvs_data = DocsParser.gather_docs_info(
        dir = "data/CVs",
        doc_output = ["name", "email", "phone", "skills", "experience", "education"],
        prompt = "Extract the complete applicant information from the CV file."
        )
    DocsParser.save_extracted_data(
        path="data/cv_extracted_info.json",
        data = cvs_data
        )