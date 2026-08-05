from preprocessing.doc_parser import DocsParser
from custom_types import Applicant, Position

llm_models = [
    "google/gemini-2.5-flash",
    "google/gemini-3.5-flash",
    "google/gemini-2.5-pro"
]

def parse_docs():
    cvs_out_path = "data/cv_extracted_info.json"
    cvs_eng_out_path = "data/cv_extracted_info_eng.json"
    pos_out_path = "data/job_descriptions_extracted_info.json"
    pos_eng_out_path = "data/job_descriptions_extracted_info_eng.json"

    cvs_parser = DocsParser(
        response_model= Applicant,
        llm_models=llm_models,
        path=cvs_out_path
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
    cvs_eng = cvs_parser.process_and_translate_docs(
        output_filename = cvs_eng_out_path,
        file_path = cvs_out_path
    )
    cvs_parser.save_extracted_data(data = cvs_eng, filename = cvs_eng_out_path)

    
    # Parsing "Vacantes"
    positions_parser = DocsParser(
        response_model= Position,
        llm_models=llm_models,
        path=pos_out_path,
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
    # Loading job descriptions and translating to English
    job_pos_eng = positions_parser.process_and_translate_docs(
        output_filename = pos_eng_out_path,
        file_path = pos_out_path
    )
    positions_parser.save_extracted_data(data = job_pos_eng, filename = pos_eng_out_path)

    return (cvs_eng, job_pos_eng)