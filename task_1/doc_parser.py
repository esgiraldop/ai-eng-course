# Script for parsing and extracting the data from a series of CVs
# and putting it into a JSON file.
from annotated_types import doc
import os
from pydantic import BaseModel

import instructor
from instructor.processing.multimodal import PDF
from custom_types import Applicant, Position, ApplicantField, PositionField
from utils import save_to_json

class DocsParser:

    reader = {
        "pdf": PDF.from_path,
        "docx": "docx extractor",
        "txt": "text extractor"
    }

    def __init__(
        self,
        response_model: Applicant | BaseModel, #TODO: replace BaseModel by the positions' model,
        llm_model
    ):
        self.response_model = response_model
        self.client = instructor.from_provider(llm_model)


    def parse_doc(self, path: str, format: str, prompt) -> Applicant | Position:
        return self.client.create(
            response_model=self.response_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        prompt,
                        self.reader[format](path)
                    ]
                }
            ]
        )

    def gather_docs_info(self, dir: str, doc_output: list[ApplicantField] | list[PositionField], prompt: str) -> list[Applicant] | list[Position]:
        data = []
        counter = 0
        num_files = len(os.listdir(dir))
        for file_name in os.listdir(dir):
            if not file_name.endswith(".pdf"):
                # TODO: Add support for files other than pdf
                continue
            response = self.parse_doc(
                path=os.path.join(dir, file_name), 
                format="pdf",
                prompt=prompt
                )
            counter += 1
            data.append({field: getattr(response, field) for field in doc_output})

            print(f"File {file_name} was parsed. {counter}/{num_files} files parsed.")
            if counter == 3:
                # TODO: for debugging purposes
                break
        return data

    def save_extracted_data(self, path: str, data: list[Applicant]):
        print("\nExporting files to json format...\n")
        save_to_json(path, data)
        print("CVs contents exported successfully")
        