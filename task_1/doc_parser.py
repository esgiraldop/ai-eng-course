# Script for parsing and extracting the data from a series of CVs
# and putting it into a JSON file.
import os
from google.genai import errors as g_error
from groq import APIError
from instructor.core import InstructorRetryException

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
        response_model: Applicant | Position,
        llm_models: list[str]
    ):
        self.response_model = response_model
        self.llm_models = llm_models

    def parse_doc(self, path: str, format: str, prompt) -> Applicant | Position:
        
        while len(self.llm_models) > 0:
            client = instructor.from_provider(self.llm_models[0])
            try: 
                response = client.create(
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
            except (
                g_error.APIError, APIError, InstructorRetryException
                ) as e:
                print(f"{type(e).__name__}: An error occurred when trying to use the llm model {self.llm_models[0]}: {e}\n")
                self.llm_models.pop(0)
                if len(self.llm_models) == 0:
                    print("I ran out of models. Aborting the task.\n")
                    return
                else:
                    print(f"Switched to model{self.llm_models[0]}\n")
            else:
                return response

    def gather_docs_info(self, dir: str, doc_output: list[ApplicantField] | list[PositionField], prompt: str, max_docs: int) -> list[Applicant] | list[Position]:
        if max_docs == 0:
            raise ValueError("max_docs cannot be zero.")
        data = []
        counter = 0
        num_files = len(os.listdir(dir))
        for file_name in os.listdir(dir):
            if len(self.llm_models) == 0:
                print("I ran out of models. Aborting the task.\n")
                return
            if not file_name.endswith(".pdf"):
                # TODO: Add support for files other than pdf
                continue
            response = self.parse_doc(
                path=os.path.join(dir, file_name), 
                format="pdf",
                prompt=prompt
                )
            if not response:
                return
            data.append({field: getattr(response, field) for field in doc_output})

            counter += 1
            print(f"File {file_name} was parsed. {counter}/{num_files} files parsed.")
            if (max_docs > -1 and counter == max_docs):
                break
        return data

    def save_extracted_data(self, path: str, data: list[Applicant]):
        print("\nExporting files to json format...\n")
        save_to_json(path, data)
        print("CVs contents exported successfully")
        