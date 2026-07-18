# Script for parsing and extracting the data from a series of CVs
# and putting it into a JSON file.
import os
import json
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
        llm_models: list[str],
        path: str
    ):
        self.response_model = response_model
        self.llm_models = llm_models
        self.output_filename = path

    def parse_doc(self, path: str, format: str, prompt: str) -> Applicant | Position:
        
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

    def get_saved_docs(self) -> list[Applicant] | list[Position]:
        if os.path.exists(self.output_filename):
            with open(self.output_filename, "r") as file:
                return json.load(file)
        else:
            print(f"File {self.output_filename} does not exists, so it's not possible to sync.")
            return []

    def gather_docs_info(
        self, dir: str, doc_output: list[ApplicantField] | list[PositionField],
        prompt: str, max_docs: int, sync: bool = True
        ) -> list[Applicant] | list[Position]:
        if max_docs == 0:
            raise ValueError("max_docs cannot be zero.")
        data = []
        counter = 0
        num_files = len(os.listdir(dir))
        if sync:
            saved_files_data = self.get_saved_docs()
            if saved_files_data:
                data = saved_files_data
                doc_filenames = [doc["file_name"] for doc in data if "file_name" in doc]
            else:
                print(f"There are not saved cvs in filename {self.output_filename}. Skipping sync...")
                sync = False
        for file_name in os.listdir(dir):
            if sync and file_name in doc_filenames:
                print(f"File {file_name} is already saved. Skipping cv parsing...")
                continue
            if len(self.llm_models) == 0:
                print("I ran out of models. Aborting the task.\n")
                return data
            if not file_name.endswith(".pdf"):
                # TODO: Add support for files other than pdf
                continue
            response = self.parse_doc(
                path=os.path.join(dir, file_name), 
                format="pdf",
                prompt=prompt
                )
            if not response:
                return data
            doc_info = {field: getattr(response, field) for field in doc_output}
            data.append({
                **doc_info,
                "file_name": file_name # for sync capabilities
                })

            counter += 1
            print(f"File {file_name} was parsed. {counter}/{num_files} files parsed.")
            if (max_docs > -1 and counter == max_docs):
                break
        return data

    def save_extracted_data(self, data: list[Applicant]):
        print("\nExporting files to json format...\n")
        save_to_json(self.output_filename, data)
        print("CVs contents exported successfully")
        