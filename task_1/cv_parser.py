# Script for parsing and extracting the data from a series of CVs
# and putting it into a JSON file.
import os

import instructor
from instructor.processing.multimodal import PDF
from custom_types import Applicant
from utils import save_to_json


# client = instructor.from_provider("google/gemini-2.5-flash")
client = instructor.from_provider("google/gemini-3.5-flash")


class CVs_parser:
    def parse_cv(self, path: str) -> Applicant:
        return client.create(
            response_model=Applicant,
            messages=[
                {
                    "role": "user",
                    "content": [
                        "Extract the complete applicant information from the CV file.",
                        PDF.from_path(path)
                    ]
                }
            ]
        )

    def gather_cvs_info(self, dir: str):
        data = []
        counter = 0
        num_files = len(os.listdir(dir))
        for file_name in os.listdir(dir):
            if not file_name.endswith(".pdf"):
                # TODO: Add support for files other than pdf
                continue
            response = self.parse_cv(os.path.join(dir, file_name))
            counter += 1
            data.append({
                "name": response.name,
                "email": response.email,
                "phone": response.phone,
                "skills": response.skills,
                "experience": response.experience,
                "education": response.education,
            })
            print(f"{counter} out of {num_files} files parsed")

        print("\nExporting files to json format...\n")
        save_to_json("data/cv_extracted_info.json", data)
        print("CVs contents exported successfully")
        