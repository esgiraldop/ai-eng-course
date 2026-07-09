# Script for parsing and extracting the data from a series of CVs
# and putting it into a JSON file.

import instructor
from pydantic import BaseModel, Field
from instructor.processing.multimodal import PDF

import os

class Applicant(BaseModel):
    name: str
    email: str
    phone: str
    skills: list[str]
    experience: list[str]
    education: list[str]

client = instructor.from_provider("google/gemini-2.5-flash")

def parse_cv():
    print(f"CWD: ${os.getcwd()}")
    return client.create(
        response_model=Applicant,
        messages=[
            {
                "role": "user",
                "content": [
                    "Extract the complete applicant information from the CV file.",
                    PDF.from_path("data/CVs/20240217 HOJA DE VIDA FREDY VASQUEZ (1).pdf")
                ]
            }
        ]
    )