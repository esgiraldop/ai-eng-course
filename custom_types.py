from pydantic import BaseModel
from typing import Literal

class Applicant(BaseModel):
    name: str
    email: str
    phone: str
    skills: list[str]
    experience: list[str]
    education: list[str]

type ApplicantField = Literal[*Applicant.model_fields.keys()]  # type: ignore