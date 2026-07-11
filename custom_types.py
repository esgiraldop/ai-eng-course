from pydantic import BaseModel
from typing import Literal

class Applicant(BaseModel):
    name: str
    email: str
    phone: str
    skills: list[str]
    experience: list[str]
    education: list[str]

class Position(BaseModel):
    company_name: str
    job_title: str
    role_description: str
    seniority_level: str
    work_arrangement: str
    location: str
    salary: str
    experience_years: str
    required_skills: list[str]
    required_education: list[str]

type ApplicantField = Literal[*Applicant.model_fields.keys()]  # type: ignore
type PositionField = Literal[*Position.model_fields.keys()]  # type: ignore