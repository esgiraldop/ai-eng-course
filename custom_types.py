from pydantic import BaseModel
from typing import Literal

class Applicant(BaseModel):
    name: str
    email: str
    linkedin: str
    phone: str
    about_me: str
    profession: str
    seniority_level: str
    location: str
    experience_years: str
    certifications: list[str]
    skills: list[str]
    experience: list[str]
    education: list[str]
    languages: list[str]

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
    required_languages: list[str]

type ApplicantField = Literal[*Applicant.model_fields.keys()]  # type: ignore
type PositionField = Literal[*Position.model_fields.keys()]  # type: ignore