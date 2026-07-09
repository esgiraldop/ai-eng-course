from pydantic import BaseModel

class Applicant(BaseModel):
    name: str
    email: str
    phone: str
    skills: list[str]
    experience: list[str]
    education: list[str]