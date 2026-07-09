import json
from custom_types import Applicant

def save_to_json(file_name: str, data: list[Applicant]):
    with open(file_name, "w") as file:
        json.dump(data, file, indent=4)