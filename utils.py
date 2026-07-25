import json
from custom_types import Applicant

def save_to_json(file_name: str, data: list):
    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)