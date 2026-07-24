# Script for parsing and extracting the data from a series of docs
# and putting it into a JSON file.
import os
import json
import time
from tqdm import tqdm

from google.genai import errors as g_error, types
from google import genai
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
            
            # Groq models expect a 'document' type rather than 'file'
            if "groq" in self.llm_models[0].lower() and format == "pdf":
                import base64
                with open(path, "rb") as f:
                    base64_data = base64.b64encode(f.read()).decode("utf-8")
                content_payload = {
                    "type": "document",
                    "document": {
                        "data": {
                            "url": f"data:application/pdf;base64,{base64_data}"
                        }
                    }
                }
            else:
                content_payload = self.reader[format](path)

            try: 
                response = client.create(
                    response_model=self.response_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                prompt,
                                content_payload
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

    def get_saved_docs(self, filename: str | None = None) -> list[Applicant] | list[Position] | list[dict]:
        target_filename = filename or self.output_filename
        if os.path.exists(target_filename):
            with open(target_filename, "r", encoding="utf-8") as file:
                return json.load(file)
        else:
            print(f"File {target_filename} does not exist, so it's not possible to sync.")
            return []

    def prepare_sync(self, filename: str | None = None) -> tuple[list[dict], set[str]]:
        saved_files_data = self.get_saved_docs(filename)
        if saved_files_data:
            doc_identifiers = set()
            for doc in saved_files_data:
                if isinstance(doc, dict):
                    doc_id = doc.get("file_name") or doc.get("id") or doc.get("doc_id")
                    if doc_id:
                        doc_identifiers.add(doc_id)
            return saved_files_data, doc_identifiers
        else:
            target_filename = filename or self.output_filename
            print(f"There are no saved docs in filename {target_filename}. Skipping sync...")
            return [], set()

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
            data, doc_filenames = self.prepare_sync(self.output_filename)
            if not data:
                sync = False
        for file_name in os.listdir(dir):
            if sync and file_name in doc_filenames:
                print(f"File {file_name} is already saved. Skipping doc parsing...")
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
        print("docs contents exported successfully")

    def process_and_translate_docs(
        self, output_filename: str, max_docs: int = -1, sync: bool = True,
        **kwargs: str | list[Applicant] | list[Position]
        ):

        if max_docs == 0:
            raise ValueError("max_docs cannot be zero.")

        if "file_path" in kwargs and "docs" in kwargs:
            raise TypeError("You cannot provide arguments 'file_path' and 'docs' at the same time.")

        if "file_path" in kwargs:
            with open(kwargs["file_path"], "r", encoding="utf-8") as f:
                docs = json.load(f)

            print(f"Loaded {len(docs)} docs from '{kwargs['file_path']}'. Starting processing...")

        if "docs" in kwargs:
            # For using after parsing the docs as PDFs and before saving the json file
            docs = kwargs["docs"]

        processed_docs = []
        doc_identifiers = set()
        if sync:
            processed_docs, doc_identifiers = self.prepare_sync(output_filename)
            if not processed_docs:
                sync = False

        docs_counter = 0

        for doc in tqdm(docs, desc="Translating docs"):
            doc_id = doc.get("file_name") or doc.get("id") or doc.get("doc_id") or "unknown"
            if sync and doc_id != "unknown" and doc_id in doc_identifiers:
                print(f"Doc '{doc_id}' is already saved. Skipping translation...")
                continue

            prompt = f"""
            You are a professional multilingual translator specializing in technical resumes.
            Analyze the provided doc JSON object:

            1. Detect if the text is primarily in 'en' (English) or 'es' (Spanish).
            2. If 'es', translate all text values into clear, professional English while strictly preserving the original JSON schema, keys, and numerical data.
            3. If 'en', return the JSON content as-is (with minor grammar polishing if necessary).
            4. Maintain exact technical terms (e.g., "FastAPI", "AWS", "Docker", "Python").

            Input doc JSON:
            {json.dumps(doc, ensure_ascii=False, indent=2)}
            """

            client = genai.Client()

            try:
                # We enforce raw JSON output matching the original structure
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,  # Low temperature for strict factual translation
                    )
                )

                # Parse translated JSON response
                translated_doc_data = json.loads(response.text)
                
                processed_docs.append(translated_doc_data)

            except Exception as e:
                print(f"\n[Error] Failed to process doc ID '{doc_id}': {e}. Appending original.")
                processed_docs.append(doc)

            docs_counter += 1
            if (max_docs > -1 and docs_counter == max_docs):
                break

            # Brief rate limit pause
            time.sleep(0.2)

        # Save standardized English JSON array
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(processed_docs, f, ensure_ascii=False)

        print(f"\nCompleted! Standardized English doc dataset saved to '{output_filename}'.")
        