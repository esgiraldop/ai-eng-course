import env_config  # noqa: F401
from preprocessing.parse_docs import parse_docs

if __name__ == "__main__":
    (cvs_eng, job_pos_eng) = parse_docs()

