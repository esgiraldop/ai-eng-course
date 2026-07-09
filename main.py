import env_config  # noqa: F401
from task_1.cv_parser import CVs_parser

if __name__ == "__main__":
    cvs_parser = CVs_parser()
    cvs_parser.gather_cvs_info("data/CVs")
