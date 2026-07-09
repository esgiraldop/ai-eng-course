import env_config  # noqa: F401
from task_1.cv_parser import parse_cv

if __name__ == "__main__":
    response = parse_cv()
    print(response)
