import json
import os
import string

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def print_search_results(search_results: list[dict]):
    for i in range(len(search_results)):
        print(f"{i + 1}. {search_results[i]["title"]}")

def process_text(text: str) -> str:
    text = text.lower()
    # text = ''.join([c for c in text if c not in string.punctuation])
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text