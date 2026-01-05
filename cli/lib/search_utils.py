import json
import os
import string
from nltk.stem import PorterStemmer

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0

stemmer = PorterStemmer()

def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        data = f.read().splitlines()
    return data

def print_search_results(search_results: list[dict]):
    for i in range(len(search_results)):
        print(f"{i + 1}. {search_results[i]["id"]} {search_results[i]["title"]}")

def preprocess_text(text: str) -> str:
    text = text.lower()
    # text = ''.join([c for c in text if c not in string.punctuation])
    text = text.translate(str.maketrans("", "", string.punctuation))
    # tokens = [token.strip() for token in text.split(" ") if len(token.strip()) > 0]
    return text

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []
    for token in tokens:
        if token:
            valid_tokens.append(token)
    stop_words = load_stopwords()
    filtered_words = []
    for word in valid_tokens:
        if word not in stop_words:
            filtered_words.append(word)
    stemmer = PorterStemmer()
    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def has_matching_token(query_tokens: list[str], title_tokens: list[str]) -> bool:
    for qt in query_tokens:
        for tt in title_tokens:
            if qt in tt:
                return True
    
    return False
