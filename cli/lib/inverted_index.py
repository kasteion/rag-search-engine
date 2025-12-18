import os
import pickle
import math
from collections import Counter, defaultdict

from .search_utils import (
    tokenize_text, 
    load_movies,
    CACHE_DIR,
    BM25_K1
)

class InvertedIndex:
    index: dict = {}
    docmap: dict = {}
    term_frequencies: dict = {}

    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies = defaultdict(Counter)

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_documents(self, term: str):
        term = term.lower()
        ids = self.index.get(term)

        if not ids:
            return []
        
        return [self.docmap.get(id) for id in sorted(ids)]
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie['id'], f"{movie['title']} f{movie['description']}")
            self.docmap[movie['id']] = movie

    def save(self):
        os.makedirs('cache', exist_ok=True)

        with open(self.index_path, 'wb') as f:
            pickle.dump(self.index, f)
        
        with open(self.docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)

        with open(self.tf_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.docmap_path) or not os.path.exists(self.tf_path):
            raise Exception("Files don't exist, build the index first!")
        
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
                
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
        
        with open(self.tf_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)
    
    def get_tf(self, doc_id, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        
        doc:dict = self.term_frequencies.get(doc_id)
        if not doc:
            return 0
        
        term = doc.get(tokens[0])
        if not term:
            return 0
        
        return term

    def get_idf(self, term):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")
        
        return math.log((len(self.docmap) + 1) / (len(self.index.get(tokens[0], [])) + 1))

    def get_bm25_idf(self, term:str):
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("term must be a single token")\
        
        N = len(self.docmap)
        df = len(self.index.get(tokens[0], []))
        
        return math.log((N - df + 0.5)/(df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1):
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1)
