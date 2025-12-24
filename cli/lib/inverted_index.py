import os
import pickle
import math
from collections import Counter, defaultdict

from .search_utils import (
    tokenize_text, 
    load_movies,
    CACHE_DIR,
    BM25_K1,
    BM25_B
)

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.tf_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

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

        with open(self.doc_lengths_path, 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        # if not os.path.exists(self.index_path) or not os.path.exists(self.docmap_path) or not os.path.exists(self.tf_path):
            # raise Exception("Files don't exist, build the index first!")
        
        with open(self.index_path, 'rb') as f:
            self.index = pickle.load(f)
                
        with open(self.docmap_path, 'rb') as f:
            self.docmap = pickle.load(f)
        
        with open(self.tf_path, 'rb') as f:
            self.term_frequencies = pickle.load(f)
        
        with open(self.doc_lengths_path, 'rb') as f:
            self.doc_lengths = pickle.load(f)
    
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

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = 1 - b + b * (doc_length / avg_doc_length)
        # return (tf * (k1 + 1)) / (tf + k1)
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return tf_component


    def __get_avg_doc_length(self) -> float:
        if len(self.doc_lengths) == 0:
            return 0.0
        
        return sum([self.doc_lengths[i] for i in self.doc_lengths]) / len(self.doc_lengths)
    
    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)
        scores= {}
        for doc_id in self.docmap:
            bm25_total = 0
            for term in tokens:
                bm25_total += self.bm25(doc_id, term)
            scores[doc_id] = bm25_total

        scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        results = []
        for doc_id, score in scores[:limit]:
            results.append({ 'id': doc_id, 'title': self.docmap[doc_id]['title'], 'score': score})
        return results