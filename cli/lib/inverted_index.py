import os
import pickle
from .search_utils import tokenize_text, load_movies

class InvertedIndex:
    index: dict = {}
    docmap: dict = {}

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for token in tokens:
            ids:set = self.index.get(token)

            if ids:
                ids.add(doc_id)
            else:
                ids = set([doc_id])
            
            self.index[token] = ids

    def get_documents(self, term: str):
        term = term.lower()
        ids = self.index.get(term)

        if not ids:
            return []
        
        return sorted(ids)
    
    def build(self):
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie['id'], f"{movie['title']} f{movie['description']}")
            self.docmap[movie['id']] = movie

    def save(self):
        os.makedirs('cache', exist_ok=True)

        with open('cache/index.pkl', 'wb') as f:
            pickle.dump(self.index, f)
        
        with open('cache/docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)
