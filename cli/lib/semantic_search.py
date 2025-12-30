from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from .search_utils import CACHE_DIR, DATA_PATH

class SemanticSearch:
    def __init__(self)->None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")
    
    def generate_embedding(self, text:str):
        if len(text.strip()) == 0:
            raise ValueError("Input text is empty")
        
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents):
        self.documents = documents
        sentences = []
        for doc in documents:
            self.document_map[doc['id']] = doc
            sentences.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(sentences, show_progress_bar=True)
        with open(self.embeddings_path, 'wb') as f:
            np.save(f, self.embeddings)
        
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc['id']] = doc

        if os.path.exists(self.embeddings_path):
            with open(self.embeddings_path, 'rb') as f:
                self.embeddings = np.load(f)
            if len(self.embeddings) == len(documents):
                return self.embeddings
            
        return self.build_embeddings(documents)

def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text):
    search = SemanticSearch()
    embedding = search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search = SemanticSearch()
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    embeddings = search.load_or_create_embeddings(data['movies'])
    print(f"Number of docs:   {len(data['movies'])}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    search = SemanticSearch()
    embedding = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")