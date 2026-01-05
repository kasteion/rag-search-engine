from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
import textwrap

from .search_utils import CACHE_DIR, DATA_PATH, DEFAULT_SEARCH_LIMIT, DEFAULT_CHUNK_SIZE

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
    
    def search(self, query, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        q_embedding = self.generate_embedding(query)

        similarity_scores = []
        for i in range(len(self.embeddings)):
            embedding = self.embeddings[i]
            document = self.documents[i]
            score = cosine_similarity(q_embedding, embedding)
            similarity_scores.append((score, document))

        similarity_scores = sorted(similarity_scores, key=lambda t: t[0], reverse=True)
        
        return similarity_scores[:limit]


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

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search_command(query, limit=DEFAULT_SEARCH_LIMIT):
    search = SemanticSearch()

    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    search.load_or_create_embeddings(data['movies'])

    results = search.search(query, limit)
    for i in range(len(results)):
        score = results[i][0]
        doc = results[i][1]
        print(f"{i+1}. {doc['title']} (score: {score:.4f})")
        print(textwrap.shorten(doc['description'], width=80, placeholder="..."))

def chunk_command(text:str, chunk_size=DEFAULT_CHUNK_SIZE, overlap=0):
    if overlap > chunk_size:
        overlap = chunk_size
        
    words = text.split()
    start, end = 0, chunk_size
    chunks = []
    while end < len(words):
        chunks.append(' '.join(words[start:end]))
        start = end - overlap
        end = start + chunk_size
    
    if end >= len(words):
        chunks.append(' '.join(words[start:]))

    print(f"Chunking {len(text)} characters")
    for i in range(len(chunks)):
        print(f"{i+1}. {chunks[i]}")

def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size

    return chunks
