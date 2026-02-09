import json

from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from .search_utils import DATA_PATH

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32", documents = list[map]):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)
    
    def embed_image(self, path):
        try:
            img = Image.open(path)
            embeddings = self.model.encode([img])
            return embeddings[0]
        except Exception as e:
            raise e
    
    def search_with_image(self, path: str):
        image_embedding = self.embed_image(path)

        similarities: list[dict] = []
        for i, text_embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(text_embedding, image_embedding)
            similarities.append(self.documents[i])
            similarities[i]['similarity_score'] = similarity
        
        similarities = sorted(similarities, key=lambda s: s['similarity_score'], reverse=True)
        return similarities[:5]


def verify_image_embedding(path):
    search = MultimodalSearch()
    embedding = search.embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(path):
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    search = MultimodalSearch(documents=data["movies"])
    return search.search_with_image(path)
