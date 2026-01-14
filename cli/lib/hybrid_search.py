import os
import json

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import DATA_PATH


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_embeddings(documents)
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        keyword_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search(query, limit * 500)
        semantic_results = [{'id': r['id'], 'title': r['title'], 'description': r['description'][:100], 'score': s} for s, r in semantic_results]

        # combined_scores = []
        # combined_scores.extend([r['score'] for r in keyword_results])
        # combined_scores.extend([r['score'] for r in semantic_results])
        
        # normalized_scores = normalize_scores(combined_scores)
        
        keyword_scores = normalize_scores([r['score'] for r in keyword_results])
        semantic_scores = normalize_scores([r['score'] for r in semantic_results])


        weighted_results = {}
        for i, r in enumerate(keyword_results):
            id = r['id']
            score = keyword_scores[i]
            if id not in weighted_results:
                weighted_results[id] = {
                    'title': r['title'],
                    'description': r['description'],
                    'hybrid_score': 0.0,
                    'bm25_score': 0.0,
                    'semantic_score': 0.0
                }
            if score > weighted_results[id]['bm25_score']:
                weighted_results[id]['bm25_score'] = score
        
        for i, r in enumerate(semantic_results):
            id = r['id']
            score = semantic_scores[i]
            if id not in weighted_results:
                weighted_results[id] = {
                    'title': r['title'],
                    'description': r['description'],
                    'hybrid_score': 0.0,
                    'bm25_score': 0.0,
                    'semantic_score': 0.0
                }
            if score > weighted_results[id]['semantic_score']:
                weighted_results[id]['semantic_score'] = score
        
        hybrid_results = []
        for k, v in weighted_results.items():
            hybrid_results.append({
                'id': k,
                'title': v['title'],
                'description': v['description'],
                'hybrid_score': hybrid_score(v['bm25_score'], v['semantic_score'], alpha),
                'bm25_score': v['bm25_score'],
                'semantic_score': v['semantic_score']
            })
            # weighted_results[k]['hybrid_score'] = hybrid_score(v['bm25_score'], v['semantic_score'], alpha)
        
        combined_results = sorted(hybrid_results, key=lambda x: x["hybrid_score"], reverse=True)
        return combined_results[:limit]

    def rrf_search(self, query, k, limit=10):
        keyword_results = self._bm25_search(query, limit * 500)
        # semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        semantic_results = self.semantic_search.search(query, limit * 500)
        semantic_results = [{'id': r['id'], 'title': r['title'], 'description': r['description'][:100], 'score': s} for s, r in semantic_results]

        ranked_results = {}
        for rank, r in enumerate(keyword_results, start=1):
            id = r['id']
            s = rrf_score(rank, k)
            if id not in ranked_results:
                ranked_results[id] = {'title': r['title'], 'description': r['description'], 'bm25_rank': rank, 'rrf_score': s}
            
        for rank, r in enumerate(semantic_results, start=1):
            id = r['id']
            s = rrf_score(rank, k)
            if id not in ranked_results:
                ranked_results[id] = {'title': r['title'], 'description': r['description'], 'semantic_rank': rank, 'rrf_score': s}
            else:
                ranked_results[id]['semantic_rank'] = rank
                ranked_results[id]['rrf_score'] = ranked_results[id]['rrf_score'] + rrf_score(rank, k)

        combined_results = []
        for k, v, in ranked_results.items():
            combined_results.append({
                'id': k,
                'title': v['title'],
                'description': v['description'],
                'bm25_rank': v.get  ('bm25_rank'),
                'semantic_rank': v.get('semantic_rank'),
                'rrf_score': v['rrf_score']
            })
        
        sorted_results = sorted(combined_results, key=lambda r: r["rrf_score"], reverse=True)
        return sorted_results[:limit]

def normalize_scores_command(scores: list[float]):
    normalized_scores = normalize_scores(scores)
    for score in normalized_scores:
        print(f"* {score:.4f}")

def normalize_scores(scores: list[float]):
    if not scores:
        return []

    max_score = None
    min_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score
        if min_score is None or score < min_score:
            min_score = score
    
    if min_score == max_score:
        return [1.0 for _ in scores]

    return [(s - min_score)/(max_score - min_score) for s in scores]

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def weighted_search_command(query:str, alpha:float, limit:int):
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    search = HybridSearch(data['movies'])

    results = search.weighted_search(query, alpha, limit)
    for i, r in enumerate(results, start=1):
        print(f"{i}. {r['title']}")
        print(f"    Hybrid Score: {r['hybrid_score']}")
        print(f"    BM25: {r['bm25_score']}, Semantic: {r['semantic_score']}")
        print(f"    {r['description']}")

    # for i, (k, v) in enumerate(results.items(), start=1):
    #     print(f"{i}. {v['title']}")
    #     print(f"   Hybrid Score: {v['hybrid_score']}")
    #     print(f"   BM25: {v['bm25_score']}, Semantic: {v['semantic_score']}")
    #     print(f"   {v['description']}")
        # print(i, k, v)

def rrf_score(rank, k=60):
    return 1 / (k + rank)

def rrf_search_command(query: str, k: int, limit:int):
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)
    
    search = HybridSearch(data['movies'])

    results = search.rrf_search(query, k, limit)

    for i, r in enumerate(results, start=1):
        print(f"{i}. {r['title']}")
        print(f"     RRF Score: {r['rrf_score']}")
        print(f"     BM25 Rank: {r['bm25_rank']}, Semantic: {r['semantic_rank']}")
        print(f"     {r['description']}")
