from .search_utils import tokenize_text, BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT
from .inverted_index import InvertedIndex

class KeywordSearch:
    index: InvertedIndex

    def __init__(self, index: InvertedIndex):
        self.index = index

    def keyword_search(self, keyword: str, limit: int = 5) -> list[str]:
        # movies = load_movies()
        query_tokens = tokenize_text(keyword)
        # search_results = [movie for movie in movies if process_text(keyword) in process_text(movie["title"])]
        search_results = []
        # for movie in movies:
        #     title_tokens = tokenize_text(movie["title"])
        #     if has_matching_token(query_tokens, title_tokens):
        #         search_results.append(movie)
        for token in query_tokens:
            search_results.extend(self.index.get_documents(token))
            if len(search_results) >= limit:
                break

        search_results = search_results[:limit]
        search_results.sort(key=id)
        return search_results[:limit]
    
    def bm25_idf_command(self, term:str)->float:
        self.index.load()
        bm25idf = self.index.get_bm25_idf(term)
        return bm25idf
    
    def bm25_tf_command(self, doc_id, term:str, k1=BM25_K1, b=BM25_B):
        self.index.load()
        return self.index.get_bm25_tf(doc_id, term, k1, b)
    
    def bm25_search_command(self, query:str, limit=DEFAULT_SEARCH_LIMIT):
        self.index.load()
        return self.index.bm25_search(query, limit)

 