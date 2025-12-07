from .search_utils import load_movies, tokenize_text, has_matching_token

def keyword_search(keyword: str, limit: int = 5) -> list[str]:
    movies = load_movies()
    query_tokens = tokenize_text(keyword)
    # search_results = [movie for movie in movies if process_text(keyword) in process_text(movie["title"])]
    search_results = []
    for movie in movies:
        title_tokens = tokenize_text(movie["title"])
        if has_matching_token(query_tokens, title_tokens):
            search_results.append(movie)
    search_results = search_results[:limit]
    search_results.sort(key=id)
    return search_results[:limit]
