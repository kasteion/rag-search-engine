from .search_utils import load_movies

def keyword_search(keyword: str, limit: int = 5) -> list[str]:
    movies = load_movies()
    search_results = [movie for movie in movies if keyword in movie["title"]]
    search_results = search_results[:limit]
    search_results.sort(key=id)
    return search_results[:limit]
