import argparse

from lib.hybrid_search import (
    normalize_scores_command,
    weighted_search_command,
    rrf_search_command,
)
from lib.search_utils import (
    DEFAULT_HYBRID_SEARCH_ALPHA,
    DEFAULT_HYBRID_SEARCH_LIMIT
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Normalize scores with min-max normalization")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to normalize")

    weighted_search_parser = subparsers.add_parser("weighted-search", help="Hybrid search")
    weighted_search_parser.add_argument("query", type=str, help="Query to search")
    weighted_search_parser.add_argument("--alpha", type=float, default=DEFAULT_HYBRID_SEARCH_ALPHA, help="Configurable alpha")
    weighted_search_parser.add_argument("--limit", type=int, default=DEFAULT_HYBRID_SEARCH_LIMIT, help="Results limit")

    rrf_search_parser = subparsers.add_parser("rrf-search", help="Reciprocal Rank Fusion")
    rrf_search_parser.add_argument("query", type=str, help="Query to search")
    rrf_search_parser.add_argument("-k", type=int, default=60, help="k constant")
    rrf_search_parser.add_argument("--limit", type=int, default=5, help="Results limit")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_scores_command(args.scores)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
