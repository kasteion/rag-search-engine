#!/usr/bin/env python3

import argparse
import json

from lib.keyword_search import keyword_search
from lib.search_utils import print_search_results, DEFAULT_SEARCH_LIMIT
from lib.inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build movies index")

    args = parser.parse_args()

    match args.command:
        case "search":
            print("Searching for:", args.query)
            search_results = keyword_search(args.query, DEFAULT_SEARCH_LIMIT)
            print_search_results(search_results)
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
            docs = index.get_documents('merida') 
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
