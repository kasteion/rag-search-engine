#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model, 
    embed_text, 
    verify_embeddings, 
    embed_query_text, 
    search_command, 
    chunk_command,
    semantic_chunk_command,
    embed_chunks_command,
    search_chunked_command
)

from lib.search_utils import (
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_SEMANTIC_CHUNK_OVERLAP,
    DEFAULT_SEARCH_CHUNK_LIMIT
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Verifies semantic search model")

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate text embedding")
    embed_text_parser.add_argument("text", type=str, help="Text to embed")

    subparsers.add_parser("verify_embeddings", help="Verifies embeddings")

    embedquery_parser = subparsers.add_parser("embedquery", help="Embeds query")
    embedquery_parser.add_argument("query", type=str, help="User query to embed")

    search_parser = subparsers.add_parser("search", help="Does a semantic search")
    search_parser.add_argument("query", type=str, help="User query")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Search results limit")

    chunk_parser = subparsers.add_parser("chunk", help="Chunks text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size")
    chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap between chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Chunk text by sentences")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_SEMANTIC_CHUNK_SIZE, help="Sentences per chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_SEMANTIC_CHUNK_OVERLAP, help="Overlap between chunks")

    subparsers.add_parser("embed_chunks", help="Embed chunks by sentences")

    search_chunked_parser = subparsers.add_parser("search_chunked", help="Smantic search on chunked data")
    search_chunked_parser.add_argument("query", type=str, help="query to search for")
    search_chunked_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_CHUNK_LIMIT, help="results limit")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            chunk_command(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks_command()
        case "search_chunked":
            search_chunked_command(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
