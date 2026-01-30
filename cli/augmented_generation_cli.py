import os
import argparse

from dotenv import load_dotenv
from google import genai

from lib.hybrid_search import (
    rrf_search
)

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            docs, _ = rrf_search(query, 60, 5)

            client = genai.Client(api_key=api_key)
            model = "gemini-2.5-flash"

            prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Query: {query}

            Documents:
            {docs}

            Provide a comprehensive answer that addresses the query:"""
            
            response = client.models.generate_content(model=model, contents=prompt)

            print("Search Results:")
            for r in docs:
                print(f" - {r['title']}")

            print("\nRAG Response:")
            print(response.text)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
