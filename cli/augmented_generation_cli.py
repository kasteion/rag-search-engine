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
    rag_parser.add_argument("--limit", type=int, default=5, help="Results limit")

    summarize_parser = subparsers.add_parser("summarize", help="Perform RAG (search + summarize)")
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Results limit")

    citations_parser = subparsers.add_parser("citations", help="Perform RAG (seach + citations)")
    citations_parser.add_argument("query", type=str, help="Search query for RAG")
    citations_parser.add_argument("--limit", type=int, default=5, help="Results limit")

    question_parser = subparsers.add_parser("question", help="Perform RAG (search + answer)")
    question_parser.add_argument("question", type=str, help="Question for RAG")
    question_parser.add_argument("--limit", type=int, default=5, help="Results limit")

    args = parser.parse_args()

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    match args.command:
        case "rag":
            query = args.query
            docs, _ = rrf_search(query, 60, args.limit)

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

        case "summarize":
            query = args.query
            docs, _ = rrf_search(query, 60, args.limit)

            prompt = f"""
            Provide information useful to this query by synthesizing information from multiple search results in detail.
            The goal is to provide comprehensive information so that users know what their options are.
            Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
            This should be tailored to Hoopla users. Hoopla is a movie streaming service.
            Query: {query}
            Search Results:
            {docs}
            Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
            """
            response = client.models.generate_content(model=model, contents=prompt)

            print("Search Results:")
            for r in docs:
                print(f" - {r['title']}")

            print("\nLLM Summary:")
            print(response.text)
        
        case "citations":
            query = args.query
            docs, _ = rrf_search(query, 60, args.limit)

            prompt = f"""Answer the question or provide information based on the provided documents.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

            Query: {query}

            Documents:
            {docs}

            Instructions:
            - Provide a comprehensive answer that addresses the query
            - Cite sources using [1], [2], etc. format when referencing information
            - If sources disagree, mention the different viewpoints
            - If the answer isn't in the documents, say "I don't have enough information"
            - Be direct and informative

            Answer:"""
            response = client.models.generate_content(model=model, contents=prompt)

            print("Search Results:")
            for r in docs:
                print(f" - {r['title']}")

            print("\nLLM Answer:")
            print(response.text)
        
        case "question":
            question = args.question
            docs, search = rrf_search(question, 60, args.limit)
            context = [search.semantic_search.document_map[d['id']] for d in docs]
            
            prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

            This should be tailored to Hoopla users. Hoopla is a movie streaming service.

            Question: {question}

            Documents:
            {context}

            Instructions:
            - Answer questions directly and concisely
            - Be casual and conversational
            - Don't be cringe or hype-y
            - Talk like a normal person would in a chat conversation

            Answer:"""
            response = client.models.generate_content(model=model, contents=prompt)

            print("Search Results:")
            for r in docs:
                print(f" - {r['title']}")

            print("\nLLM Answer:")
            print(response.text)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
