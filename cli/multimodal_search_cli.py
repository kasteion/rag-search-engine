import argparse

from lib.multimodal_search import (
    verify_image_embedding,
    image_search_command
)

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    vie_parser = subparsers.add_parser("verify_image_embedding", help="Verifies image embedding")
    vie_parser.add_argument("path", type=str, help="Image path")

    image_search_parser = subparsers.add_parser("image_search", help="Search by image")
    image_search_parser.add_argument("path", type=str, help="Image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)
        case "image_search":
            results = image_search_command(args.path)
            for i, r in enumerate(results, start=1):
                print(f"{i}. {r['title']} (similarity: {r["similarity_score"]:.3f})")
                print(f"   {r['description'][:100]}\n")
        
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
