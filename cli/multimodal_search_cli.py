import argparse

from lib.multimodal_search import (
    verify_image_embedding
)

def main():
    parser = argparse.ArgumentParser(description="Multimodal Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    vie_parser = subparsers.add_parser("verify_image_embedding", help="Verifies image embedding")
    vie_parser.add_argument("path", type=str, help="Image path")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.path)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
