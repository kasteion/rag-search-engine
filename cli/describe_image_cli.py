import os
import argparse
import mimetypes

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")

def main():
    parser = argparse.ArgumentParser(description="Multimodal CLI")

    parser.add_argument("--image", type=str, help="The path to an image file")
    parser.add_argument("--query", type=str, help="Text query to rewrite based on the image")

    args = parser.parse_args()

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-flash"

    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    with open(args.image, "rb") as f:
        img = f.read()
    
    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary"""

    parts = [
        system_prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        args.query.strip(),
    ]

    response = client.models.generate_content(model=model, contents=parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")

if __name__ == "__main__":
    main()
