import argparse
import json
import os

from lib.search_utils import GOLDEN_DATASET_PATH
from lib.hybrid_search import rrf_search

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    # run evaluation logic here
    with open(GOLDEN_DATASET_PATH, 'r') as f:
        golden_dataset = json.load(f)

    print(f"k={limit}")
    
    for test_case in golden_dataset['test_cases']:
        query = test_case['query']
        relevant_docs = test_case['relevant_docs']

        print(f"- Query: {query}")

        results, _ = rrf_search(query, 60, limit)

        retrieved = []
        relevant_retrieved = []
        for r in results:
            title = r['title']
            retrieved.append(title)
            if title in relevant_docs:
                relevant_retrieved.append(title)
        
        precision = len(relevant_retrieved) / len(results)
        recall = len(relevant_retrieved) / len(relevant_docs)

        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved)}")
        print(f"  - Relevant: {', '.join(relevant_retrieved)}")

if __name__ == "__main__":
    main()
