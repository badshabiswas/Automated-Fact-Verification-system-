import os
import json
import pandas as pd
from elasticsearch import Elasticsearch, helpers
import sys
sys.path.append('.')
from config import Config

# Initialize configuration
config = Config()

# Initialize Elasticsearch (credentials should be in environment)
es = Elasticsearch(
    os.getenv('ELASTICSEARCH_URL', "https://127.0.0.1:9200"),
    basic_auth=(os.getenv('ELASTICSEARCH_USER', "elastic"), os.getenv('ELASTICSEARCH_PASSWORD', "your_password")),
    verify_certs=False
)

def search_articles(query, index_name="wikipedia", top_k=10):
    """
    Search indexed articles in Elasticsearch by relevance.
    :param query: Search query string.
    :param index_name: Name of the Elasticsearch index.
    :param top_k: Number of top results to return.
    :return: List of relevant articles.
    """
    response = es.search(
        index=index_name,
        body={
            "query": {
                "match": {
                    "content": query
                }
            },
            "size": top_k
        }
    )

    results = [
        {
            "title": hit["_source"]["title"],
            "url": hit["_source"]["url"],
            "content": hit["_source"]["content"],
            "score": hit["_score"]
        }
        for hit in response["hits"]["hits"]
    ]
    return results

def search_claims_and_save_results(claims, index_name="wikipedia", output_file="search_results.json", top_k=10):
    """
    Search claims in Elasticsearch and save top results to a file.
    :param claims: List of claims to search.
    :param index_name: Name of the Elasticsearch index.
    :param output_file: Path to save the results.
    :param top_k: Number of top results to retrieve per claim.
    """
    results = []
    for idx, claim in enumerate(claims, 1):
        print(f"Processing claim {idx}/{len(claims)}: {claim}")
        top_results = search_articles(claim, index_name=index_name, top_k=top_k)
        results.append({
            "claim": claim,
            "top_results": top_results
        })

    # Save results to a JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved search results to {output_file}")

# Path to your dataset file
file_path = config.get_dataset_path('scifact', 'data', 'dev.csv')

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Extract claims from the DataFrame
claims = df['claim'].tolist()

# Search claims and save results
output_file = config.get_dataset_path('scifact', 'SciFact Relevant Doc/Dev/Wikipedia', 'Original_search_results.json')
search_claims_and_save_results(claims, index_name="wikipedia", output_file=output_file, top_k=10)
