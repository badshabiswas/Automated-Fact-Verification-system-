import pandas as pd
import ast
from nltk.tokenize import sent_tokenize
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import sys
sys.path.append('..')
from config import Config

# Initialize configuration
config = Config()

# 1. Load all Wikipedia articles from JSON file
articles_file_path = config.get_dataset_path('scifact', 'Dev/Relevant Doc/Wikipedia', 'search_results_Negated.json')
with open(articles_file_path, "r") as json_file:
    articles = json.load(json_file)

# Convert the list of articles into a dictionary for fast lookup
articles_dict = {}
for article in articles:
    claim = article.get('claim')
    top_results = article.get('top_results', [])
    contents = [result.get('content', '') for result in top_results]
    articles_dict[claim] = contents

# Extract claims as a list from the loaded articles
all_claims = list(articles_dict.keys())
print(f"Loaded {len(all_claims)} claims.")

# Load all sentences of all articles for each claim into a big list of lists.
claim_sentences = []
for claim in all_claims:
    all_sentences = []
    
    contents = articles_dict.get(claim, [])
    for content in contents:
        if content:
            sentences = sent_tokenize(content)
            sentences = [s.lower() for s in sentences]
            all_sentences.extend(sentences)
    
    claim_sentences.append(all_sentences)

print("Collected all sentences from articles!")

# Load sentence transformer for selecting evidence sentences.
SENTENCE_EMBEDDING_MODEL = 'copenlu/spiced'
model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)

# model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-tas-b")


# Find top 10 sentences for each claim
top_sentences = []

for idx in range(len(all_claims)):
    claim = all_claims[idx]
    sents = claim_sentences[idx]
    
    if sents:
        sents_embeddings = model.encode(sents, convert_to_tensor=True)
        claim_embedding = model.encode(claim, convert_to_tensor=True)
        cos_scores = util.cos_sim(claim_embedding, sents_embeddings)[0]
        top_results = torch.topk(cos_scores, k=10)

        np_results = top_results[1].detach().cpu().numpy()
        top_sentences.append(np_results)
    else:
        top_sentences.append([])

selected_sentences = []
for idx in range(len(all_claims)):
    top = top_sentences[idx]
    if len(top) > 0:
        top = np.sort(top)
        sents = np.array(claim_sentences[idx])[top]    
    else:
        sents = []
    selected_sentences.append(sents)

print("Printing selected_sentences length: ")
print(len(selected_sentences))

# Create a joint list of concatenated claims and evidence
joint_list = []
for idx in range(len(all_claims)):
    joint = all_claims[idx] + " [SEP] "
    joint += " ".join(selected_sentences[idx])
    joint_list.append(joint)

print("Printing joint list length: ")
print(len(joint_list))

# Save this in a file before the final step.
output_file_path = config.get_dataset_path('scifact', 'Dev/Evidence Sentences/Wikipedia', 'SciFact_joint_lines_wiki_negative.txt')
with open(output_file_path, "w") as f:
    for example in joint_list:
        f.write(example)
        f.write("\n")

print("I am done, bro")
