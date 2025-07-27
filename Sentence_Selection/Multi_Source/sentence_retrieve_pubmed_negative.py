import pandas as pd
import ast
from nltk.tokenize import sent_tokenize
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import sys
sys.path.append('..')
from config import Config

# Initialize configuration
config = Config()

# Path to the claims text file
claims_file_path = config.get_dataset_path('scifact', 'Dev', 'negated_claims_dev.txt')

# Read the claims from the text file (each line contains a claim)
with open(claims_file_path, "r") as file:
    all_claims = [line.strip() for line in file.readlines()]



# 2. Load the data using ast.literal_eval for non-standard JSON (like single quotes)
pmid_file_path = config.get_dataset_path('scifact', 'Dev/Relevant Doc/Pubmed', 'bm25_Pubhealth_pmids_2024_Negative.txt')
with open(pmid_file_path, 'r') as file:
    data = file.read()
    results = ast.literal_eval(data)

# Organize PMIDs by claim (list of lists)
all_ids = [list(pmid_scores.keys()) for pmid_scores in results.values()]
print(f"Loaded PMIDs for {len(all_ids)} claims.")

# 3. Load all PubMed abstracts
abstracts_path = os.path.join(config.BASE_DATA_DIR, "pubmed_landscape_abstracts_2024.csv")
abstracts = pd.read_csv(abstracts_path)


# Ensure the PMID column exists and is of the correct type
abstracts['PMID'] = abstracts['PMID'].astype(str)



#Load all sentences of all 10 abstracts for each claim into a big list of lists.
claim_sentences = list()
for ids in all_ids:
    all_sentences = list()
    
    for doc_id in ids:
        # Filter the DataFrame for the specific PMID
        abstract_row = abstracts[abstracts['PMID'] == doc_id]
        abstract = abstract_row['AbstractText'].values[0]
        sentences = sent_tokenize(abstract)
        all_sentences.extend(sentences)
        all_sentences = [s.lower() for s in all_sentences]   
    claim_sentences.append(all_sentences)
print("collected all sentences from abstracts!")   

#Load sentence transformer for selecting evidence sentences.
SENTENCE_EMBEDDING_MODEL = 'copenlu/spiced'
model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)

# model = SentenceTransformer("sentence-transformers/msmarco-distilbert-base-tas-b")

#print("loaded sentence model!")


#Find top 10 sentences for each claim (top 10 performed well, but top 5 could maybe bring less noise)
top_sentences = list()

print(len(all_claims))

for idx in range(len(all_claims)):
  
    claim = all_claims[idx]
    sents = claim_sentences[idx]
    
    sents_embeddings = model.encode(sents, convert_to_tensor=True)
    claim_embedding = model.encode(claim, convert_to_tensor=True)
    cos_scores = util.cos_sim(claim_embedding, sents_embeddings)[0]
    top_results = torch.topk(cos_scores, k=10)
    
    np_results = top_results[1].detach().cpu().numpy()
    top_sentences.append(np_results)



selected_sentences = list()
for idx in range(len(all_claims)):
    top = top_sentences[idx]
    # top = top_sentences[idx]
    top = np.sort(top)
    sents = np.array(claim_sentences[idx])[top]    

    selected_sentences.append(sents)
    
print("Printing selected_sentences length: ")
print(len(selected_sentences))    

 # Create a joint list of concatenated claims and evidence, in form of "claim [SEP] evidence1 evidence2 ... evidenceN"
joint_list = list()
for idx in range(len(all_claims)):
    joint = all_claims[idx] + " [SEP] "
    for s in selected_sentences[idx]:
        joint += s
        joint += " "
    joint_list.append(joint)
print("Printing joint list length: ")
print(len(joint_list))    
# Save this in a file before the final step.
output_file_path = config.get_dataset_path('scifact', 'Dev/Evidence Sentences/Pubmed', 'SciFact_joint_lines_negative_pubmed.txt')
with open(output_file_path, "w") as f:
	for example in joint_list:
		f.write(example)
		f.write("\n")
        

print("I am done, bro")
      
