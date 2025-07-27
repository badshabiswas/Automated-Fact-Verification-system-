'''
Now find the top 10 most similar documents for every query from a claim verification / fact-checking dataset.

Let's use SciFact for example (https://aclanthology.org/2020.emnlp-main.609.pdf).

'''



from sentence_transformers import SentenceTransformer, util
import glob
import torch 
import pandas as pd
import numpy as np
import csv
import os
import sys
sys.path.append('..')
from config import Config

# Initialize configuration
config = Config()

# Load embedding models from configuration
DOCUMENT_EMBEDDING_MODEL = config.DEFAULT_EMBEDDING_MODEL
QUERY_EMBEDDING_MODEL = config.DEFAULT_EMBEDDING_MODEL  # needs to be the same model so the results make sense
'''
Now find the top 10 most similar documents for every query from a claim verification / fact-checking dataset.

Let's use SciFact for example (https://aclanthology.org/2020.emnlp-main.609.pdf).

'''


# Path to the claims text file
claims_file_path = config.get_dataset_path('scifact', 'data/Claim Only', 'negated_claims_refined.txt')

# Read the claims from the text file (each line contains a claim)
with open(claims_file_path, "r") as file:
    claims = [line.strip() for line in file.readlines()]

# Load all numpy arrays with embeddings
embeddings_dir = os.path.join(config.CACHE_DIR, "pubmed_embeddings")
npfiles = glob.glob(os.path.join(embeddings_dir, "*.npy"))
npfiles.sort()
all_np_arrays = list()
for npfile in npfiles:
    arr = np.load(npfile)
    if arr.shape == (100000, 768):
        all_np_arrays.append(arr)
    else:
        print(f"Handling inconsistent file manually: {npfile}, shape: {arr.shape}")
        # Manually add the smaller array without reshaping
        all_np_arrays.append(arr)

#This will created an array with shape (20M, 768), with all document embeddings. 
document_embeddings = np.concatenate(all_np_arrays, axis=0).reshape(-1, 768)


# Load query embedding model.
sentence_model = SentenceTransformer(QUERY_EMBEDDING_MODEL, device="cpu")
print("I'm writing on file now...It's working")
output_file_path = config.get_dataset_path('scifact', 'Relevant Doc/Pubmed', 'pubmed_scifact_pmids_New_Slumrm_Negative_kamalkraj.txt')
with open(output_file_path, "w") as f:
    for query in claims:
        query_embedding = sentence_model.encode(query, convert_to_tensor=True)
        sims = util.cos_sim(query_embedding, document_embeddings)
      
        f.write("\n")
        f.write(str(torch.topk(sims, 10).values[0].tolist()))
        f.write("\n")
        f.write(str(torch.topk(sims, 10).indices[0].tolist()))
        f.write("\n")
        f.write("\n")
print("Done, man")