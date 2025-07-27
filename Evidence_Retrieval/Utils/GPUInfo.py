# from sentence_transformers import SentenceTransformer, util
# import torch
# import glob
# import pandas as pd
# import numpy as np

# # Load ClinicalBERT model and tokenizer
# DOCUMENT_EMBEDDING_MODEL = "pritamdeka/S-BioBert-snli-multinli-stsb"
# QUERY_EMBEDDING_MODEL = "pritamdeka/S-BioBert-snli-multinli-stsb"

# # Check if GPU is available and set device accordingly
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # Load query embedding model with GPU acceleration if available
# sentence_model = SentenceTransformer(QUERY_EMBEDDING_MODEL, device=device)

# # SciFact dataset path
# file_path = config.get_dataset_path('scifact', 'data', 'claims_train.jsonl')
# df = pd.read_json(file_path, lines=True)
# claims = df['claim'].tolist()

# # Load all numpy arrays with embeddings
# npfiles = glob.glob(os.path.join(config.CACHE_DIR, "pubmed_embeddings", "*.npy"))
# npfiles.sort()
# all_np_arrays = [np.load(npfile) for npfile in npfiles]

# # Concatenate and reshape document embeddings
# document_embeddings = np.concatenate(all_np_arrays, axis=0).reshape(-1, 768)
# document_embeddings = torch.tensor(document_embeddings, device=device)

# # Write results to output file
# output_file = config.get_dataset_path('scifact', 'Relevant Doc/Pubmed', 'pubmed_scifact_pmids_New_GPU_Check.txt')
# print("Processing queries and writing to file...")

# with open(output_file, "w") as f:
#     for query in claims:
#         query_embedding = sentence_model.encode(query, convert_to_tensor=True, device=device)

#         # Compute cosine similarity on GPU
#         sims = util.cos_sim(query_embedding, document_embeddings)

#         # Get top-10 most similar documents
#         top_values, top_indices = torch.topk(sims, 10)

#         # Write to file
#         f.write("\n")
#         f.write(str(top_values[0].cpu().tolist()))  # Move to CPU for writing
#         f.write("\n")
#         f.write(str(top_indices[0].cpu().tolist()))  # Move to CPU for writing
#         f.write("\n\n")

# print("Done, processed all queries!")

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

# Load ClinicalBERT model and tokenizer
DOCUMENT_EMBEDDING_MODEL = config.DEFAULT_EMBEDDING_MODEL
QUERY_EMBEDDING_MODEL = config.DEFAULT_EMBEDDING_MODEL

# Load all the PubMed abstracts into one variable.
embeddings_dir = os.path.join(config.CACHE_DIR, "pubmed_embeddings")
npfiles = glob.glob(os.path.join(embeddings_dir, "*.npy"))
npfiles.sort()

all_arrays = list()
for npfile in npfiles:
    print(npfile)
    all_arrays.append(np.load(npfile))
    
stacked = np.vstack(all_arrays)
stacked.shape


# Use the same embedding model to embed the query.
sentence_model = SentenceTransformer(QUERY_EMBEDDING_MODEL, device="cpu")

#The query you wish to embed...
query = "0-dimensional biomaterials lack inductive properties."
query_embedding = sentence_model.encode(query, convert_to_tensor=True)

# Load all numpy arrays with embeddings
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

#Calculate all the cosine similarities 
similarity_values = util.cos_sim(query_embedding, document_embeddings)

#PubMed IDs (PMIDs) of the most similar documents.
most_similar_ids = torch.topk(similarity_values, 10).indices[0].tolist()

#Cosine similarity values between the query and most similar documents.
most_similar_values = torch.topk(similarity_values, 10).values[0].tolist()