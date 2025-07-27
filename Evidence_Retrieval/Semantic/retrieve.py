import json
from sentence_transformers import SentenceTransformer, util
import glob
import torch 
import pandas as pd
import numpy as np
import sys
sys.path.append('..')
from config import Config

# Initialize configuration
config = Config()

# Load models (same for documents and queries to ensure proper alignment)
DOCUMENT_EMBEDDING_MODEL = config.DEFAULT_EMBEDDING_MODEL
QUERY_EMBEDDING_MODEL = config.DEFAULT_EMBEDDING_MODEL

# ------------------------------------------------------------------
# 1. Read Averitec claims
# ------------------------------------------------------------------
file_path = config.get_dataset_path('averitec', '', 'claims_labels_unique.csv')
df = pd.read_csv(file_path)
claims = df['claim'].tolist()

# ------------------------------------------------------------------
# 2. Load all document embeddings
# ------------------------------------------------------------------
embedding_dir = os.path.join(config.CACHE_DIR, "pubmed_embeddings")
npfiles = glob.glob(os.path.join(embedding_dir, "*.npy"))
npfiles.sort()

all_np_arrays = []
for npfile in npfiles:
    arr = np.load(npfile)
    if arr.shape == (100000, 768):
        all_np_arrays.append(arr)
    else:
        print(f"Handling inconsistent file manually: {npfile}, shape: {arr.shape}")
        # Manually add smaller array (or handle accordingly)
        all_np_arrays.append(arr)

document_embeddings = np.concatenate(all_np_arrays, axis=0).reshape(-1, 768)

# ------------------------------------------------------------------
# 3. Load the query embedding model
# ------------------------------------------------------------------
sentence_model = SentenceTransformer(QUERY_EMBEDDING_MODEL, device="cpu")

# ------------------------------------------------------------------
# 4. For each claim, find top 10 most similar document embeddings
# ------------------------------------------------------------------
output_data = []  # will store results for each claim

for query in claims:
    # Encode the query
    query_embedding = sentence_model.encode(query, convert_to_tensor=True)
    
    # Calculate cosine similarities
    sims = util.cos_sim(query_embedding, document_embeddings)  # shape: (1, #docs)
    
    # Retrieve top-10 similarities
    top_k_values, top_k_indices = torch.topk(sims, 10)
    # Since top_k_values, top_k_indices will have shape (1, 10), extract the first row
    top_k_values = top_k_values[0].tolist()
    top_k_indices = top_k_indices[0].tolist()
    
    # Build the result dictionary for this claim
    result = {
        "claim": query,
        "top_10_indices": top_k_indices,
        "top_10_scores": top_k_values
    }
    output_data.append(result)

# ------------------------------------------------------------------
# 5. Save the results as JSON
# ------------------------------------------------------------------
output_json_file = config.get_dataset_path('averitec', 'Relevant Doc/Pubmed', 'Original_pubmed_averitec.json')
with open(output_json_file, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print("Done! Results saved to", output_json_file)
