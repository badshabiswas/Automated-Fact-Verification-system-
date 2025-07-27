import os
os.environ["TRANSFORMERS_BACKEND"] = "torch"

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import torch
import sys
sys.path.append('..')
from config import Config

# Initialize configuration
config = Config()

'''
While there are many versions of S-BERT to use for sentence embeddings (https://www.sbert.net/docs/pretrained_models.html),
there are not so many available for the biomedical domain.

Some relevant ones I found were:
1) https://huggingface.co/kamalkraj/BioSimCSE-BioLinkBERT-BASE
2) https://huggingface.co/pritamdeka/S-BioBert-snli-multinli-stsb 

I opted for the first one.

I encoded all the abstracts and saved them on my disk. These experiments were started before vector databases became 
a very popular thing, so probably a better idea nowadays would be to store them in a vector DB, instead of a disk.
Still, this also works.
'''

print("Loading data...")

# Load abstracts using chunks to optimize memory usage
chunk_size = 100000  # Process in batches of 100,000 abstracts to manage memory
abstracts_path = os.path.join(config.BASE_DATA_DIR, "pubmed_landscape_abstracts_2024.csv")
abstracts = pd.read_csv(abstracts_path, usecols=["AbstractText"])

# Check if GPU is available and set the device accordingly
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the SentenceTransformer model on GPU
DOCUMENT_EMBEDDING_MODEL = "kamalkraj/BioSimCSE-BioLinkBERT-BASE"
QUERY_EMBEDDING_MODEL = DOCUMENT_EMBEDDING_MODEL  # Must be the same for consistency

sentence_model = SentenceTransformer(DOCUMENT_EMBEDDING_MODEL, device=device)

# Function to encode abstracts and save embeddings
def encode_and_save(step, abstracts_chunk):
    abstracts_slice = abstracts_chunk.AbstractText.tolist()
    print(f"Encoding batch {step}...")

    # Perform encoding on GPU with optimal batch size
    encoded_slice = sentence_model.encode(
        abstracts_slice, 
        batch_size=512,  # Adjust based on available GPU memory
        show_progress_bar=True, 
        convert_to_numpy=True, 
        normalize_embeddings=True  # Useful for cosine similarity retrieval
    )

    # Save embeddings to disk
    embeddings_dir = os.path.join(config.CACHE_DIR, "pubmed_embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    output_file = os.path.join(embeddings_dir, f"step{step}.npy")
    np.save(output_file, encoded_slice)
    print(f"Batch {step} saved to {output_file}")

print("Starting encoding process...")

# Process data in chunks to optimize memory usage and speed up processing on GPU
for step, chunk in enumerate(pd.read_csv(abstracts_path, 
                                         chunksize=chunk_size, usecols=["AbstractText"])):
    encode_and_save(step, chunk)

print("Encoding process completed successfully.")
