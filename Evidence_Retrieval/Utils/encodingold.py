from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import os
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

abstracts_path = os.path.join(config.BASE_DATA_DIR, "pubmed_landscape_abstracts.csv")
abstracts = pd.read_csv(abstracts_path)
abstracts


DOCUMENT_EMBEDDING_MODEL = config.DEFAULT_EMBEDDING_MODEL
QUERY_EMBEDDING_MODEL = config.DEFAULT_EMBEDDING_MODEL  # needs to be the same model so the results make sense

sentence_model = SentenceTransformer(DOCUMENT_EMBEDDING_MODEL, device="cuda:0")

        
# Generate embeddings iteratively for each 100k documents and save as "npy" (numpy format).
# This took many hours and I ran it overnight.
embeddings_dir = os.path.join(config.CACHE_DIR, "pubmed_embeddings_old")
os.makedirs(embeddings_dir, exist_ok=True)

for step in range(1,205):
    abstracts_slice = abstracts[step*100000:(step+1)*100000].AbstractText.tolist()
    encoded_slice = sentence_model.encode(abstracts_slice)

    output_file = os.path.join(embeddings_dir, f"step{step}.npy")
    with open(output_file,'wb') as f:
        np.save(f, encoded_slice)
        

# Load all the documents and sort. This potentially requires a lot of RAM.
# If using a vector DB, then they can be persistend on a disk.
import glob
npfiles = glob.glob(os.path.join(embeddings_dir, "*.npy"))
npfiles.sort()
npfiles          