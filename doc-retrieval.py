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

# Load ClinicalBERT model and tokenizer
DOCUMENT_EMBEDDING_MODEL = "kamalkraj/BioSimCSE-BioLinkBERT-BASE"
QUERY_EMBEDDING_MODEL = "kamalkraj/BioSimCSE-BioLinkBERT-BASE" #needs to be the same model so the results make sense
'''
Now find the top 10 most similar documents for every query from a claim verification / fact-checking dataset.

Let's use SciFact for example (https://aclanthology.org/2020.emnlp-main.609.pdf).

'''

#SciFact
scifact_df = pd.read_csv("/scratch/mbiswas2/Fact Checking/Datasets/scifact_no-nei_dataset.csv", index_col=[0])
scifact_claims = scifact_df.claim.tolist()
scifact_labels = scifact_df.label.tolist()

claims = scifact_claims

#Load all numpy arrays with embeddings
npfiles = glob.glob("/projects/ouzuner/mbiswas2/pubmed/Embedding_old_Pubmed/*.npy")
npfiles.sort()
all_np_arrays = list()
for npfile in npfiles:
    all_np_arrays.append(np.load(npfile))    
all_np_arrays = np.array(all_np_arrays)

#This will created an array with shape (20M, 768), with all document embeddings. 
document_embeddings = all_np_arrays.reshape(-1, 768)

#Load query embedding model.
sentence_model = SentenceTransformer(QUERY_EMBEDDING_MODEL, device="cpu")
print("I'm writing on file now...It's working")
with open("/home/mbiswas2/Fact Checking/Pubmed_Retrieve_PMID_Scifact/pubmed_scifact_pmids_full_old.txt", "w") as f:
    for query in claims:
        query_embedding = sentence_model.encode(query, convert_to_tensor=True)
        
        sims = util.cos_sim(query_embedding, document_embeddings)
        f.write(query)
        f.write("\n")
        f.write(str(torch.topk(sims, 10).values[0].tolist()))
        f.write("\n")
        f.write(str(torch.topk(sims, 10).indices[0].tolist()))
        f.write("\n")
        f.write("\n")
print("Done, man")