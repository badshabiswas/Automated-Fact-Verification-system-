'''
Other than semantic search, the process can also be done with the classic BM25 search.

There is a library called 'retriv' that provides a simple function to build an inverted index out of your document corpus.
It took almost two hours to construct the index, but after that it can be pickled and used easily. 
Batch search of 1000 claims at once takes milliseconds over the index.

'''

from retriv import SparseRetriever

#Create the SparseRetriever object that will be used for BM25 search.
sr = SparseRetriever(
  index_name="pubmed-index",
  model="bm25",
  min_df=10,
  tokenizer="whitespace",
  stemmer="english",
  stopwords="english",
  do_lowercasing=True,
  do_ampersand_normalization=True,
  do_special_chars_normalization=True,
  do_acronyms_normalization=True,
  do_punctuation_removal=True,
)

corpus_path = "/projects/ouzuner/mbiswas2/pubmed_landscape_abstracts_2024.csv"


#Construct the inverted index.
import time
start = time.time()

sr = sr.index_file(
  path=corpus_path,  # File kind is automatically inferred
  show_progress=True,         # Default value
  callback=lambda doc: {      # Callback defaults to None.
    "id": doc["PMID"],
    "text": doc["AbstractText"],          
    }
  )

duration = time.time() - start
print(duration)
#Duration: 5772.615013837814


# #Pickle the file.
# import pickle
# file = open('/projects/ouzuner/mbiswas2/pickled_sr', 'wb')
# pickle.dump(sr, file)

# sr.search(
#   query="colorectal cancer high fiber diet",    # What to search for        
#   return_docs=True,          # Default value, return the text of the documents
#   cutoff=10,                # Default value, number of results to return
# )

import pandas as pd
import numpy as np
import csv 

scifact_df = pd.read_csv("/scratch/mbiswas2/Fact Checking/Datasets/scifact_no-nei_dataset.csv", index_col=[0])
#scifact_df = pd.read_csv("/scratch/mbiswas2/Fact Checking/Datasets/scifact_no-nei_dataset_small_50.csv", index_col=[0])
scifact_claims = scifact_df.claim.tolist()
scifact_labels = scifact_df.label.tolist()

# Print the number of claims and labels to verify.
print(f"Number of claims: {len(scifact_claims)}")
print(f"Number of labels: {len(scifact_labels)}")

claims = scifact_claims
print(len(claims))


claims = scifact_claims
print(len(claims))

#Batch search for the whole dataset
query_list = list()

idx = 0
for c in claims:
    c = c.lower()
    d = dict()
    d["id"] = str(idx)
    d["text"] = c
    query_list.append(d)
    idx += 1

results = sr.msearch(
  queries=query_list,
  cutoff=10,
)
print(results)

#Print all the results
with open("/projects/ouzuner/mbiswas2/bm25_scifact_pmids_updated.txt", "w") as f:
    f.write(str(results))