'''
Other than semantic search, the process can also be done with the classic BM25 search.

There is a library called 'retriv' that provides a simple function to build an inverted index out of your document corpus.
It took almost two hours to construct the index, but after that it can be pickled and used easily. 
Batch search of 1000 claims at once takes milliseconds over the index.

'''

import pandas as pd
import os
from retriv import SparseRetriever
import sys
sys.path.append('..')
from config import Config

# Initialize configuration
config = Config()

# Specify a directory with more space for retriv index
custom_save_dir = os.path.join(config.CACHE_DIR, "retriv_index")

# Ensure the directory exists
os.makedirs(custom_save_dir, exist_ok=True)

# Set environment variable to change retriv's default save path
os.environ["RETRIV_COLLECTIONS_DIR"] = custom_save_dir



#Create the SparseRetriever object that will be used for BM25 search.
sr = SparseRetriever(
  index_name="pubmed-index_2023",
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

corpus_path = os.path.join(config.BASE_DATA_DIR, "pubmed_landscape_abstracts_2024.csv")


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


#Pickle the file.
import pickle
sr.save()




#Search for a single query
sr.search(
  query="colorectal cancer high fiber diet",    # What to search for        
  return_docs=True,          # Default value, return the text of the documents
  cutoff=10,                # Default value, number of results to return
)

# Path to the claims text file
claims_file_path = config.get_dataset_path('averitec', '', 'negated_claims_API.txt')

# Read the claims from the text file (each line contains a claim)
with open(claims_file_path, "r") as file:
    claims = [line.strip() for line in file.readlines()]

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

# Print all the results
output_file = config.get_dataset_path('averitec', 'Relevant Doc/Pubmed', 'bm25_averitec_pmids_2024_Negative.txt')
with open(output_file, "w") as f:
    f.write(str(results))