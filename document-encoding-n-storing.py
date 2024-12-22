from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import csv

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


abstracts = pd.read_csv("/projects/ouzuner/mbiswas2/pubmed_landscape_abstracts.csv")

DOCUMENT_EMBEDDING_MODEL = "kamalkraj/BioSimCSE-BioLinkBERT-BASE"
QUERY_EMBEDDING_MODEL = "kamalkraj/BioSimCSE-BioLinkBERT-BASE" #needs to be the same model so the results make sense

sentence_model = SentenceTransformer(DOCUMENT_EMBEDDING_MODEL, device="cuda:0")

        
# Generate embeddings iteratively for each 100k documents and save as "npy" (numpy format).
# This took many hours and I ran it overnight.
print("I started encoding...")
for step in range(1,205):
    abstracts_slice = abstracts[step*100000:(step+1)*100000].AbstractText.tolist()
    encoded_slice = sentence_model.encode(abstracts_slice)

    with open("/home/mbiswas2/Fact Checking/PubMed/embeddings/20.6/" + str(step) + ".npy",'wb') as f:
        np.save(f, encoded_slice)


print("Hi boss, I am done")
