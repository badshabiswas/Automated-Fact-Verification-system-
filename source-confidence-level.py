import pandas as pd
import numpy as np

print(f"Hey I have started the script..")

abstracts = pd.read_csv("/projects/ouzuner/mbiswas2/pubmed_landscape_abstracts.csv")

print(f"The length of abstract is: {len(abstracts)}")