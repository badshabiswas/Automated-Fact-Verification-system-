'''
This is the final step, where the claim-evidence pairs get a verdict prediction from an NLI model.
These predictions are compared to the gold labels from manual annotators and we get some F1 metrics.

The NLI model used is DeBERTa-v3, fine-tuned on some popular NLI datasets:
https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
This is probably the best encoder-only model for NLI tasks (see the GLUE leaderboard for some others).

'''

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


import ast
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
from sentence_transformers import SentenceTransformer, util


# Load all claims and document PMIDs from the file.
import ast
import pandas as pd

# Load all claims and document PMIDs from the updated file format
all_ids = list()
with open("/projects/ouzuner/mbiswas2/bm25_scifact_pmids.txt", "r") as f:
    data = ast.literal_eval(f.read())  # Read the entire file and parse as a dictionary
    
    for key, value in data.items():
        # Extract only the document IDs from the value (which is a dictionary)
        ids = list(value.keys())  
        all_ids.append(ids)

# Load claims from the scifact dataset and use them as all_claims
scifact_df = pd.read_csv("/scratch/mbiswas2/Fact Checking/Datasets/scifact_no-nei_dataset.csv", index_col=[0])
scifact_claims = scifact_df.claim.tolist()
scifact_labels = scifact_df.label.tolist()

# Use scifact claims as the only claims
all_claims = scifact_claims

# print("All claims (from SciFact claims):")
# print(all_claims)
# print("All document IDs:")
# print(all_ids)

#Load all PubMed abstracts.
abstracts = pd.read_csv("/projects/ouzuner/mbiswas2/pubmed_landscape_abstracts.csv")
abstracts_text = abstracts.AbstractText.tolist()
print("loaded abstracts!")


#Load all sentences of all 10 abstracts for each claim into a big list of lists.
claim_sentences = list()
for ids in all_ids:
    all_sentences = list()
    
    for doc_id in ids:
        abstract = abstracts_text[doc_id]
        sentences = sent_tokenize(abstract)
        all_sentences.extend(sentences)
        all_sentences = [s.lower() for s in all_sentences]   
    claim_sentences.append(all_sentences)
print("collected all sentences from abstracts!")   

#Load sentence transformer for selecting evidence sentences.
SENTENCE_EMBEDDING_MODEL = 'copenlu/spiced'
model = SentenceTransformer(SENTENCE_EMBEDDING_MODEL)
#print("loaded sentence model!")


#Find top 10 sentences for each claim (top 10 performed well, but top 5 could maybe bring less noise)
top_sentences = list()

print(len(all_claims))

for idx in range(len(all_claims)):
  
    claim = all_claims[idx]
    sents = claim_sentences[idx]
    
    sents_embeddings = model.encode(sents, convert_to_tensor=True)
    claim_embedding = model.encode(claim, convert_to_tensor=True)
    cos_scores = util.cos_sim(claim_embedding, sents_embeddings)[0]
    top_results = torch.topk(cos_scores, k=10)
    
    np_results = top_results[1].detach().cpu().numpy()
    top_sentences.append(np_results)



selected_sentences = list()
for idx in range(len(all_claims)):
    top = top_sentences[idx]
    # top = top_sentences[idx]
    top = np.sort(top)
    sents = np.array(claim_sentences[idx])[top]    

    selected_sentences.append(sents)
    
print("Printing selected_sentences length: ")
print(len(selected_sentences))    

 # Create a joint list of concatenated claims and evidence, in form of "claim [SEP] evidence1 evidence2 ... evidenceN"
joint_list = list()
for idx in range(len(all_claims)):
    joint = all_claims[idx] + " [SEP] "
    for s in selected_sentences[idx]:
        joint += s
        joint += " "
    joint_list.append(joint)


file_path = '/scratch/mbiswas2/Fact Checking/Code/healthfc_joint_lines.txt'

output_excel_path = "/home/mbiswas2/ondemand/fact-checking/Implementation comparison/Comparing-Knowledge-Sources-main/Code/Result/predictions_scifact_clinicalBERT_DeBERTa-v3-large.xlsx"  # Change this to your desired path


tokenizer = AutoTokenizer.from_pretrained("KatoHF/deberta-v3-large-classifier")
model = AutoModelForSequenceClassification.from_pretrained("KatoHF/deberta-v3-large-classifier")

#The dataset class, consiting of encoding of the joint line and its gold label.
class CtDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


#Creates a NumPy array with results predictions.
# def get_result(joint_list, indices, model, tokenizer):
def get_result(joint_list, model, tokenizer):
    nli_test = joint_list
    nli_encoded = tokenizer(nli_test, return_tensors='pt',
                             truncation_strategy='only_first', add_special_tokens=True, padding=True)
    nli_dataset = CtDataset(nli_encoded, np.zeros(len(nli_test)))

    test_loader = DataLoader(nli_dataset, batch_size=16,
                             drop_last=False, shuffle=False, num_workers=4)

    model.eval()
    model = model.to("cuda")
    
    result = np.zeros(len(test_loader.dataset))    
    index = 0

    with torch.no_grad():
        for batch_num, instances in enumerate(test_loader):
            input_ids = instances["input_ids"].to("cuda")
            attention_mask = instances["attention_mask"].to("cuda")
            logits = model(input_ids=input_ids,
                                          attention_mask=attention_mask)[0]
            probs = logits.softmax(dim=1)

            #If the entailment score was bigger than contradiction score, predict "SUPPORTED" (positive). 
            #Otherwise, predict "REFUTED" (negative).
            pred = probs[:,0] > probs[:,1]
            pred = np.array(pred.cpu()).astype(int)

            result[index : index + pred.shape[0]] = pred.flatten()
            index += pred.shape[0]

    return result


def print_scores(actual_values, predicted_values):
    # Calculate precision
    precision = precision_score(actual_values, predicted_values, average = "binary")

    # Calculate recall
    recall = recall_score(actual_values, predicted_values, average = "binary")

    # Calculate F1 score
    f1 = f1_score(actual_values, predicted_values, average = "binary")

    # Calculate accuracy
    accuracy = accuracy_score(actual_values, predicted_values)

    # Print the results
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)
    
    
def read_claims_from_file(file_path):
    """
    Reads claims from a text file where each line is a separate claim.
    
    Args:
    - file_path (str): Path to the text file containing claims.
    
    Returns:
    - list of str: List containing each claim as an individual string.
    """
    claims_list = []
    with open(file_path, 'r') as file:
        for line in file:
            claim = line.strip()
            if claim:
                claims_list.append(claim)
    return claims_list

healthfc_Claims_joint = read_claims_from_file(file_path)
    
print(f"Printing the helthfc label:{healthfc_yesno_labels}")
#Prediction
prediction_result = get_result(joint_list, model, tokenizer)

#Predict scores
print_scores(scifact_labels, prediction_result)



results_df = pd.DataFrame({
    'Claim': scifact_claims,
    'Actual Label': scifact_labels,
    'Predicted Label': prediction_result
})

# Save results to an Excel file
results_df.to_excel(output_excel_path, index=False)

print(f"Results have been saved to {output_excel_path}")

