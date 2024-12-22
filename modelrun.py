from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


import ast
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import csv 

# Load the NLI model and tokenizer


# Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("1231czx/llama3_it_ultra_list_and_bold500")
model = AutoModelForSequenceClassification.from_pretrained("1231czx/llama3_it_ultra_list_and_bold500")



# NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, model_max_length=1024)
# model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)

# Define the dataset class
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

# Function to get predictions from the model
def get_result(joint_list, model, tokenizer):
    nli_encoded = tokenizer(joint_list, return_tensors='pt',
                             truncation_strategy='only_first', add_special_tokens=True, padding=True)
    nli_dataset = CtDataset(nli_encoded, np.zeros(len(joint_list)))

    test_loader = DataLoader(nli_dataset, batch_size=8, drop_last=False, shuffle=False, num_workers=4)


    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    result = np.zeros(len(test_loader.dataset))    
    index = 0

    with torch.no_grad():
        for batch_num, instances in enumerate(test_loader):
            input_ids = instances["input_ids"].to("cuda")
            attention_mask = instances["attention_mask"].to("cuda")
            logits = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            probs = logits.softmax(dim=1)

            pred = probs[:,0] > probs[:,1]  # Adjust for binary classification
  # Entailment vs. Contradiction
            pred = np.array(pred.cpu()).astype(int)

            result[index : index + pred.shape[0]] = pred.flatten()
            index += pred.shape[0]

    return result

# Function to read joint claims
def read_joint_claims(input_file):
    joint_claims_list = []
    with open(input_file, 'r') as f_in:
        for line in f_in:
            line = line.strip()
            if line:
                joint_claims_list.append(line)
    return joint_claims_list

# Read joint claims from file
input_file_path = '/home/mbiswas2/Fact-Checking/Google/parsed_claim_evidence.txt'
joint_claims = read_joint_claims(input_file_path)

# Load the SciFact dataset
scifact_df = pd.read_csv("/scratch/mbiswas2/Fact Checking/Datasets/scifact_no-nei_dataset.csv", index_col=[0])
scifact_claims = scifact_df.claim.tolist()
scifact_labels = scifact_df.label.tolist()

# Get predictions from the NLI model
prediction_result = get_result(joint_claims, model, tokenizer)

# Generate classification report
target_names = ['refuted', 'supported']
evaluation_report = classification_report(scifact_labels, prediction_result, target_names=target_names, output_dict=True)

# Save the results and evaluation report to Excel
df_joint_claims = pd.DataFrame({'Claim-Evidence': joint_claims, 'Actual_Label': scifact_labels, 'Predicted_Label': prediction_result})
df_evaluation_report = pd.DataFrame(evaluation_report).transpose()

# Save the DataFrames to Excel
output_file = '/home/mbiswas2/Fact-Checking/Results/bart-large-mnli_NLI_Prediction_Report.xlsx'
with pd.ExcelWriter(output_file) as writer:
    df_joint_claims.to_excel(writer, sheet_name='Joint_Claims', index=False)
    df_evaluation_report.to_excel(writer, sheet_name='Evaluation_Report', index=True)

# Calculate confusion matrix
conf_matrix = confusion_matrix(scifact_labels, prediction_result)

# Print the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Save the confusion matrix to Excel
df_conf_matrix = pd.DataFrame(conf_matrix, index=target_names, columns=target_names)
with pd.ExcelWriter(output_file, mode='a') as writer:  # Append to the existing Excel file
    df_conf_matrix.to_excel(writer, sheet_name='Confusion_Matrix')

print(f"Results saved to {output_file}")


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



#Predict scores
print_scores(scifact_labels, prediction_result)

