from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import numpy as np
import csv

NLI_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

# SciFact dataset
scifact_df = pd.read_csv("/scratch/mbiswas2/Fact Checking/Datasets/scifact_no-nei_dataset.csv", index_col=[0])
scifact_claims = scifact_df.claim.tolist()
scifact_labels = scifact_df.label.tolist()

# Print the number of claims and labels to verify.
print(f"Number of claims: {len(scifact_claims)}")
print(f"Number of labels: {len(scifact_labels)}")

claims = scifact_claims
print(len(claims))

tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL, model_max_length=1024)
model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)

# Dataset class consisting of encoding of the joint line and its gold label.
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

# Function to get prediction results
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
            logits = model(input_ids=input_ids, attention_mask=attention_mask)[0]
            probs = logits.softmax(dim=1)

            # Store the probability of the "entailment" class (index 0)
            result[index : index + probs.shape[0]] = probs[:,0].cpu().numpy()  # Entailment probability

            index += probs.shape[0]

    return result

# Function to read joint claims from a file
def read_joint_claims(input_file):
    joint_claims_list = list()

    # Open and read the input file
    with open(input_file, 'r') as f_in:
        for line in f_in:
            line = line.strip()
            if line:
                joint_claims_list.append(line)
    
    return joint_claims_list

# Specify the input file path
input_file_path = '/home/mbiswas2/Fact-Checking/Google/parsed_claim_evidence.txt'

# Call the function to read joint claims and store in a list
joint_claims = read_joint_claims(input_file_path)

# Print to verify
print("Number of joint claims:", len(joint_claims))

# Prediction
prediction_result = get_result(joint_claims, model, tokenizer)

# Define a threshold for classification (e.g., 0.5 for entailment)
threshold = 0.5
predicted_labels = (prediction_result >= threshold).astype(int)  # 1 for entailment, 0 for non-entailment

# Convert actual labels from scifact dataset (assuming entailment is labeled as 1 and refuted as 0)
# In case your dataset uses other labels, map them accordingly
actual_labels = np.array([1 if label == 'SUPPORT' else 0 for label in scifact_labels])

# Generate classification report
print("Classification Report:")
report = classification_report(actual_labels, predicted_labels, target_names=['Non-Entailment', 'Entailment'])
print(report)

# Optionally, save the classification report to a file
report_output_path = "/home/mbiswas2/Fact-Checking/Results/classification_report_scifact_google.txt"
with open(report_output_path, 'w') as f:
    f.write(report)
    
print(f"Classification report saved to {report_output_path}")

# Save predictions to an Excel file
output_excel_path = "/home/mbiswas2/Fact-Checking/Results/predictions_scifact_google.xlsx"

results_df = pd.DataFrame({
    'Claim': scifact_claims,
    'Actual Label': scifact_labels,
    'Predicted Probability (Entailment)': prediction_result
})

# Save results to an Excel file
results_df.to_excel(output_excel_path, index=False)

print(f"Results have been saved to {output_excel_path}")
