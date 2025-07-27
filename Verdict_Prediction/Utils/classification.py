import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import classification_report
import sys
sys.path.append('..')
from config import Config

# Initialize configuration
config = Config()

# Load SciFact dataset and map labels to integers
train_data_path = config.get_dataset_path('scifact', 'data', 'train.csv')
scifact_df = pd.read_csv(train_data_path, index_col=[0])

scifact_labels = scifact_df.label.tolist()  # Extract true labels

# Load joint lines (Claim-Evidence pairs from text file)
joint_lines_file = config.get_dataset_path('scifact', 'Evidence Sentences/Pubmed/Final top sentences', 'scifact_joint_lines_pubmed_negative_spiced_merged.txt')
with open(joint_lines_file, "r", encoding="utf-8") as f:
    joint_list = [line.strip() for line in f if line.strip()]

# Verify loaded data
print(f"Loaded {len(joint_list)} joint lines from the text file.")
print(f"Loaded {len(scifact_labels)} labels from the CSV.")

# Prepare Dataset for Hugging Face
data_dict = {"text": joint_list, "label": scifact_labels}
dataset = Dataset.from_dict(data_dict)

# Split dataset into train and test sets (80% train, 20% test)
split_dataset = dataset.train_test_split(test_size=0.2)
train_dataset = split_dataset["train"]
test_dataset = split_dataset["test"]

# # Step 2: Load Model and Tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)




# Tokenization Function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Tokenize Train and Test Data
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns and format for PyTorch
tokenized_train = tokenized_train.remove_columns(["text"])
tokenized_test = tokenized_test.remove_columns(["text"])
tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

# Step 3: Define Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Ensure saving is done at the end of each epoch
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer
)

# Step 4: Train the Model
trainer.train()

# Step 5: Evaluate the Model
predictions = trainer.predict(tokenized_test)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = np.array(test_dataset["label"])

# Generate Classification Report
report = classification_report(y_true, y_pred, target_names=["Refuted", "Not Enough Info", "Supported"])
print("Classification Report:\n", report)
