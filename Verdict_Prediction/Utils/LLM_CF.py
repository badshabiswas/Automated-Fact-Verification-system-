# Import necessary modules
import pandas as pd
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import Dataset
from sklearn.metrics import classification_report
import sys
sys.path.append('..')
from config import Config

# Initialize configuration
config = Config()

# Load SciFact dataset and map labels to strings for text generation
train_data_path = config.get_dataset_path('scifact', 'data', 'train.csv')
scifact_df = pd.read_csv(train_data_path, index_col=[0])
label_map = {0: "Refuted", 1: "Not Enough Info", 2: "Supported"}
scifact_labels = [label_map[label] for label in scifact_df.label.tolist()]

# Load joint lines (Claim-Evidence pairs)
joint_lines_file = config.get_dataset_path('scifact', 'Evidence Sentences/Pubmed/Final top sentences', 'scifact_joint_lines_pubmed_negative_spiced_merged.txt')
with open(joint_lines_file, "r", encoding="utf-8") as f:
    joint_list = [line.strip() for line in f if line.strip()]

# Verify data loading
print(f"Loaded {len(joint_list)} joint lines.")
print(f"Loaded {len(scifact_labels)} labels.")

# Prepare dataset for Hugging Face
data_dict = {"text": joint_list, "label": scifact_labels}
dataset = Dataset.from_dict(data_dict)

# Split dataset into train and test (80/20 split)
split_dataset = dataset.train_test_split(test_size=0.2)
train_dataset, test_dataset = split_dataset["train"], split_dataset["test"]

# Load flan-t5-base model and tokenizer
model_name = "google/flan-t5-xxl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Tokenization function
def preprocess_function(examples):
    inputs = [f"classify: {text}" for text in examples["text"]]
    targets = examples["label"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=10, truncation=True, padding="max_length")
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Tokenize datasets
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True)

# Prepare datasets for PyTorch
tokenized_train = tokenized_train.remove_columns(["text", "label"])
tokenized_test = tokenized_test.remove_columns(["text", "label"])
tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    load_best_model_at_end=True,
    logging_dir="./logs"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Move model to device

# Generate predictions and evaluate
print("Generating predictions...")
predictions = []
for example in test_dataset["text"]:
    input_text = f"classify: {example}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}  # Move inputs to the same device as the model
    
    outputs = model.generate(**inputs)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    predictions.append(decoded_output)

# Generate classification report
decoded_labels = test_dataset["label"]
print("Classification Report:\n", classification_report(decoded_labels, predictions, target_names=["Refuted", "Not Enough Info", "Supported"]))
