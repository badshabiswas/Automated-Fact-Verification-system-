import os
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from accelerate import Accelerator
import sys
sys.path.append('..')
from config import Config

# Validate configuration and initialize config
Config.validate_config()
config = Config()

# Model configuration
model_path = config.DEFAULT_LLM_MODEL
cache_dir = config.CACHE_DIR

accelerator = Accelerator()
device = accelerator.device

# Load tokenizer and model
print(f"Loading tokenizer from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    token=config.HUGGINGFACE_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    device_map="auto",
    offload_folder="offload",
    torch_dtype=torch.bfloat16,
    token=config.HUGGINGFACE_TOKEN
)

model.eval()

# File paths (configurable)
input_file = config.get_dataset_path('pubhealth', 'Evidence Sentences/Google/Final Doc', 'Original_Google_search_results_merged.txt')
output_file = config.get_dataset_path('pubhealth', 'Result/Local', 'PubHealth_classified_claims.txt')
error_log_file = config.get_dataset_path('pubhealth', 'Result/Local', 'PubHealth_error_log.txt')

# Load data
claims, evidences = [], []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if "[SEP]" in line:
            claim, evidence = line.strip().split("[SEP]")
            claims.append(claim.strip())
            evidences.append(evidence.strip())

# Check progress
processed_claims = set()
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        processed_claims = set(line.split("\t")[0].strip() for line in f if line.strip())

# Classification function
def classify_claim_evidence(claim, evidence):
    prompt = (
        f"Facts: {evidence}\n\n"
        f"Statement: {claim}\n\n"
        f"Is the statement entailed by the given facts?\n"
        "(A) false (B) true (C) unproven (D) mixture\n"
        "Answer:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=5,
        temperature=0.0,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    classification = output_text.split("Answer:")[-1].strip().split()[0]

    return classification

# Processing loop
with open(output_file, "a", encoding="utf-8") as out_f, open(error_log_file, "a", encoding="utf-8") as err_f:
    for idx, (claim, evidence) in enumerate(zip(claims, evidences)):
        if claim in processed_claims:
            print(f"Skipping already processed claim {idx + 1}/{len(claims)}")
            continue

        retries, success = 3, False
        while retries > 0 and not success:
            try:
                classification = classify_claim_evidence(claim, evidence)
                if classification not in ["A", "B", "C", "D"]:
                    raise ValueError(f"Unexpected classification: {classification}")

                label_map = {"A": "false", "B": "true", "C": "unproven", "D": "mixture"}
                label = label_map[classification]

                out_f.write(f"{claim}\t{evidence}\t{label}\n")
                out_f.flush()

                processed_claims.add(claim)
                print(f"Claim {idx + 1}/{len(claims)} classified: {label}")
                success = True
            except Exception as e:
                retries -= 1
                print(f"Error classifying claim {idx + 1}: {e}")
                if retries == 0:
                    err_f.write(f"Failed claim {idx + 1}: {claim}\nError: {e}\n")
                    err_f.flush()
                    print(f"Logged error for claim {idx + 1}")
            finally:
                time.sleep(2)

print("Classification completed.")
