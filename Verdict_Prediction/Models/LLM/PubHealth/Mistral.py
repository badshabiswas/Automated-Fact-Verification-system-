import os
import time
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import sys
sys.path.append(\'..\')
from config import Config

# Validate configuration and initialize config
Config.validate_config()
config = Config()
import re

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------

# # List of input files
# input_files = [
#     config.get_dataset_path('PubHealth\'.lower(), \'Evidence Sentences/Wikipedia/Final Result/PubHealth_joint_lines_wiki_original.txt\'),
#     config.get_dataset_path('PubHealth\'.lower(), \'Evidence Sentences/Wikipedia/Final Result/Original_Google_search_results_merged.txt\'),
#      config.get_dataset_path('PubHealth\'.lower(), \'Evidence Sentences/Pubmed/Final Result/PubHealth_joint_lines_Original_pubmed.txt\'),
#     config.get_dataset_path('PubHealth\'.lower(), \'Evidence Sentences/Pubmed/Final Result/Original_Google_search_results_merged.txt\'),
#       config.get_dataset_path('PubHealth\'.lower(), \'Evidence Sentences/Google/Final Doc/Original_search_results.txt\'),
#     config.get_dataset_path('PubHealth\'.lower(), \'Evidence Sentences/Google/Final Doc/Original_Google_search_results_merged.txt\')
# ]

# # Corresponding output and error log files (one for each input file)
# output_files = [
#     config.get_dataset_path('PubHealth\'.lower(), \'Result/Wikipedia/Mistral/SciFact_Wikipedia_classified_claims_original.txt\'),
#     config.get_dataset_path('PubHealth\'.lower(), \'Result/Wikipedia/Mistral/SciFact_Wikipedia_classified_claims__merged.txt\'),
#         config.get_dataset_path('PubHealth\'.lower(), \'Result/Pubmed/Mistral/SciFact_Pubmed_classified_claims_original.txt\'),
#     config.get_dataset_path('PubHealth\'.lower(), \'Result/Pubmed/Mistral/SciFact_Pubmed_classified_claims__merged.txt\'),
#           config.get_dataset_path('PubHealth\'.lower(), \'Result/Google/Mistral/SciFact_Google_classified_claims_original.txt\'),
#     config.get_dataset_path('PubHealth\'.lower(), \'Result/Google/Mistral/SciFact_Google_classified_claims__merged.txt\')

    
# ]

# error_log_files = [
#     config.get_dataset_path('PubHealth\'.lower(), \'Result/Wikipedia/Mistral/SciFact_Wikipedia_error_log_original.txt\'),
#     config.get_dataset_path('PubHealth\'.lower(), \'Result/Wikipedia/Mistral/SciFact_Wikipedia_error_log__merged.txt\'),
#     config.get_dataset_path('PubHealth\'.lower(), \'Result/Pubmed/Mistral/SciFact_Pubmed_error_log_original.txt\'),
#     config.get_dataset_path('PubHealth\'.lower(), \'Result/Pubmed/Mistral/SciFact_Pubmed_error_log__merged.txt\'),
#         config.get_dataset_path('PubHealth\'.lower(), \'Result/Google/Mistral/SciFact_Google_error_log_original.txt\'),
#     config.get_dataset_path('PubHealth\'.lower(), \'Result/Google/Mistral/SciFact_Google_error_log__merged.txt\')
# ]



# List of input files
input_files = [
    config.get_dataset_path('PubHealth\'.lower(), \'Evidence Sentences/Merged/merged_evidence.txt\')
]

# Corresponding output and error log files (one for each input file)
output_files = [
    config.get_dataset_path('PubHealth\'.lower(), \'Result/Merged/Mistral/PubHealth_merged_classified_claims.txt\')
]

error_log_files = [
    config.get_dataset_path('PubHealth\'.lower(), \'Result/Merged/Mistral/PubHealth_merged_error_log_original.txt\')
]







# Model and Hugging Face cache directory
model_path = "mistralai/Mistral-Large-Instruct-2411"
cache_dir = config.CACHE_DIR"

# Maximum retries for errors
max_retries = config.MAX_RETRIES
throttle_delay = config.THROTTLE_DELAY  # Delay between retry attempts

# --------------------------------------------------------------------------------
# Load Model and Tokenizer Once
# --------------------------------------------------------------------------------

accelerator = Accelerator()
device = accelerator.device

print(f"Loading tokenizer from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, cache_dir=cache_dir, token=config.HUGGINGFACE_TOKEN
)

if tokenizer.pad_token is None:
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
model = accelerator.prepare(model)  # Wrap model for multi-GPU/CPU fallback

# --------------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------------

def extract_classification_with_regex(response_content):

    match = re.search(r"(?:Answer:\s*)?([ABCD])", response_content)
    if match:
        return match.group(1)
    raise ValueError(f"Unexpected response format: {response_content}")

def classify_claim_evidence(claim, evidence):
    """Use the model to classify claim-evidence pairs."""
    
    prompt = (
        "<s>[SYSTEM_PROMPT] You are an AI model trained to verify claims based on provided evidence.[/SYSTEM_PROMPT]"
        "[INST] (Task) Answer the following question:\n"
        f"(Input) Facts: {evidence_text}\n"
        f"Statement: {claim}\n"
        "Is the statement entailed by the given facts?\n"
        "(A) false (B) true (C) unproven (D) mixture \n\n"
        "Please answer A or B or C or D: [/INST]"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        gen_output = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = gen_output.sequences[0, prompt_length:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    classification_char = extract_classification_with_regex(output_text)
    prob = 0.0

    for i, score_distribution in enumerate(gen_output.scores):
        log_probs = torch.log_softmax(score_distribution[0], dim=-1)
        token_id = new_tokens[i].item()
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)

        if token_str.strip() == classification_char:
            classification_logprob = log_probs[token_id].item()
            prob = math.exp(classification_logprob)
            break

    return classification_char, prob

classification_map = {"A": "false", "B": "true", "C": "unproven", "D": "mixture"}

# --------------------------------------------------------------------------------
# Process Each Input File
# --------------------------------------------------------------------------------

for input_file, output_file, error_log_file in zip(input_files, output_files, error_log_files):
    print(f"\nProcessing file: {input_file}")

    claims, evidences = [], []

    # Load claim-evidence pairs
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("[SEP]")
                if len(parts) == 2:
                    claims.append(parts[0].strip())
                    evidences.append(parts[1].strip())

    total_claims = len(claims)
    print(f"Total claims to process: {total_claims}")

    processed_claims = set()
    
    # Load previously processed claims
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    processed_claims.add(parts[0])

    with open(output_file, "a", encoding="utf-8") as output_f, \
         open(error_log_file, "a", encoding="utf-8") as error_log:

        for idx in range(total_claims):
            claim = claims[idx]
            evidence_text = evidences[idx]

            if claim in processed_claims:
                print(f"Skipping processed claim {idx + 1}/{total_claims}: {claim}")
                continue

            attempts = 0
            while attempts < max_retries:
                try:
                    print(f"Processing claim {idx + 1}/{total_claims} (Attempt {attempts + 1}): {claim}")
                    classification_char, probability = classify_claim_evidence(claim, evidence_text)

                    if classification_char not in classification_map:
                        raise ValueError(f"Unexpected classification: {classification_char}")

                    classification_label = classification_map[classification_char]
                    output_f.write(f"{claim}\t{evidence_text}\t{classification_label}\t{probability:.4f}\n")
                    output_f.flush()

                    processed_claims.add(claim)
                    break

                except Exception as e:
                    attempts += 1
                    error_message = f"Failed claim {idx + 1}: {claim}\n{e}\n"
                    error_log.write(error_message)
                    error_log.flush()

    print(f"Finished processing file: {input_file}")

print("\nAll files processed successfully!")