import os
import time
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import sys
sys.path.append('.')
from config import Config

# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------

# Validate configuration and initialize config
Config.validate_config()
config = Config()

# Paths for input and output files (configurable)
joint_lines_file = config.get_dataset_path('pubhealth', 'Evidence Sentences/Google/Final Doc', 'Original_Google_search_results_merged.txt')
output_file = config.get_dataset_path('pubhealth', 'Result/Local/Google/Mistral', 'PubHealth_classified_claims_test.txt')
error_log_file = config.get_dataset_path('pubhealth', 'Result/Local/Google/Mistral', 'PubHealth_error_log_test.txt')

# Local Hugging Face model directory or Hub ID
cache_dir = config.CACHE_DIR
model_path = "unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"

# Maximum number of retry attempts
max_retries = 3

# Delay between attempts to avoid spamming the model
throttle_delay = 0  # seconds

# --------------------------------------------------------------------------------
# Setup Accelerate + Loading Model
# --------------------------------------------------------------------------------

accelerator = Accelerator()
device = accelerator.device

print(f"Loading tokenizer from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    token=config.HUGGINGFACE_TOKEN
)

# Ensure we have a valid pad token (some causal LMs may not define one)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model from: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    device_map="auto",         # Let Accelerate manage device placement
    offload_folder="offload",  # Where to offload large layers if necessary
    torch_dtype=torch.bfloat16, # Use bfloat16 if your hardware supports it
    # token=config.HUGGINGFACE_TOKEN,
    token=config.HUGGINGFACE_TOKEN
)

model.eval()

# Wrap model with Accelerator for multi-GPU / CPU fallback
model = accelerator.prepare(model)




import re

def extract_classification_with_regex(response_content):
    match = re.search(r"(?:Answer:\s*)?([ABCD])", response_content)  # Make 'Answer:' optional
    if match:
        return match.group(1)  # Extract 'A', 'B', or 'C'
    raise ValueError(f"Unexpected response format: {response_content}")


# --------------------------------------------------------------------------------
# Step 1: Load and process the claim-evidence pairs from the text file
# --------------------------------------------------------------------------------

claims = []
evidences = []

with open(joint_lines_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split("[SEP]")
            if len(parts) == 2:
                claims.append(parts[0].strip())
                evidences.append(parts[1].strip())
            else:
                print(f"Skipping malformed line: {line}")

total_claims = len(claims)
print(f"Total processed claim-evidence pairs: {total_claims}")
if total_claims > 0:
    print(f"Sample Claim: {claims[0]}")
    print(f"Sample Evidence: {evidences[0]}")

# --------------------------------------------------------------------------------
# Step 2: Load progress from the existing output file
# --------------------------------------------------------------------------------

processed_claims = set()
if os.path.exists(output_file):
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            # We assume each line is "claim \t evidence \t classification \t prob"
            # The first element is the claim
            parts = line.strip().split("\t")
            if parts:
                processed_claims.add(parts[0])

print(f"Resuming from {len(processed_claims)} processed claims...")

# --------------------------------------------------------------------------------
# Helper: Classification function
# --------------------------------------------------------------------------------

def classify_claim_evidence(claim, evidence):

    # Construct the prompt
    prompt = (
        f"(Task) Answer the following question:\n\n"
        f"(Input) Facts: {evidence}\n\n"
        f"Statement: {claim}\n\n"
        "Is the statement entailed by the given facts?"
        "(A) false (B) true (C) unproven (D) mixture \n\n"
        "Please answer just A or B or C or D: "
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    # print(f"The input is {inputs} That is all about input")
    # Generate: Return dict with scores to compute log probabilities
    with torch.no_grad():
        gen_output = model.generate(
            **inputs,
            max_new_tokens=50,  # short generation
            temperature=0.0,    # greedy decoding
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Decode the entire output
    prompt_length = inputs["input_ids"].shape[-1]
    new_tokens = gen_output.sequences[0, prompt_length:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f"Raw Response: {output_text}")

    print("finished")
    classification_char = extract_classification_with_regex(output_text)

    # We'll get the per-token log probs
    generated_length = len(gen_output.scores)
    # print(f"Generated tokens: {new_tokens} (Length: {generated_length})")
    sum_logprob = 0.0
    classification_logprob = None

    for i, score_distribution in enumerate(gen_output.scores):
        # score_distribution[0] has shape [vocab_size]
        log_probs = torch.log_softmax(score_distribution[0], dim=-1)

        token_id = new_tokens[i].item()
        token_str = tokenizer.decode([token_id], skip_special_tokens=True)

        # # Accumulate for average logprob if desired
        # sum_logprob += log_probs[token_id].item()

        # If this exact token is the letter we want
        if token_str.strip() == classification_char:  
            classification_logprob = log_probs[token_id].item()
            # If you only need that single logprob, you can break
            break

    if classification_logprob is not None:
        prob = math.exp(classification_logprob)
        print(f"Classification token: {classification_char}, logprob={classification_logprob:.4f}, prob={prob:.4f}")
    else:
        print("Could not find classification token among the generated tokens!")
        # Provide a default probability or raise an exception:
        prob = 0.0

    return classification_char, prob

# Map the classification characters to full labels
classification_map = {"A": "false", "B": "true", "C": "unproven", "D": "mixture"}

# --------------------------------------------------------------------------------
# Step 3: Classify each claim-evidence pair, logging results
# --------------------------------------------------------------------------------

with open(output_file, "a", encoding="utf-8") as output_f, \
     open(error_log_file, "a", encoding="utf-8") as error_log:

    for idx in range(total_claims):
        claim = claims[idx]
        evidence_text = evidences[idx]

        # Skip if already processed
        if claim in processed_claims:
            print(f"Skipping already processed claim {idx + 1}/{total_claims}: {claim}")
            continue

        success = False
        attempts = 0

        while not success and attempts < max_retries:
            try:
                print(f"Processing claim {idx + 1}/{total_claims} (Attempt {attempts + 1}): {claim}")

                classification_char, probability = classify_claim_evidence(claim, evidence_text)

                if classification_char not in classification_map:
                    raise ValueError(f"Unexpected classification: {classification_char}")

                # Convert label
                classification_label = classification_map[classification_char]

                # Write to output
                output_f.write(f"{claim}\t{evidence_text}\t{classification_label}\t{probability:.4f}\n")
                output_f.flush()

                processed_claims.add(claim)
                print(f"Classified claim {idx + 1}: {classification_label} (Probability: {probability:.4f})")
                success = True

            except Exception as e:
                attempts += 1
                print(f"Error processing claim {idx + 1} (Attempt {attempts}): {e}")
                if attempts >= max_retries:
                    # Log the error
                    error_message = f"Failed claim {idx + 1}/{total_claims}: {claim}\n{e}\n"
                    error_log.write(error_message)
                    error_log.flush()
                    print(f"Logged error for claim {idx + 1}.")
            finally:
                # Delay to avoid spamming requests
                time.sleep(throttle_delay)

print(f"Classification completed. {len(processed_claims)} claims classified successfully.")
print(f"Classified claims saved to '{output_file}'.")
print(f"Errors logged to '{error_log_file}'.")
