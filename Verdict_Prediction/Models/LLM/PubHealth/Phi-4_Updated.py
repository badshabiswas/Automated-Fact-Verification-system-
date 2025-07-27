#!/usr/bin/env python
"""
phi4_pubhealth_classification.py
Classify PubHealth claims using microsoft/phi-4 in a batch manner.
The script mirrors the structure used for the LIAR dataset script.
"""

import os, math, re, time
import torch
from accelerate import Accelerator
import sys
sys.path.append('..')
from config import Config

# Validate configuration and initialize config
Config.validate_config()
config = Config()
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------
# Configuration
# --------------------------------------------------
input_files = [
    config.get_dataset_path('PubHealth\'.lower(), \'Evidence Sentences/Merged/merged_evidence.txt\'),
]

output_files = [
    config.get_dataset_path('PubHealth\'.lower(), \'Result/Merged/Phi-4/PubHealth_merged_classified_claims.txt\'),
]

error_log_files = [
    config.get_dataset_path('PubHealth\'.lower(), \'Result/Merged/Phi-4/PubHealth_merged_error_log.txt\'),
]

cache_dir = config.CACHE_DIR
model_path = "microsoft/phi-4"
auth_token = config.HUGGINGFACE_TOKEN

max_retries = config.MAX_RETRIES
throttle_delay = config.THROTTLE_DELAY  # seconds between retries

# --------------------------------------------------
# Model & Tokenizer
# --------------------------------------------------
accelerator = Accelerator()
device = accelerator.device

print(f"Loading tokenizer: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(
    model_path, cache_dir=cache_dir, token=auth_token
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model: {model_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    device_map="auto",
    offload_folder="offload",
    torch_dtype=torch.bfloat16,
    token=auth_token,
)
model.eval()
model = accelerator.prepare(model)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
classification_map = {
    "A": "false",
    "B": "true",
    "C": "unproven",
    "D": "mixture",
}

regex_pattern = re.compile(r"(?:Answer:\s*)?([ABCD])")  # capture A–D

def extract_classification(response: str) -> str:
    m = regex_pattern.search(response)
    if not m:
        raise ValueError(f"Unexpected response: {response}")
    return m.group(1)

def classify_claim_evidence(claim: str, evidence: str):
    """Return (letter, probability)."""
    prompt = (
        "<|im_start|>system<|im_sep|>\n"
        "You are a highly knowledgeable AI model trained to verify statements based on given facts.\n"
        "<|im_end|>\n"
        "<|im_start|>user<|im_sep|>\n"
        f"(Task) Answer the following question:\n\n"
        f"(Input) Facts: {evidence}\n\n"
        f"Statement: {claim}\n\n"
        "Is the statement entailed by the given facts?\n"
        "(A) false (B) true (C) unproven (D) mixture\n\n"
        "Please answer just A or B or C or D:\n"
        "<|im_end|>\n"
        "<|im_start|>assistant<|im_sep|>"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        generation = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    prompt_len = inputs["input_ids"].shape[-1]
    new_tokens = generation.sequences[0, prompt_len:]
    output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    letter = extract_classification(output_text)

    # probability of the letter token
    prob = 0.0
    for i, score_dist in enumerate(generation.scores):
        token_id = new_tokens[i].item()
        token_str = tokenizer.decode([token_id], skip_special_tokens=True).strip()
        wanted = "(" + letter
        if token_str in (letter, wanted):
            logprob = torch.log_softmax(score_dist[0], dim=-1)[token_id]
            prob = math.exp(logprob.item())
            break

    return letter, prob

# --------------------------------------------------
# Main loop
# --------------------------------------------------
for in_path, out_path, err_path in zip(input_files, output_files, error_log_files):
    print(f"\nProcessing file: {in_path}")
    claims, evidences = [], []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            parts = line.split("[SEP]")
            if len(parts) == 2:
                claims.append(parts[0].strip())
                evidences.append(parts[1].strip())

    total = len(claims)
    print(f"Total claim-evidence pairs: {total}")

    processed = set()
    if os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                part = line.split("\t", 1)
                if part:
                    processed.add(part[0])

    with open(out_path, "a", encoding="utf-8") as out_f, \
         open(err_path, "a", encoding="utf-8") as err_f:

        for idx, (claim, evidence) in enumerate(zip(claims, evidences), 1):
            if claim in processed:
                print(f"[{idx}/{total}] skip")
                continue

            for attempt in range(1, max_retries + 1):
                try:
                    print(f"[{idx}/{total}] attempt {attempt}")
                    letter, prob = classify_claim_evidence(claim, evidence)
                    label = classification_map[letter]
                    out_f.write(
                        f"{claim}\t{evidence}\t{label}\t{prob:.4f}\n"
                    )
                    out_f.flush()
                    processed.add(claim)
                    break
                except Exception as e:
                    err_f.write(f"Failed on claim {idx}: {claim}\n{e}\n")
                    err_f.flush()
                    if attempt < max_retries:
                        time.sleep(throttle_delay)
                    else:
                        print(f"Giving up on claim {idx}")

print("All files processed successfully!")
