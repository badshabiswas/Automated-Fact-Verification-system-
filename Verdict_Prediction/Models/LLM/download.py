# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch
# # Set cache directory
# cache_dir = config.CACHE_DIR

# # Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(
#     "google/gemma-3-27b-it",
#     cache_dir=cache_dir,  # Custom cache path
#     token=config.HUGGINGFACE_TOKEN
# )

# # Load model
# model = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-3-27b-it",
#     cache_dir=cache_dir,  # Custom cache path
#     device_map="auto",
#     offload_folder="offload",
#     torch_dtype=torch.bfloat16,
#     token=config.HUGGINGFACE_TOKEN
# )


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
sys.path.append('.')
from config import Config

# Initialize configuration
Config.validate_config()
config = Config()

# Set custom cache directory
cache_dir = config.CACHE_DIR

# Hugging Face token (from environment variable for security)
hf_token = config.HUGGINGFACE_TOKEN

# Model ID
model_id = "Qwen/Qwen2.5-72B-Instruct"

# Download and cache tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    token=hf_token
)

# Download and cache model
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    device_map="auto",
    offload_folder="offload",
    torch_dtype=torch.bfloat16,
    token=hf_token
)

print("✅ Model and tokenizer downloaded successfully!")
