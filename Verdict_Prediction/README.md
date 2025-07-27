# Verdict Prediction

This module contains Large Language Model (LLM) implementations for classifying fact-checking claims based on retrieved evidence.

## Supported Models

### 🦙 **Llama Models**
- **Llama-405B**: Meta's largest instruct model
- **Llama-70B**: Balanced performance and efficiency
- **Files**: `llama_405b.py`, `llama_70b.py`

### 🌟 **Mistral Models**
- **Mistral-Large**: High-performance instruction-following
- **File**: `mistral_large.py`

### 🔬 **Phi Models**
- **Phi-4**: Microsoft's reasoning-focused model
- **File**: `phi_4.py`

### 🎯 **Qwen Models**
- **Qwen2.5-72B**: Alibaba's multilingual model
- **File**: `qwen_2.5.py`



## Prediction Pipeline

1. **Evidence Formatting**: Format claim-evidence pairs for model input
2. **Prompt Engineering**: Create structured prompts for classification
3. **Model Inference**: Generate predictions with confidence scores
4. **Output Processing**: Extract and standardize predictions
5. **Error Handling**: Manage API failures and retries
6. **Batch Processing**: Process multiple claims efficiently

## Classification Schema

### Standard Classes
- **Supported**: Evidence supports the claim
- **Refuted**: Evidence contradicts the claim  
- **Not Enough Info**: Insufficient evidence for determination

### Extended Classes (dataset-specific)
- **True/False**: Binary classification
- **Multiple degrees**: Fine-grained truth assessments

## Model Organization

```
Verdict_Prediction/
├── Models/                 # Model-specific implementations
│   ├── Local/             # Local model deployments
│   └── API/               # API-based models
├── Prompts/               # Prompt templates and engineering
├── Evaluation/            # Model evaluation and comparison
└── Utils/                 # Shared utilities and helpers
```

## Configuration

Model settings in `config.py`:
- Model paths and cache directories
- API credentials
- Inference parameters (temperature, max tokens)
- Retry and timeout settings 