# API-Based Models

This directory is reserved for API-based Large Language Model implementations.

## Purpose

While the project primarily uses local model deployments (in `../Models/LLM/`), this directory can be used for:

- **External API Integrations**: Models accessed via REST APIs
- **Cloud-Based Models**: Services like OpenAI GPT, Anthropic Claude, etc.
- **Third-Party Model Services**: Any external model providers

## Structure

When adding API-based models, organize them as:

```
API/
├── openai/
│   ├── gpt4_api.py
│   └── gpt3_api.py
├── anthropic/
│   ├── claude_api.py
│   └── utils.py
└── custom_provider/
    └── custom_model_api.py
```

## Configuration

API-based models should use the centralized configuration system:

```python
from config import Config
config = Config()

# Add API-specific configuration to config.py as needed
api_key = config.YOUR_API_KEY
model_endpoint = config.YOUR_API_ENDPOINT
```

## Implementation Guidelines

1. **Use environment variables** for API keys and endpoints
2. **Follow the same input/output format** as local models
3. **Implement retry logic** for API failures
4. **Add rate limiting** to respect API quotas
5. **Log API usage** for monitoring and debugging

## Current Status

Currently, this directory is empty as the project focuses on local model deployments for better reproducibility and control. 