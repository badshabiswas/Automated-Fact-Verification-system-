# 🔬 Fact-Checking Research Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Research](https://img.shields.io/badge/research-fact--checking-green.svg)](https://github.com/yourusername/fact-checking-research)

A comprehensive, anonymized research framework for automated fact verification using state-of-the-art large language models and advanced evidence retrieval techniques.

## 🔄 Research Pipeline

This framework implements a systematic fact-checking pipeline:

```
📊 Dataset → 🔍 Evidence Retrieval → 📝 Sentence Selection → ⚖️ Verdict Prediction → 📈 Disagreement Analysis
```

### **1. 📊 Dataset Management**
- Support for multiple fact-checking datasets (Averitec, Liar, PubHealth, SciFact)
- Automated data preprocessing and validation
- Configurable dataset paths and formats

### **2. 🔍 Evidence Retrieval**
- **BM25 Retrieval**: Traditional keyword-based document retrieval
- **Semantic Search**: Dense retrieval using biomedical embeddings
- **Web Integration**: Real-time evidence from Google, Wikipedia, PubMed
- **Knowledge Graphs**: Advanced evidence discovery through graph structures

### **3. 📝 Sentence Selection**
- Multi-source evidence extraction and ranking
- Biomedical embedding models optimized for scientific claims
- Intelligent sentence filtering and deduplication

### **4. ⚖️ Verdict Prediction**
- **Large Language Models**: Llama (70B, 405B), Mistral, Phi-4, Qwen 2.5
- **Multi-Source Evidence**: Integrated classification across different evidence sources
- **Dataset-Specific**: Specialized configurations for each dataset

### **5. 📈 Disagreement Analysis**
- Inter-model agreement and confidence analysis
- Uncertainty quantification and reliability assessment
- Professional visualization and statistical reporting

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for large models)
- 16GB+ RAM (32GB+ recommended for large models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fact-checking-research.git
   cd fact-checking-research
   ```

2. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r Verdict_Prediction/Models/LLM/requirements.txt
   pip install python-dotenv retriv groq matplotlib seaborn
   ```

4. **Configure settings**
   ```bash
   cp env.template .env
   # Edit .env with your paths and API keys
   ```

### Basic Usage

1. **Configure your environment** in `.env`:
   ```bash
   BASE_DATA_DIR=/path/to/your/data
   CACHE_DIR=/path/to/model/cache
   HUGGINGFACE_TOKEN=your_token_here
   ```

2. **Run evidence retrieval**:
   ```bash
   cd Evidence_Retrieval/BM25/
   python BM25.py
   ```

3. **Execute fact verification**:
   ```bash
   cd Verdict_Prediction/Models/LLM/
   python Llama-70B.py
   ```

4. **Analyze results**:
   ```bash
   cd Disagreement_Analysis/
   python source_confidence_analysis.py
   ```

## 📚 Documentation

- **[Workflow Guide](WORKFLOW_GUIDE.md)**: Complete step-by-step research pipeline
- **[Configuration](config.py)**: Centralized settings and paths
- **[Environment Setup](env.template)**: Template for environment variables

## 🗂️ Project Structure

```
fact-checking-research/
├── config.py                      # Centralized configuration
├── env.template                   # Environment setup template
├── Dataset/                       # 📊 Data management
│   └── Raw/                      # Dataset storage
├── Evidence_Retrieval/           # 🔍 Document discovery
│   ├── BM25/                     # Keyword-based retrieval
│   ├── Semantic/                 # Embedding-based search
│   ├── Web_Search/               # External APIs
│   └── Utils/                    # Helper scripts
├── Sentence_Selection/           # 📝 Evidence extraction
│   └── Multi_Source/             # Source integration
├── Verdict_Prediction/           # ⚖️ Model predictions
│   ├── Models/                   # Model implementations
│   │   └── LLM/                 # Large language models (Llama, Mistral, Phi-4, Qwen)
│   ├── API/                      # API-based models
│   └── Utils/                    # Shared utilities
└── Disagreement_Analysis/        # 📈 Analysis and comparison
    └── source_confidence_analysis.py  # Advanced analytics
```

## 🔬 Research Features

### **Advanced Models**
- **Large Language Models**: Latest generation models for fact verification
- **Biomedical Embeddings**: Specialized models for scientific fact-checking
- **Multi-Source Integration**: Evidence from academic, web, and knowledge sources

### **Comprehensive Analysis**
- **Uncertainty Quantification**: Model confidence and reliability metrics
- **Source Reliability**: Comparative analysis across evidence sources
- **Professional Visualization**: Research-grade plots and statistical reports

### **Reproducible Research**
- **Configurable Pipeline**: All settings managed through environment variables
- **Documented Workflow**: Step-by-step research methodology
- **Clean Architecture**: Modular, maintainable codebase

## 📊 Supported Datasets

| Dataset | Domain | Classes | Description |
|---------|--------|---------|-------------|
| **SciFact** | Scientific | 3-class | Scientific claim verification |
| **Averitec** | Multi-domain | 3-class | Real-world fact-checking |
| **PubHealth** | Health | 4-class | Medical claim verification |
| **Liar** | Political | 6-class | Political statement analysis |

## 🤖 Supported Models

### **Large Language Models**
- Meta Llama (70B, 405B)
- Mistral Large Instruct
- Microsoft Phi-4
- Alibaba Qwen 2.5

### **Specialized Approaches**
- Multi-dataset configurations
- BioClinical embeddings for scientific domains
- BM25 + Semantic hybrid retrieval

## 📈 Performance

The framework achieves competitive performance across multiple benchmarks:
- Advanced uncertainty quantification
- Source-specific optimization
- Professional statistical analysis

*Detailed results and comparisons available in research publications.*

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Follow the existing code style** and anonymization patterns
4. **Add tests** for new functionality
5. **Update documentation** as needed
6. **Submit a pull request**

### Development Guidelines

- All paths must use the configuration system (`config.py`)
- No hardcoded personal or institutional data
- Follow the established workflow structure
- Include proper error handling and logging

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{fact_checking_framework,
  title={Fact-Checking Research Framework},
  author={[Your Name/Institution]},
  year={2024},
  url={https://github.com/yourusername/fact-checking-research},
  note={A comprehensive framework for automated fact verification}
}
```

## 🔗 Related Work

- [SciFact Dataset](https://aclanthology.org/2020.emnlp-main.609/)
- [Averitec Challenge](https://fever.ai/task.html)
- [PubHealth](https://github.com/neemakot/Health-Fact-Checking)
- [Liar Dataset](https://aclanthology.org/W17-5511/)

## ⚠️ Important Notes

- **Configuration Required**: Set up your `.env` file before running
- **Resource Intensive**: Large models require significant computational resources
- **API Limits**: External services may have rate limits
- **Responsible Use**: Ensure ethical use of fact-checking capabilities

---

**Built for reproducible, ethical fact-checking research** 🔬✨ 