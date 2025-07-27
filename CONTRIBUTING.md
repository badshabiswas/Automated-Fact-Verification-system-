# Contributing to Fact-Checking Research Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to this research project.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Git
- Understanding of fact-checking and NLP concepts

### Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/fact-checking-research.git
   cd fact-checking-research
   ```

2. **Set up development environment**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r Verdict_Prediction/Models/LLM/requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp env.template .env
   # Configure your .env file
   ```

## 📋 Contribution Guidelines

### Code Standards

#### **Anonymization Requirements**
- ✅ NO hardcoded personal paths or usernames
- ✅ NO institutional references (`/scratch/`, `/projects/`)
- ✅ NO hardcoded API keys or tokens
- ✅ ALL paths must use `config.py` system

```python
# ✅ Good - Use configuration
from config import Config
config = Config()
data_path = config.get_dataset_path('scifact', 'Evidence', 'file.txt')

# ❌ Bad - Hardcoded path
data_path = "/scratch/username/data/file.txt"
```

#### **Code Structure**
- Follow the established workflow organization
- Use descriptive variable and function names
- Include proper error handling
- Add comprehensive docstrings

#### **Import Standards**
```python
# For scripts in subdirectories
import sys
sys.path.append('..')
from config import Config

# Initialize configuration
config = Config()
```

### **Workflow Integration**

All contributions must fit within the research pipeline:

```
📊 Dataset → 🔍 Evidence Retrieval → 📝 Sentence Selection → ⚖️ Verdict Prediction → 📈 Analysis
```

Place code in appropriate directories:
- **Evidence_Retrieval/**: Document discovery and retrieval
- **Sentence_Selection/**: Evidence extraction and ranking
- **Verdict_Prediction/**: Model implementations and classification
- **Disagreement_Analysis/**: Result analysis and comparison

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear description** of the issue
2. **Steps to reproduce** the problem
3. **Expected vs actual behavior**
4. **Environment information** (Python version, OS, etc.)
5. **Error logs** if applicable

Use the issue template:
```markdown
## Bug Description
[Clear description of the bug]

## Steps to Reproduce
1. [First step]
2. [Second step]
3. [...]

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
- OS: [Windows/Mac/Linux]
- Python: [3.8/3.9/3.10/etc.]
- GPU: [CUDA version if applicable]
```

## 💡 Feature Requests

For new features, please:

1. **Check existing issues** to avoid duplicates
2. **Describe the motivation** for the feature
3. **Explain the use case** in research context
4. **Consider the impact** on the workflow pipeline

## 🔄 Pull Request Process

### Before Submitting
- [ ] Code follows anonymization guidelines
- [ ] All tests pass (if applicable)
- [ ] Documentation is updated
- [ ] Changes fit within workflow structure
- [ ] No sensitive data included

### Pull Request Template
```markdown
## Description
[Brief description of changes]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Code tested locally
- [ ] No sensitive data included
- [ ] Configuration system used properly

## Related Issues
Closes #[issue number]
```

### Review Process
1. **Automated checks** will run
2. **Maintainer review** for code quality and research relevance
3. **Discussion** if changes are needed
4. **Merge** once approved

## 📚 Documentation

### Code Documentation
- Use clear, descriptive docstrings
- Include parameter and return type information
- Provide usage examples for complex functions

```python
def classify_claims(claims, evidence, model_name="deberta-v3"):
    """
    Classify fact-checking claims using specified model.
    
    Args:
        claims (list): List of claim strings
        evidence (list): List of evidence strings
        model_name (str): Name of the model to use
        
    Returns:
        tuple: (predictions, confidence_scores)
        
    Example:
        >>> predictions, scores = classify_claims(
        ...     ["Climate change is real"], 
        ...     ["Scientific consensus confirms climate change"]
        ... )
    """
```

### README Updates
- Update usage examples for new features
- Add new models to supported models list
- Update performance benchmarks if applicable

## 🎯 Research Focus Areas

We particularly welcome contributions in:

### **Model Integration**
- New LLM implementations
- Specialized domain models
- Hybrid retrieval approaches

### **Dataset Support**
- Additional fact-checking datasets
- Preprocessing pipelines
- Evaluation metrics

### **Analysis Tools**
- Advanced uncertainty quantification
- Bias detection and mitigation
- Cross-dataset evaluation

### **Performance Optimization**
- Memory efficiency improvements
- GPU utilization optimization
- Caching mechanisms

## 🔒 Security and Ethics

### Data Privacy
- Never commit sensitive or personal data
- Use synthetic examples in documentation
- Respect dataset licensing terms

### Ethical Considerations
- Consider bias implications of changes
- Ensure responsible fact-checking practices
- Document limitations and potential misuse

## 📞 Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub discussions for research questions
- **Documentation**: Check existing docs and workflow guide

## 🏷️ Release Process

### Version Numbering
We use semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking changes to API or workflow
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, minor improvements

### Release Criteria
- All tests passing
- Documentation updated
- No known security issues
- Backward compatibility maintained (when possible)

---

Thank you for contributing to advancing fact-checking research! 🔬✨ 