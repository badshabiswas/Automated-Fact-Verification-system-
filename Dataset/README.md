# Dataset

This folder contains the raw datasets and preprocessing scripts for the fact-checking research project.

## Supported Datasets

### 📊 **Averitec**
- **Description**: Real-world fact-checking dataset with complex evidence requirements
- **Claims**: Multi-domain factual claims requiring evidence from multiple sources
- **Ground Truth**: Supported/Refuted/Not Enough Info

### 📊 **Liar** 
- **Description**: Political statements dataset from PolitiFact
- **Claims**: Political statements and campaign promises
- **Ground Truth**: True/False/Half-True/Mostly-True/Pants-on-Fire/Barely-True

### 📊 **PubHealth**
- **Description**: Health-related claims requiring scientific evidence
- **Claims**: Medical and health-related statements
- **Ground Truth**: True/False/Unproven/Mixture

### 📊 **SciFact**
- **Description**: Scientific fact verification dataset
- **Claims**: Scientific claims requiring research paper evidence
- **Ground Truth**: Supported/Refuted/Not Enough Info

## Directory Structure

```
Dataset/
├── Raw/                    # Original dataset files
│   ├── Averitec/
│   ├── Liar/
│   ├── PubHealth/
│   └── SciFact/
├── Processed/              # Preprocessed and cleaned datasets
├── Statistics/             # Dataset statistics and analysis
└── Preprocessing/          # Scripts for data cleaning and preparation
```

## Configuration

Dataset paths are configured through the main `config.py` file using:
```python
config.get_dataset_path('dataset_name', 'subdirectory', 'filename')
``` 