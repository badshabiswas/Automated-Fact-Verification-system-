import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Centralized configuration for the fact-checking research project.
    All hardcoded paths and settings should be configured here.
    """
    
    # Base directories - can be overridden by environment variables
    BASE_DATA_DIR = os.getenv('BASE_DATA_DIR', '/data/fact_checking')
    CACHE_DIR = os.getenv('CACHE_DIR', '/cache/models')
    SCRATCH_DIR = os.getenv('SCRATCH_DIR', '/tmp/scratch')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR', '/output')
    
    # API Configuration
    HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    
    # Model Configuration
    DEFAULT_EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'kamalkraj/BioSimCSE-BioLinkBERT-BASE')
    DEFAULT_LLM_MODEL = os.getenv('LLM_MODEL', 'mistralai/Mistral-Large-Instruct-2411')
    
    # Dataset paths (now in workflow structure)
    DATASETS = {
        'averitec': {
            'base_dir': Path(BASE_DATA_DIR) / 'Dataset' / 'Raw' / 'Averitec',
            'claims_file': 'claims_labels_unique.csv',
            'evidence_dir': 'Evidence Sentence',
            'results_dir': 'Result',
            'relevant_doc_dir': 'Relevant Doc'
        },
        'liar': {
            'base_dir': Path(BASE_DATA_DIR) / 'Dataset' / 'Raw' / 'Liar',
            'evidence_dir': 'Evidence Sentences',
            'results_dir': 'Result',
            'relevant_doc_dir': 'Relevant Doc'
        },
        'pubhealth': {
            'base_dir': Path(BASE_DATA_DIR) / 'Dataset' / 'Raw' / 'PubHealth',
            'evidence_dir': 'Evidence Sentences',
            'results_dir': 'Result',
            'relevant_doc_dir': 'Relevant Doc'
        },
        'scifact': {
            'base_dir': Path(BASE_DATA_DIR) / 'Dataset' / 'Raw' / 'SciFact',
            'evidence_dir': 'Evidence Sentences',
            'results_dir': 'Result',
            'relevant_doc_dir': 'Relevant Doc'
        }
    }

    # Workflow-specific directories
    EVIDENCE_RETRIEVAL_DIR = Path(BASE_DATA_DIR) / 'Evidence_Retrieval'
    SENTENCE_SELECTION_DIR = Path(BASE_DATA_DIR) / 'Sentence_Selection'
    VERDICT_PREDICTION_DIR = Path(BASE_DATA_DIR) / 'Verdict_Prediction'
    DISAGREEMENT_ANALYSIS_DIR = Path(BASE_DATA_DIR) / 'Disagreement_Analysis'
    
    # Processing configuration
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    THROTTLE_DELAY = float(os.getenv('THROTTLE_DELAY', '1.0'))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', '1000'))
    
    # Model specific settings
    TEMPERATURE = float(os.getenv('TEMPERATURE', '0.0'))
    MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', '50'))
    
    # SLURM/Cluster configuration
    SLURM_PARTITION = os.getenv('SLURM_PARTITION', 'gpu')
    SLURM_QOS = os.getenv('SLURM_QOS', 'gpu')
    SLURM_TIME = os.getenv('SLURM_TIME', '24:00:00')
    SLURM_MEMORY = os.getenv('SLURM_MEMORY', '100GB')
    SLURM_GPU_TYPE = os.getenv('SLURM_GPU_TYPE', 'A100.80gb:1')
    
    @classmethod
    def get_dataset_path(cls, dataset_name, subdir='', filename=''):
        """
        Get a path for a specific dataset, subdirectory, and filename.
        
        Args:
            dataset_name: One of 'averitec', 'liar', 'pubhealth', 'scifact'
            subdir: Subdirectory within the dataset (e.g., 'Evidence Sentence')
            filename: Optional filename
            
        Returns:
            Path object for the requested location
        """
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        base_path = cls.DATASETS[dataset_name]['base_dir']
        
        if subdir:
            base_path = base_path / subdir
        
        if filename:
            base_path = base_path / filename
            
        return base_path
    
    @classmethod
    def ensure_directories(cls):
        """Create necessary directories if they don't exist."""
        directories = [
            cls.BASE_DATA_DIR,
            cls.CACHE_DIR,
            cls.SCRATCH_DIR,
            cls.OUTPUT_DIR
        ]
        
        for dataset_info in cls.DATASETS.values():
            directories.append(dataset_info['base_dir'])
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present."""
        required_vars = {
            'HUGGINGFACE_TOKEN': cls.HUGGINGFACE_TOKEN,
        }

        missing_vars = [var for var, value in required_vars.items() if not value]

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}. "
                f"Please check your .env file or environment configuration."
            )

# Convenience function to get configured paths
def get_config():
    """Get a configured Config instance."""
    return Config() 