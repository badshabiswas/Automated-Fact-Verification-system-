# Evidence Retrieval

This module handles the retrieval of relevant documents and evidence for fact-checking claims.

## Retrieval Methods

### 🔍 **BM25 Retrieval**
- **File**: `bm25_retrieval.py`
- **Description**: Traditional keyword-based retrieval using BM25 algorithm
- **Use Case**: Fast baseline retrieval for lexically similar documents
- **Output**: Ranked list of document IDs with relevance scores

### 🧠 **Semantic Retrieval**
- **File**: `semantic_retrieval.py`
- **Description**: Dense retrieval using sentence transformers
- **Models**: BioSimCSE-BioLinkBERT-BASE for biomedical domains
- **Use Case**: Capturing semantic similarity beyond keyword matching
- **Output**: Ranked list of document IDs with cosine similarity scores

### 🔗 **Elasticsearch Integration**
- **Files**: `elasticsearch_search.py`, `elasticsearch_negated.py`
- **Description**: Real-time search using Elasticsearch
- **Use Case**: Interactive search and exploration
- **Output**: Structured search results with metadata

### 🌐 **Web Search Integration**
- **Files**: `google_search.py`, `wikipedia_search.py`
- **Description**: Integration with external search APIs
- **Use Case**: Retrieving fresh evidence not in static corpora
- **Output**: Web search results with URLs and snippets

## Pipeline

1. **Index Building**: Create searchable indices from document corpora
2. **Query Processing**: Clean and prepare claims for retrieval
3. **Retrieval**: Execute search using selected method
4. **Post-processing**: Filter and rank results
5. **Output**: Generate standardized evidence files

## Configuration

Configure retrieval parameters in `config.py`:
- Embedding models
- Index directories
- Search parameters (top-k, similarity thresholds)
- API credentials for external services 