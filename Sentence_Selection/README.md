# Sentence Selection

This module extracts the most relevant sentences from retrieved documents to serve as evidence for claim verification.

## Selection Methods

### 🎯 **Sentence Embedding Similarity**
- **Files**: `sentence_selection_*.py`
- **Model**: SPICED (Sentence-level Proximity In Clinical Evidence Detection)
- **Method**: Compute cosine similarity between claim and candidate sentences
- **Output**: Top-K most relevant sentences per claim

### 📊 **Multi-Source Integration**
- **Wikipedia**: `sentence_retrieve_wiki.py`
- **PubMed**: `sentence_retrieve_pubmed.py`
- **Google Search**: `sentence_retrieve_google.py`
- **Combined**: `sentence_merge.py`

## Process Flow

1. **Document Loading**: Load retrieved documents from previous stage
2. **Sentence Segmentation**: Split documents into individual sentences
3. **Embedding**: Encode sentences using sentence transformers
4. **Similarity Computation**: Calculate claim-sentence similarities
5. **Ranking**: Rank sentences by relevance scores
6. **Selection**: Select top-K sentences (typically 5-10)
7. **Formatting**: Create claim-evidence pairs in standardized format

## Output Format

Selected sentences are formatted as:
```
[CLAIM] [SEP] [EVIDENCE_SENTENCE_1] [EVIDENCE_SENTENCE_2] ... [EVIDENCE_SENTENCE_K]
```

## Quality Control

- **Deduplication**: Remove duplicate sentences across sources
- **Length Filtering**: Filter out very short or very long sentences
- **Relevance Thresholding**: Only include sentences above similarity threshold
- **Source Diversity**: Ensure evidence comes from multiple sources when possible

## Configuration

Configure selection parameters:
- Sentence embedding model
- Number of top sentences (K)
- Similarity thresholds
- Maximum evidence length 