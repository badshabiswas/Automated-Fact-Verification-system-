"""
Enhanced Document Retrieval for Fact Verification

Finds the top K most similar documents for claims using biomedical embeddings.
Optimized for scientific and medical fact-checking with BioClinical models.
"""

import sys
sys.path.append('..')
from config import Config

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
import os
import pickle
from pathlib import Path
import json

# Initialize configuration
config = Config()

class EnhancedDocumentRetriever:
    """Enhanced document retrieval using biomedical embeddings"""
    
    def __init__(self, 
                 embedding_model="kamalkraj/BioSimCSE-BioLinkBERT-BASE",
                 top_k=10):
        self.embedding_model_name = embedding_model
        self.top_k = top_k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load embedding model
        self.model = SentenceTransformer(
            embedding_model,
            cache_folder=config.CACHE_DIR,
            device=self.device
        )
        
        print(f"✅ Loaded embedding model: {embedding_model}")
        print(f"📱 Using device: {self.device}")
        
        # Storage for document embeddings
        self.document_embeddings = None
        self.document_texts = None
        self.document_metadata = None

    def load_document_corpus(self, corpus_name='pubmed'):
        """Load document corpus for retrieval"""
        try:
            if corpus_name.lower() == 'pubmed':
                corpus_path = os.path.join(config.BASE_DATA_DIR, 'Evidence_Retrieval', 'Corpus', 'pubmed_abstracts.csv')
            elif corpus_name.lower() == 'wikipedia':
                corpus_path = os.path.join(config.BASE_DATA_DIR, 'Evidence_Retrieval', 'Corpus', 'wikipedia_articles.csv')
            else:
                corpus_path = os.path.join(config.BASE_DATA_DIR, 'Evidence_Retrieval', 'Corpus', f'{corpus_name}_corpus.csv')
            
            if not os.path.exists(corpus_path):
                print(f"⚠️ Corpus file not found: {corpus_path}")
                print("💡 Please ensure your document corpus is available in the Evidence_Retrieval/Corpus/ directory")
                return False
            
            # Load document corpus
            df = pd.read_csv(corpus_path)
            
            # Extract text and metadata
            if 'abstract' in df.columns:
                self.document_texts = df['abstract'].fillna('').tolist()
            elif 'text' in df.columns:
                self.document_texts = df['text'].fillna('').tolist()
            elif 'content' in df.columns:
                self.document_texts = df['content'].fillna('').tolist()
            else:
                print(f"❌ No text column found in corpus. Available columns: {df.columns.tolist()}")
                return False
            
            # Store metadata
            self.document_metadata = df.to_dict('records')
            
            print(f"📚 Loaded {len(self.document_texts)} documents from {corpus_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading corpus: {e}")
            return False

    def encode_document_corpus(self, corpus_name='pubmed', force_recompute=False):
        """Encode document corpus and cache embeddings"""
        
        # Check for cached embeddings
        cache_path = os.path.join(config.CACHE_DIR, f'{corpus_name}_embeddings.pkl')
        
        if not force_recompute and os.path.exists(cache_path):
            print(f"📦 Loading cached embeddings from {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.document_embeddings = cache_data['embeddings']
                    self.document_texts = cache_data['texts']
                    self.document_metadata = cache_data['metadata']
                print(f"✅ Loaded {len(self.document_embeddings)} cached embeddings")
                return True
            except Exception as e:
                print(f"⚠️ Error loading cached embeddings: {e}")
        
        # Load corpus if not already loaded
        if self.document_texts is None:
            if not self.load_document_corpus(corpus_name):
                return False
        
        # Encode documents
        print(f"🔄 Encoding {len(self.document_texts)} documents...")
        print("⏳ This may take several minutes for large corpora...")
        
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(self.document_texts), batch_size):
            batch_texts = self.document_texts[i:i+batch_size]
            batch_embeddings = self.model.encode(
                batch_texts,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            all_embeddings.append(batch_embeddings)
            
            if i % (batch_size * 10) == 0:
                print(f"📊 Processed {i + len(batch_texts)}/{len(self.document_texts)} documents")
        
        # Concatenate all embeddings
        self.document_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Cache embeddings
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        cache_data = {
            'embeddings': self.document_embeddings,
            'texts': self.document_texts,
            'metadata': self.document_metadata
        }
        
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"💾 Cached embeddings to {cache_path}")
        print(f"✅ Document encoding complete!")
        return True

    def retrieve_documents(self, claims, output_file=None):
        """Retrieve top-K documents for each claim"""
        
        if self.document_embeddings is None:
            print("❌ Document corpus not encoded. Please run encode_document_corpus() first.")
            return None
        
        print(f"🔍 Retrieving top-{self.top_k} documents for {len(claims)} claims...")
        
        # Encode queries
        query_embeddings = self.model.encode(
            claims,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        
        # Compute similarities
        similarities = util.semantic_search(
            query_embeddings, 
            self.document_embeddings, 
            top_k=self.top_k
        )
        
        # Format results
        retrieval_results = []
        for i, claim in enumerate(claims):
            claim_results = {
                'claim': claim,
                'retrieved_documents': []
            }
            
            for result in similarities[i]:
                doc_idx = result['corpus_id']
                similarity_score = result['score']
                
                doc_info = {
                    'document_id': doc_idx,
                    'similarity_score': similarity_score,
                    'text': self.document_texts[doc_idx],
                    'metadata': self.document_metadata[doc_idx] if self.document_metadata else None
                }
                
                claim_results['retrieved_documents'].append(doc_info)
            
            retrieval_results.append(claim_results)
        
        # Save results if requested
        if output_file:
            output_path = os.path.join(config.OUTPUT_DIR, 'Evidence_Retrieval', output_file)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(retrieval_results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Retrieval results saved to: {output_path}")
        
        return retrieval_results

    def evaluate_retrieval_quality(self, claims, ground_truth_docs=None):
        """Evaluate retrieval quality with metrics"""
        results = self.retrieve_documents(claims)
        
        if ground_truth_docs and len(ground_truth_docs) == len(claims):
            # Calculate retrieval metrics
            total_relevant_found = 0
            total_relevant = len(ground_truth_docs)
            
            for i, claim_results in enumerate(results):
                retrieved_ids = {doc['document_id'] for doc in claim_results['retrieved_documents']}
                relevant_ids = set(ground_truth_docs[i]) if isinstance(ground_truth_docs[i], list) else {ground_truth_docs[i]}
                
                relevant_found = len(retrieved_ids.intersection(relevant_ids))
                total_relevant_found += relevant_found
            
            recall_at_k = total_relevant_found / total_relevant if total_relevant > 0 else 0
            print(f"📊 Recall@{self.top_k}: {recall_at_k:.4f}")
        
        # Calculate average similarity scores
        all_scores = []
        for claim_results in results:
            scores = [doc['similarity_score'] for doc in claim_results['retrieved_documents']]
            all_scores.extend(scores)
        
        avg_similarity = np.mean(all_scores)
        print(f"📊 Average Similarity Score: {avg_similarity:.4f}")
        
        return results

def load_claims_for_retrieval(dataset_name='scifact'):
    """Load claims from dataset for retrieval"""
    try:
        dataset_path = config.get_dataset_path(dataset_name, '', f'{dataset_name}_dataset.csv')
        if not os.path.exists(dataset_path):
            dataset_path = config.get_dataset_path(dataset_name, 'Dev', f'{dataset_name}_no-nei_dataset.csv')
        
        df = pd.read_csv(dataset_path, index_col=[0])
        claims = df.claim.tolist()
        
        print(f"📋 Loaded {len(claims)} claims from {dataset_name}")
        return claims
        
    except Exception as e:
        print(f"❌ Error loading claims: {e}")
        return []

def main():
    """Main function for enhanced document retrieval"""
    print("🚀 Starting Enhanced Document Retrieval...")
    
    # Initialize retriever
    retriever = EnhancedDocumentRetriever(top_k=10)
    
    # Encode document corpus (PubMed example)
    if retriever.encode_document_corpus('pubmed'):
        
        # Load claims for retrieval
        claims = load_claims_for_retrieval('scifact')
        
        if claims:
            # Retrieve documents
            results = retriever.retrieve_documents(
                claims[:50],  # Process first 50 claims for demo
                output_file='enhanced_retrieval_results.json'
            )
            
            print("✅ Enhanced document retrieval complete!")
        else:
            print("❌ No claims loaded for retrieval")
    else:
        print("❌ Failed to encode document corpus")

if __name__ == "__main__":
    main() 