"""
Source Confidence Analysis for Fact Verification

Analyzes confidence levels and reliability of different evidence sources.
Useful for understanding source quality and evidence strength.
"""

import sys
sys.path.append('..')
from config import Config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# Initialize configuration
config = Config()

class SourceConfidenceAnalyzer:
    """Analyzer for source confidence and reliability in fact verification"""
    
    def __init__(self):
        self.results_data = {}
        self.confidence_metrics = {}
        
    def load_evidence_sources(self, dataset_name='scifact'):
        """Load evidence from different sources for analysis"""
        sources = ['Google', 'Pubmed', 'Wikipedia', 'Merged']
        
        for source in sources:
            try:
                # Try to load evidence files
                evidence_path = config.get_dataset_path(
                    dataset_name, 
                    f'Evidence Sentences/{source}/Final Result',
                    f'{dataset_name}_{source.lower()}_evidence.txt'
                )
                
                if os.path.exists(evidence_path):
                    evidence_data = []
                    with open(evidence_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                evidence_data.append({
                                    'claim': parts[0],
                                    'evidence': parts[1],
                                    'source': source
                                })
                    
                    self.results_data[source] = evidence_data
                    print(f"📚 Loaded {len(evidence_data)} evidence entries from {source}")
                
            except Exception as e:
                print(f"⚠️ Could not load {source} evidence: {e}")
        
        return len(self.results_data) > 0
    
    def load_prediction_results(self, dataset_name='scifact'):
        """Load prediction results from different models and sources"""
        model_sources = ['Llama-405B', 'Llama-70B', 'Mistral', 'Phi-4', 'Qwen']
        evidence_sources = ['Google', 'Pubmed', 'Wikipedia', 'Merged']
        
        prediction_results = {}
        
        for model in model_sources:
            for source in evidence_sources:
                try:
                    result_path = config.get_dataset_path(
                        dataset_name,
                        f'Result/{source}/{model}',
                        f'{dataset_name}_{source.lower()}_classified_claims.txt'
                    )
                    
                    if os.path.exists(result_path):
                        results = []
                        with open(result_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                parts = line.strip().split('\t')
                                if len(parts) >= 3:
                                    results.append({
                                        'claim': parts[0],
                                        'evidence': parts[1],
                                        'prediction': parts[2],
                                        'confidence': float(parts[3]) if len(parts) > 3 else 0.0,
                                        'model': model,
                                        'source': source
                                    })
                        
                        key = f"{model}_{source}"
                        prediction_results[key] = results
                        print(f"📊 Loaded {len(results)} predictions from {model} + {source}")
                
                except Exception as e:
                    print(f"⚠️ Could not load {model} + {source} results: {e}")
        
        return prediction_results
    
    def analyze_source_confidence(self, prediction_results):
        """Analyze confidence levels across different sources"""
        confidence_by_source = {}
        accuracy_by_source = {}
        
        for key, results in prediction_results.items():
            model, source = key.split('_')
            
            if source not in confidence_by_source:
                confidence_by_source[source] = []
                accuracy_by_source[source] = []
            
            confidences = [r['confidence'] for r in results if r['confidence'] > 0]
            confidence_by_source[source].extend(confidences)
        
        # Calculate statistics
        source_stats = {}
        for source, confidences in confidence_by_source.items():
            if confidences:
                source_stats[source] = {
                    'mean_confidence': np.mean(confidences),
                    'std_confidence': np.std(confidences),
                    'median_confidence': np.median(confidences),
                    'count': len(confidences)
                }
        
        return source_stats
    
    def analyze_model_confidence(self, prediction_results):
        """Analyze confidence levels across different models"""
        confidence_by_model = {}
        
        for key, results in prediction_results.items():
            model, source = key.split('_')
            
            if model not in confidence_by_model:
                confidence_by_model[model] = []
            
            confidences = [r['confidence'] for r in results if r['confidence'] > 0]
            confidence_by_model[model].extend(confidences)
        
        # Calculate statistics
        model_stats = {}
        for model, confidences in confidence_by_model.items():
            if confidences:
                model_stats[model] = {
                    'mean_confidence': np.mean(confidences),
                    'std_confidence': np.std(confidences),
                    'median_confidence': np.median(confidences),
                    'count': len(confidences)
                }
        
        return model_stats
    
    def generate_confidence_visualizations(self, prediction_results, output_dir=None):
        """Generate confidence analysis visualizations"""
        if not output_dir:
            output_dir = os.path.join(config.OUTPUT_DIR, 'Disagreement_Analysis', 'Figures')
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for visualization
        viz_data = []
        for key, results in prediction_results.items():
            model, source = key.split('_')
            for result in results:
                if result['confidence'] > 0:
                    viz_data.append({
                        'Model': model,
                        'Source': source,
                        'Confidence': result['confidence'],
                        'Prediction': result['prediction']
                    })
        
        if not viz_data:
            print("⚠️ No confidence data available for visualization")
            return
        
        df_viz = pd.DataFrame(viz_data)
        
        # 1. Confidence distribution by source
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_viz, x='Source', y='Confidence')
        plt.title('Confidence Distribution by Evidence Source')
        plt.ylabel('Confidence Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_by_source.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence distribution by model
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df_viz, x='Model', y='Confidence')
        plt.title('Confidence Distribution by Model')
        plt.ylabel('Confidence Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_by_model.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confidence heatmap
        confidence_matrix = df_viz.groupby(['Model', 'Source'])['Confidence'].mean().unstack()
        plt.figure(figsize=(10, 8))
        sns.heatmap(confidence_matrix, annot=True, fmt='.3f', cmap='viridis')
        plt.title('Average Confidence by Model and Source')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confidence visualizations saved to: {output_dir}")
    
    def generate_confidence_report(self, dataset_name='scifact', output_file=None):
        """Generate comprehensive confidence analysis report"""
        
        # Load prediction results
        prediction_results = self.load_prediction_results(dataset_name)
        
        if not prediction_results:
            print("❌ No prediction results found for confidence analysis")
            return
        
        # Analyze confidence by source and model
        source_stats = self.analyze_source_confidence(prediction_results)
        model_stats = self.analyze_model_confidence(prediction_results)
        
        # Generate visualizations
        self.generate_confidence_visualizations(prediction_results)
        
        # Create text report
        report_lines = [
            f"# Source Confidence Analysis Report",
            f"Dataset: {dataset_name.upper()}",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"- Analyzed {len(prediction_results)} model-source combinations",
            f"- Covered {len(source_stats)} evidence sources",
            f"- Evaluated {len(model_stats)} different models",
            "",
            "## Confidence by Evidence Source",
        ]
        
        for source, stats in source_stats.items():
            report_lines.extend([
                f"### {source}",
                f"- Mean Confidence: {stats['mean_confidence']:.4f}",
                f"- Std Deviation: {stats['std_confidence']:.4f}",
                f"- Median Confidence: {stats['median_confidence']:.4f}",
                f"- Total Predictions: {stats['count']}",
                ""
            ])
        
        report_lines.append("## Confidence by Model")
        for model, stats in model_stats.items():
            report_lines.extend([
                f"### {model}",
                f"- Mean Confidence: {stats['mean_confidence']:.4f}",
                f"- Std Deviation: {stats['std_confidence']:.4f}",
                f"- Median Confidence: {stats['median_confidence']:.4f}",
                f"- Total Predictions: {stats['count']}",
                ""
            ])
        
        # Save report
        if not output_file:
            output_file = f'{dataset_name}_source_confidence_report.md'
        
        output_path = os.path.join(config.OUTPUT_DIR, 'Disagreement_Analysis', 'Reports', output_file)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📝 Confidence analysis report saved to: {output_path}")
        
        return source_stats, model_stats

def main():
    """Main function for source confidence analysis"""
    print("🔍 Starting Source Confidence Analysis...")
    
    # Initialize analyzer
    analyzer = SourceConfidenceAnalyzer()
    
    # Run analysis for SciFact dataset
    source_stats, model_stats = analyzer.generate_confidence_report('scifact')
    
    if source_stats and model_stats:
        print("\n📊 Source Confidence Analysis Summary:")
        print(f"Evidence Sources Analyzed: {list(source_stats.keys())}")
        print(f"Models Analyzed: {list(model_stats.keys())}")
        
        # Print top confidence sources
        best_source = max(source_stats.items(), key=lambda x: x[1]['mean_confidence'])
        best_model = max(model_stats.items(), key=lambda x: x[1]['mean_confidence'])
        
        print(f"Highest Confidence Source: {best_source[0]} ({best_source[1]['mean_confidence']:.4f})")
        print(f"Highest Confidence Model: {best_model[0]} ({best_model[1]['mean_confidence']:.4f})")
    
    print("✅ Source confidence analysis complete!")

if __name__ == "__main__":
    main() 