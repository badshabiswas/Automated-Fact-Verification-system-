# Disagreement Analysis

This module analyzes disagreements between different models and provides uncertainty quantification for fact-checking predictions.

## Analysis Components

### 📊 **Model Disagreement**
- **Inter-model Variance**: Compare predictions across different LLMs
- **Confidence Correlation**: Analyze relationship between confidence and accuracy
- **Error Pattern Analysis**: Identify systematic biases and failure modes

### 📈 **Confidence Calibration**
- **Probability Extraction**: Extract confidence scores from model outputs
- **Calibration Analysis**: Measure how well confidence reflects accuracy
- **Uncertainty Quantification**: Provide reliability estimates

### 🎯 **Performance Metrics**
- **Accuracy Analysis**: Overall and per-class performance
- **Agreement Matrices**: Model-to-model agreement rates
- **Confidence Distribution**: Statistical analysis of uncertainty

## Visualization

### 📋 **Confidence Plots**
- **KDE Plots**: Kernel Density Estimation of confidence distributions
- **Calibration Curves**: Reliability diagrams for probability calibration
- **Grid Visualizations**: Multi-model comparison matrices

### 📊 **Agreement Analysis**
- **Confusion Matrices**: Inter-model prediction agreements
- **Error Distribution**: Types and frequencies of disagreements
- **Confidence vs Accuracy**: Scatter plots and correlation analysis

## Output Files

```
Disagreement_Analysis/
├── Results/
│   ├── Model_Comparison/      # Cross-model analysis
│   ├── Confidence_Analysis/   # Uncertainty quantification
│   └── Error_Analysis/        # Failure mode analysis
├── Figures/
│   ├── confidence_kde_grid.png
│   ├── calibration_curves.png
│   └── agreement_matrices.png
└── Reports/
    ├── statistical_summary.txt
    └── disagreement_report.pdf
```

## Key Insights

This analysis helps identify:
- **Reliable Predictions**: High-confidence cases with model agreement
- **Uncertain Cases**: Low-confidence or high-disagreement instances
- **Model Strengths**: Each model's comparative advantages
- **Dataset Challenges**: Inherently difficult claims requiring human review

## Research Applications

- **Model Selection**: Choose optimal models for specific domains
- **Ensemble Methods**: Combine models based on confidence and agreement
- **Human-in-the-Loop**: Flag uncertain cases for manual review
- **Bias Detection**: Identify systematic errors and model limitations 