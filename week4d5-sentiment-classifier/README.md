# W4D5 — Sentiment Classifier

## Objective
Compare Logistic Regression and Random Forest for binary sentiment
classification using TF-IDF features and labelled IMDb movie reviews.

## Dataset
- Source: Stanford IMDb Large Movie Review Dataset
- Local subset: 5,000 training reviews
- Positive reviews: 2,500
- Negative reviews: 2,500
- Train/test split: 70% / 30%, stratified
- The local CSV is excluded from Git.

Dataset source:
https://ai.stanford.edu/~amaas/data/sentiment/

## Models
1. TF-IDF + Logistic Regression
2. TF-IDF + Random Forest Classifier

Both models use the same train/test split.

## Results

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 0.8513 | 0.8347 |
| Precision | 0.8518 | 0.8457 |
| Recall | 0.8507 | 0.8187 |
| ROC-AUC | 0.9353 | 0.9091 |

## Conclusion
Logistic Regression performed better on this test split across all four
reported metrics. These results are specific to the selected 5,000-review
subset and the current train/test split.

## Generated outputs
- model_comparison.csv
- classification_report.txt
- logistic_regression_confusion_matrix.png
- random_forest_confusion_matrix.png
- roc_curve.png

## Run
Activate the project's Python virtual environment, install the required
dependencies, and run:

python sentiment_classifier.py
