# Week 2 Project: End-to-End Preprocessing Pipeline

## Objective

Implement an end-to-end machine learning preprocessing and evaluation
pipeline using the approved AI/ML stack:

- NumPy
- Pandas
- Scikit-Learn

## Workflow

1. Load the Iris dataset using Scikit-Learn.
2. Store features and target using Pandas.
3. Check dataset shape and missing values.
4. Convert the data to NumPy arrays for Scikit-Learn processing.
5. Split the data into 80% training and 20% testing sets.
6. Standardize features using StandardScaler.
7. Combine preprocessing and Logistic Regression using a Scikit-Learn Pipeline.
8. Perform 5-fold Stratified Cross-Validation.
9. Train the final pipeline using the complete training data.
10. Evaluate the pipeline on the unseen test set.

## Data Leakage Prevention

StandardScaler is placed inside the Scikit-Learn Pipeline.

This ensures that scaling parameters are learned only from the appropriate
training data during cross-validation and final training. The test set
remains completely unseen until final evaluation.

## Results

- Total samples: 150
- Training samples: 120
- Testing samples: 30
- Missing values: 0
- Mean Cross-Validation Accuracy: 0.9583
- CV Accuracy Standard Deviation: 0.0264
- Final Test Accuracy: 0.9333

## Output Evidence

```text
Dataset shape: (150, 4)
Missing values: 0

Total samples: 150
Training samples: 120
Testing samples: 30
Mean CV accuracy: 0.9583
CV accuracy standard deviation: 0.0264
Test accuracy: 0.9333

Pipeline completed successfully.
Preprocessing was performed without data leakage.
