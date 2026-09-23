# W2D3 - Handling Imbalanced Data Using SMOTE

## Objective

Handle class imbalance using Synthetic Minority Over-sampling Technique (SMOTE).

## Workflow

1. Create an imbalanced dataset.
2. Separate features and target.
3. Split the data into training and testing sets.
4. Apply SMOTE only to the training data.
5. Verify the class distribution after resampling.
6. Keep the test data unchanged.

## Results

Original dataset:

- Class 0: 18 samples
- Class 1: 2 samples

Training data before SMOTE:

- Class 0: 14 samples
- Class 1: 2 samples

Training data after SMOTE:

- Class 0: 14 samples
- Class 1: 14 samples

The training classes were successfully balanced.

## Why SMOTE is applied only to training data

SMOTE is applied only to the training data to prevent data leakage. The test data remains unchanged so that model evaluation is performed on unseen, original data.

## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-Learn
- imbalanced-learn
