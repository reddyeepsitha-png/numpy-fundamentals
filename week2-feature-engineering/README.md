# W2D4: Train/Test Split & Cross-Validation

## Objective

Implement train/test splitting and cross-validation using the approved AI/ML stack:
NumPy, Pandas, and Scikit-Learn.

## Implementation

- Loaded the Iris dataset using Scikit-Learn.
- Split the dataset into 80% training and 20% testing data.
- Used StandardScaler through a Pipeline to prevent data leakage.
- Applied 5-fold Stratified Cross-Validation.
- Trained Logistic Regression on the complete training set.
- Evaluated the final model on unseen test data.

## Results

- Total samples: 150
- Training samples: 120
- Testing samples: 30
- Mean Cross-Validation Accuracy: 0.9583
- Test Accuracy: 0.9333

## Key Learning

The test set is kept completely separate until final evaluation.
Using a Pipeline ensures that scaling is fitted only on the appropriate
training portion during cross-validation, preventing data leakage.
