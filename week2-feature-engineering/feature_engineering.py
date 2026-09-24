"""
W2D4: Train/Test Split & Cross-Validation

Demonstrates:
- Train/test split
- Feature scaling without data leakage
- 5-fold cross-validation
- Final test-set evaluation
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target


# Split the dataset into training and testing sets
# 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Create a pipeline to prevent data leakage.
# StandardScaler is fitted separately inside each CV training fold.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])


# Perform 5-fold stratified cross-validation on training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)


# Train the final pipeline on the complete training data
pipeline.fit(X_train, y_train)


# Evaluate the final model on unseen test data
y_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)


# Display results
print("W2D4: Train/Test Split & Cross-Validation")
print("-" * 50)
print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
