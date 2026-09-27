"""
Week 2 Project: End-to-End Preprocessing Pipeline

Demonstrates:
- Loading data with Pandas
- Separating features and target
- Train/test split
- Standardization without data leakage
- Stratified 5-fold cross-validation
- Logistic Regression classification
- Final evaluation on unseen test data

Approved AI/ML stack:
NumPy, Pandas, Scikit-Learn
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. Load the Iris dataset
iris = load_iris()

X = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

y = pd.Series(
    iris.target,
    name="target"
)


# 2. Basic data validation
print("Dataset shape:", X.shape)
print("Missing values:", X.isnull().sum().sum())

# Convert Pandas objects to NumPy arrays for Scikit-Learn
X = X.to_numpy()
y = y.to_numpy()


# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 4. Build preprocessing + model pipeline
# The scaler is fitted only on training data,
# preventing data leakage.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])


# 5. Five-fold stratified cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)


# 6. Train final pipeline
pipeline.fit(X_train, y_train)


# 7. Evaluate on unseen test data
y_pred = pipeline.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_pred
)


# 8. Display results
print("\nWeek 2 Project: End-to-End Preprocessing Pipeline")
print("-" * 55)

print(f"Total samples: {len(X)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {np.mean(cv_scores):.4f}")
print(f"CV accuracy standard deviation: {np.std(cv_scores):.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

print("\nPipeline completed successfully.")
print("Preprocessing was performed without data leakage.")
