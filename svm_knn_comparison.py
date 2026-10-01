
"""
W3D4: SVM & KNN — When to Use What
Compare Support Vector Machine and K-Nearest Neighbors
using the approved NumPy, Pandas, and Scikit-learn stack.
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

# 1. Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dataset shape:", X.shape)
print("\nClass distribution:")
print(y.value_counts())

# 2. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 3. Define models with feature scaling
models = {
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale"),
    ),
    "KNN": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    ),
}

# 4. Train and evaluate each model
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
    })

    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))

# 5. Display the comparison
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df.round(4).to_string(index=False))

# 6. Tune selected hyperparameters
print("\nHyperparameter comparison:")

for k in [3, 5, 7, 9]:
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=k),
    )
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    print(f"KNN (k={k}) Accuracy: {accuracy_score(y_test, predictions):.4f}")

for c_value in [0.1, 1, 10]:
    svm = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=c_value, gamma="scale"),
    )
    svm.fit(X_train, y_train)
    predictions = svm.predict(X_test)
    print(f"SVM (C={c_value}) Accuracy: {accuracy_score(y_test, predictions):.4f}")
