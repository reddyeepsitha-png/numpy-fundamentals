
"""
W3D5: Hyperparameter Tuning using GridSearchCV and RandomizedSearchCV.

Dataset: Breast Cancer Wisconsin
Stack: NumPy, Pandas, Scikit-learn
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    """Load the dataset and return features and target."""
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data
    y = dataset.target
    return X, y


def build_models():
    """Create scaled SVM and KNN pipelines."""
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC()),
    ])

    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier()),
    ])

    return svm_pipeline, knn_pipeline


def tune_models(X_train, y_train):
    """Tune SVM and KNN using grid and randomized search."""
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    svm_pipeline, knn_pipeline = build_models()

    svm_grid = {
        "model__C": [0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"],
    }

    knn_grid = {
        "model__n_neighbors": [3, 5, 7, 9, 11],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    }

    searches = {
        "SVM GridSearch": GridSearchCV(
            svm_pipeline,
            svm_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        ),
        "KNN GridSearch": GridSearchCV(
            knn_pipeline,
            knn_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        ),
        "SVM RandomSearch": RandomizedSearchCV(
            svm_pipeline,
            svm_grid,
            n_iter=10,
            cv=cv,
            scoring="accuracy",
            random_state=42,
            n_jobs=-1,
        ),
        "KNN RandomSearch": RandomizedSearchCV(
            knn_pipeline,
            knn_grid,
            n_iter=10,
            cv=cv,
            scoring="accuracy",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    for name, search in searches.items():
        print(f"\nRunning {name}...")
        search.fit(X_train, y_train)

        results[name] = search
        print("Best parameters:", search.best_params_)
        print(f"Best CV accuracy: {search.best_score_:.4f}")

    return results


def evaluate_models(results, X_test, y_test):
    """Evaluate tuned models on held-out test data."""
    rows = []

    for name, search in results.items():
        predictions = search.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        rows.append({
            "Search": name,
            "Best CV Accuracy": search.best_score_,
            "Test Accuracy": accuracy,
            "Best Parameters": str(search.best_params_),
        })

        print(f"\n{name}")
        print(f"Test accuracy: {accuracy:.4f}")
        print(classification_report(y_test, predictions))

    results_df = pd.DataFrame(rows)
    results_df.to_csv("tuning_results.csv", index=False)

    print("\n=== Hyperparameter Tuning Comparison ===")
    print(results_df.to_string(index=False))
    print("\nResults saved to tuning_results.csv")

    return results_df


def main():
    """Run the complete hyperparameter tuning workflow."""
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Dataset shape:", X.shape)
    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    tuned_models = tune_models(X_train, y_train)
    evaluate_models(tuned_models, X_test, y_test)


if __name__ == "__main__":
    main()
