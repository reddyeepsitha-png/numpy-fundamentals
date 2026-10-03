
"""
W3D3: Decision Trees and Random Forests
Objective:
- Understand decision tree classification.
- Compare unrestricted and tuned decision trees.
- Demonstrate overfitting control.
- Evaluate a Random Forest classifier.
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def main():
    # Load the built-in Breast Cancer Wisconsin dataset.
    dataset = load_breast_cancer()

    # Store features and target in Pandas data structures.
    X = pd.DataFrame(
        dataset.data,
        columns=dataset.feature_names,
    )
    y = pd.Series(dataset.target, name="target")

    print("Dataset shape:", X.shape)
    print("\nClass distribution:")
    print(y.value_counts())

    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Model 1: Decision Tree without a depth limit.
    unrestricted_tree = DecisionTreeClassifier(
        random_state=42
    )
    unrestricted_tree.fit(X_train, y_train)

    # Model 2: Tuned Decision Tree to control complexity.
    tuned_tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        random_state=42,
    )
    tuned_tree.fit(X_train, y_train)

    # Model 3: Random Forest with 100 decision trees.
    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
    )
    forest.fit(X_train, y_train)

    # Compare training and testing accuracy.
    models = {
        "Unrestricted Decision Tree": unrestricted_tree,
        "Tuned Decision Tree": tuned_tree,
        "Random Forest": forest,
    }

    results = []

    for name, model in models.items():
        train_predictions = model.predict(X_train)
        test_predictions = model.predict(X_test)

        train_accuracy = accuracy_score(
            y_train, train_predictions
        )
        test_accuracy = accuracy_score(
            y_test, test_predictions
        )

        results.append({
            "Model": name,
            "Train Accuracy": round(train_accuracy, 4),
            "Test Accuracy": round(test_accuracy, 4),
            "Tree Depth": (
                model.get_depth()
                if isinstance(model, DecisionTreeClassifier)
                else "Multiple"
            ),
        })

    results_df = pd.DataFrame(results)

    print("\n--- MODEL COMPARISON ---")
    print(results_df.to_string(index=False))

    # Evaluate the tuned decision tree.
    tuned_predictions = tuned_tree.predict(X_test)

    print("\n--- CONFUSION MATRIX ---")
    print(confusion_matrix(y_test, tuned_predictions))

    print("\n--- CLASSIFICATION REPORT ---")
    print(
        classification_report(
            y_test,
            tuned_predictions,
            target_names=dataset.target_names,
        )
    )

    # Display readable decision rules.
    print("\n--- DECISION TREE RULES ---")
    print(
        export_text(
            tuned_tree,
            feature_names=list(X.columns),
            max_depth=3,
        )
    )

    # Identify the most important Random Forest features.
    importance = pd.Series(
        forest.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    print("\n--- TOP 10 IMPORTANT FEATURES ---")
    print(importance.head(10).to_string())

    # Basic validation checks.
    assert len(tuned_predictions) == len(y_test)
    assert np.isfinite(
        results_df["Train Accuracy"]
    ).all()
    assert np.isfinite(
        results_df["Test Accuracy"]
    ).all()
    assert results_df["Test Accuracy"].between(0, 1).all()
    assert np.isclose(forest.feature_importances_.sum(), 1.0)

    print("\nAll sanity checks passed.")


if __name__ == "__main__":
    main()
