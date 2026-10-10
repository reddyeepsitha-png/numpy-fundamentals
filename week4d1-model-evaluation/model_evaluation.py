"""
W4D1: Model Evaluation Metrics

Evaluates a Random Forest classification model using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Stratified K-Fold Cross-Validation
- ROC Curve
- Precision-Recall Curve
- Learning Curve

Approved stack:
NumPy, Pandas, Scikit-Learn, Matplotlib
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
)


RANDOM_STATE = 42
N_SPLITS = 5


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the Breast Cancer dataset."""

    data = load_breast_cancer()

    features = pd.DataFrame(
        data.data,
        columns=data.feature_names,
    )
    target = pd.Series(
        data.target,
        name="target",
    )

    return features, target


def create_model():
    return RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE
    )
    """Create a reproducible Random Forest classifier."""

    return RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
    )


def calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> Dict[str, float]:
    """Calculate classification evaluation metrics."""

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_probability),
    }


def evaluate_test_set(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate the trained model on the independent test set."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = calculate_metrics(
        y_test,
        predictions,
        probabilities,
    )

    print("\nTest Set Evaluation")
    print("=" * 40)

    for metric, score in metrics.items():
        print(f"{metric:<12}: {score:.4f}")

    print("\nClassification Report")
    print("=" * 40)
    print(classification_report(y_test, predictions))

    return metrics, probabilities


def perform_stratified_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> pd.DataFrame:
    """Evaluate multiple metrics using 5-fold Stratified CV."""

    model = create_model()

    stratified_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=stratified_cv,
        scoring=scoring,
        return_train_score=True,
    )

    summary = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
                "ROC-AUC",
            ],
            "Mean": [
                np.mean(cv_results["test_accuracy"]),
                np.mean(cv_results["test_precision"]),
                np.mean(cv_results["test_recall"]),
                np.mean(cv_results["test_f1"]),
                np.mean(cv_results["test_roc_auc"]),
            ],
            "Std": [
                np.std(cv_results["test_accuracy"]),
                np.std(cv_results["test_precision"]),
                np.std(cv_results["test_recall"]),
                np.std(cv_results["test_f1"]),
                np.std(cv_results["test_roc_auc"]),
            ],
        }
    )

    print("\n5-Fold Stratified Cross-Validation")
    print("=" * 40)
    print(summary.to_string(index=False))

    return summary


def plot_roc_curve(
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> None:
    """Generate and save the ROC curve."""

    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_test,
        probabilities,
    )

    roc_auc = auc(
        false_positive_rate,
        true_positive_rate,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"Random Forest (AUC = {roc_auc:.3f})",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "roc_curve.png",
        dpi=300,
    )
    plt.close()

    print("\nROC curve saved as roc_curve.png")


def plot_precision_recall_curve(
    y_test: pd.Series,
    probabilities: np.ndarray,
) -> None:
    """Generate and save the Precision-Recall curve."""

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label="Random Forest",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "precision_recall_curve.png",
        dpi=300,
    )
    plt.close()

    print("Precision-Recall curve saved as precision_recall_curve.png")


def plot_learning_curve(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Generate a learning curve with standard-deviation bands."""

    model = create_model()

    stratified_cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        X_train,
        y_train,
        cv=stratified_cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=-1,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    validation_mean = np.mean(
        validation_scores,
        axis=1,
    )
    validation_std = np.std(
        validation_scores,
        axis=1,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training Score",
    )

    plt.plot(
        train_sizes,
        validation_mean,
        marker="o",
        label="Validation Score",
    )

    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
    )

    plt.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.2,
    )

    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "learning_curve.png",
        dpi=300,
    )
    plt.close()

    print("Learning curve saved as learning_curve.png")


def main() -> None:
    """Run the complete model evaluation pipeline."""

    X, y = load_data()

    # Verify that the target is not present among the features.
    assert "target" not in X.columns

    # Keep the test set completely independent from cross-validation.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = create_model()
    model.fit(X_train, y_train)

    test_metrics, probabilities = evaluate_test_set(
        model,
        X_test,
        y_test,
    )

    cv_summary = perform_stratified_cv(
        X_train,
        y_train,
    )

    pd.DataFrame(
        test_metrics.items(),
        columns=["Metric", "Score"],
    ).to_csv(
        "test_metrics.csv",
        index=False,
    )

    cv_summary.to_csv(
        "cv_metrics.csv",
        index=False,
    )

    plot_roc_curve(
        y_test,
        probabilities,
    )

    plot_precision_recall_curve(
        y_test,
        probabilities,
    )

    plot_learning_curve(
        X_train,
        y_train,
    )

    print("\nEvaluation files saved successfully.")


if __name__ == "__main__":
    main()