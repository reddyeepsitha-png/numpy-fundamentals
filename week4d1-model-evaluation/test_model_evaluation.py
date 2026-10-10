"""Tests for the W4D1 model evaluation pipeline."""

import numpy as np
import pandas as pd

from model_evaluation import (
    RANDOM_STATE,
    calculate_metrics,
    create_model,
    load_data,
    perform_stratified_cv,
)


def test_load_data():
    """Verify that the breast cancer dataset loads correctly."""
    X, y = load_data()

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 30
    assert len(y.unique()) == 2


def test_model_creation():
    """Verify that the evaluation model is configured correctly."""
    model = create_model()

    assert model.random_state == RANDOM_STATE
    assert model.n_estimators == 200


def test_calculate_metrics():
    """Verify that all evaluation metrics are calculated correctly."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_proba = np.array([0.1, 0.6, 0.8, 0.9])

    metrics = calculate_metrics(y_true, y_pred, y_proba)

    expected_metrics = {
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    }

    assert set(metrics.keys()) == expected_metrics

    for score in metrics.values():
        assert 0.0 <= score <= 1.0


def test_stratified_cross_validation():
    """Verify that five-fold CV returns all required metrics."""
    X, y = load_data()

    # Use a smaller sample for a fast unit test.
    X_sample = X.iloc[:200]
    y_sample = y.iloc[:200]

    summary = perform_stratified_cv(X_sample, y_sample)

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 5
    assert set(summary["Metric"]) == {
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    }
    assert (summary["Mean"] >= 0).all()
    assert (summary["Mean"] <= 1).all()
    assert (summary["Std"] >= 0).all()