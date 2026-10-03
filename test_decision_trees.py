"""
Tests for W3D3 Decision Tree and Random Forest implementation.
"""

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def test_decision_tree():
    """Test that the Decision Tree trains and predicts correctly."""

    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        random_state=42,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert model.get_depth() <= 4
    assert np.isfinite(model.score(X_test, y_test))


def test_random_forest():
    """Test that the Random Forest trains and produces valid output."""

    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        test_size=0.2,
        random_state=42,
        stratify=data.target,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert len(predictions) == len(y_test)
    assert model.feature_importances_.shape[0] == X_train.shape[1]
    assert np.isclose(model.feature_importances_.sum(), 1.0)


if __name__ == "__main__":
    test_decision_tree()
    test_random_forest()
    print("All W3D3 tests passed.")