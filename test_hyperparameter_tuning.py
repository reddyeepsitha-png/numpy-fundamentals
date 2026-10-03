
"""Basic tests for the W3D5 hyperparameter tuning workflow."""

import numpy as np

from hyperparameter_tuning import load_data, build_models


def test_dataset():
    X, y = load_data()

    assert X.shape[0] == len(y)
    assert X.shape[1] > 0
    assert not X.isnull().values.any()
    assert set(np.unique(y)) == {0, 1}


def test_model_pipelines():
    svm, knn = build_models()

    assert svm.named_steps["scaler"] is not None
    assert knn.named_steps["scaler"] is not None
    assert svm.named_steps["model"].__class__.__name__ == "SVC"
    assert (
        knn.named_steps["model"].__class__.__name__
        == "KNeighborsClassifier"
    )
