
"""W4D5 capstone: compare Logistic Regression and Random Forest."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    RocCurveDisplay,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

OUTPUT_DIR = Path(__file__).parent
RANDOM_STATE = 42


def load_sentiment_data():
    """Load a local CSV if available; otherwise use sklearn's local dataset."""
    csv_path = OUTPUT_DIR / "sentiment.csv"

    if csv_path.exists():
        data = pd.read_csv(csv_path)
        if not {"text", "sentiment"}.issubset(data.columns):
            raise ValueError(
                "sentiment.csv must contain 'text' and 'sentiment' columns."
            )
        data = data[["text", "sentiment"]].dropna()
        X = data["text"].astype(str)
        y = data["sentiment"].astype(str).str.lower().map(
            {"negative": 0, "positive": 1, "neg": 0, "pos": 1, "0": 0, "1": 1}
        )
        if y.isna().any():
            raise ValueError(
                "Sentiment labels must be positive/negative or 1/0."
            )
        return X, y.astype(int)

    # Download-free fallback: a small, illustrative dataset.
    # Replace this with a larger real review dataset for stronger results.
    examples = [
        ("This movie was wonderful and moving", 1),
        ("I loved the excellent acting", 1),
        ("A fantastic and enjoyable story", 1),
        ("Brilliant film with great characters", 1),
        ("The movie was fun and entertaining", 1),
        ("An amazing experience from start to finish", 1),
        ("The plot was engaging and beautiful", 1),
        ("Great performances and a satisfying ending", 1),
        ("I really enjoyed this delightful film", 1),
        ("A superb movie worth watching", 1),
        ("This movie was boring and disappointing", 0),
        ("I hated the terrible acting", 0),
        ("A dreadful story and a waste of time", 0),
        ("The film was dull and poorly made", 0),
        ("Awful characters and a predictable plot", 0),
        ("The movie was frustrating and unpleasant", 0),
        ("A terrible experience from beginning to end", 0),
        ("Weak performances and a boring ending", 0),
        ("I did not enjoy this horrible film", 0),
        ("A disappointing movie that I regret watching", 0),
    ]
    frame = pd.DataFrame(examples, columns=["text", "sentiment"])
    return frame["text"], frame["sentiment"]


def main():
    X, y = load_sentiment_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            )),
        ]),
    }

    results = []
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        auc = roc_auc_score(y_test, probabilities)

        results.append({
            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "roc_auc": auc,
        })

        print(f"\n=== {name} ===")
        print(classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["negative", "positive"],
            zero_division=0,
        ))

        ConfusionMatrixDisplay.from_predictions(
            y_test,
            predictions,
            labels=[0, 1],
            display_labels=["Negative", "Positive"],
            cmap="Blues",
        )
        plt.title(f"{name} — Confusion Matrix")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(OUTPUT_DIR / f"{safe_name}_confusion_matrix.png")
        plt.close()

        RocCurveDisplay.from_predictions(
            y_test, probabilities, name=f"{name} (AUC={auc:.3f})", ax=ax
        )

    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_title("Sentiment Classifier — ROC Curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png")
    plt.close(fig)

    results_frame = pd.DataFrame(results)
    results_frame.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    (OUTPUT_DIR / "classification_report.txt").write_text(
        results_frame.to_string(index=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== Model Comparison ===")
    print(results_frame.to_string(index=False))
    print("\nSaved metrics and plots to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
