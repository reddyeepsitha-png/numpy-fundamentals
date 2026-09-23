# Handling Imbalanced Data using SMOTE
# Week 2 - AI/ML Internship

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


# ============================================================
# 1. CREATE IMBALANCED SAMPLE DATASET
# ============================================================

data = {
    "Age": [
        21, 22, 24, 25, 27, 29, 31, 33, 35, 37,
        39, 41, 43, 45, 47, 49, 51, 53, 55, 57
    ],
    "Income": [
        25000, 27000, 29000, 31000, 33000, 35000, 37000, 39000,
        41000, 43000, 45000, 47000, 49000, 51000, 53000, 55000,
        57000, 59000, 61000, 63000
    ],
    "Experience": [
        0, 1, 2, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16, 17, 18
    ],
    "Purchased": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1
    ]
}

df = pd.DataFrame(data)

print("=" * 60)
print("ORIGINAL DATASET")
print("=" * 60)
print(df)


# ============================================================
# 2. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION BEFORE SMOTE")
print("=" * 60)

class_distribution_before = df["Purchased"].value_counts().sort_index()

print(class_distribution_before)


# ============================================================
# 3. SEPARATE FEATURES AND TARGET
# ============================================================

X = df[
    [
        "Age",
        "Income",
        "Experience"
    ]
]

y = df["Purchased"]


# ============================================================
# 4. SPLIT DATA BEFORE APPLYING SMOTE
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. APPLY SMOTE ONLY TO TRAINING DATA
# ============================================================

smote = SMOTE(
    random_state=42,
    k_neighbors=1
)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)


# ============================================================
# 6. CHECK CLASS DISTRIBUTION AFTER SMOTE
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION AFTER SMOTE")
print("=" * 60)

class_distribution_after = (
    pd.Series(y_train_resampled)
    .value_counts()
    .sort_index()
)

print(class_distribution_after)


# ============================================================
# 7. DISPLAY RESAMPLED DATASET
# ============================================================

resampled_df = pd.DataFrame(
    X_train_resampled,
    columns=X.columns
)

resampled_df["Purchased"] = y_train_resampled

print("\n" + "=" * 60)
print("RESAMPLED TRAINING DATA")
print("=" * 60)

print(resampled_df)


# ============================================================
# 8. VERIFY SMOTE RESULT
# ============================================================

print("\n" + "=" * 60)
print("SMOTE VERIFICATION")
print("=" * 60)

if class_distribution_after.iloc[0] == class_distribution_after.iloc[1]:
    print("SUCCESS: Training classes are balanced after SMOTE.")
else:
    print("WARNING: Training classes are still imbalanced.")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SMOTE TASK COMPLETED")
print("=" * 60)

print("""
SMOTE was applied only to the training data.

Before SMOTE:
- Minority class had fewer samples.

After SMOTE:
- Synthetic minority samples were generated.
- Training classes are balanced.

The original test data was kept unchanged.
""")