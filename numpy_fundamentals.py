import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


# ============================================================
# 1. ARRAY CREATION AND SHAPES
# ============================================================

array_1d = np.array([1, 2, 3, 4, 5])

array_2d = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

array_3d = np.array([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])

print("1D Array:")
print(array_1d)
print("Shape:", array_1d.shape)

print("\n2D Array:")
print(array_2d)
print("Shape:", array_2d.shape)

print("\n3D Array:")
print(array_3d)
print("Shape:", array_3d.shape)


# ============================================================
# 2. BROADCASTING
# ============================================================

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

vector = np.array([10, 20, 30])

broadcast_result = matrix + vector

print("\nBroadcasting:")
print(broadcast_result)


# ============================================================
# 3. VECTORIZED OPERATIONS
# ============================================================

values = np.array([1, 2, 3, 4, 5])

squared = values ** 2
doubled = values * 2
normalized = values / np.max(values)

print("\nVectorized Operations:")
print("Original:", values)
print("Squared:", squared)
print("Doubled:", doubled)
print("Normalized:", normalized)


# ============================================================
# 4. MATRIX MULTIPLICATION
# ============================================================

A = np.array([
    [1, 2],
    [3, 4]
])

B = np.array([
    [5, 6],
    [7, 8]
])

matrix_product = A @ B

print("\nMatrix Multiplication:")
print(matrix_product)


# ============================================================
# 5. LOAD REAL CSV DATASET
# ============================================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df.to_csv("iris_dataset.csv", index=False)

print("\nCSV Dataset:")
print(df.head())


# ============================================================
# 6. NUMPY STATISTICS
# ============================================================

sepal_length = df["sepal length (cm)"].to_numpy()
sepal_width = df["sepal width (cm)"].to_numpy()
petal_length = df["petal length (cm)"].to_numpy()
petal_width = df["petal width (cm)"].to_numpy()

mean_sepal_length = np.mean(sepal_length)
std_sepal_length = np.std(sepal_length)

correlation_matrix = np.corrcoef(
    np.array([
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ])
)

print("\nStatistics:")
print("Mean Sepal Length:", mean_sepal_length)
print("Standard Deviation:", std_sepal_length)

print("\nCorrelation Matrix:")
print(correlation_matrix)


# ============================================================
# 7. CORRELATION ANALYSIS
# ============================================================

correlation = np.corrcoef(
    sepal_length,
    petal_length
)[0, 1]

print("\nCorrelation:")
print("Sepal Length vs Petal Length:", correlation)


print("\n==============================")
print("NUMPY FUNDAMENTALS COMPLETED")
print("==============================")

print("1D shape:", array_1d.shape)
print("2D shape:", array_2d.shape)
print("3D shape:", array_3d.shape)
print("Mean:", mean_sepal_length)
print("Std:", std_sepal_length)
print("Correlation:", correlation)