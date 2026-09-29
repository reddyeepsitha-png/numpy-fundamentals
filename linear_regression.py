
# W3D1: Linear Regression — Scikit-Learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load real-world California housing data
diabetes = load_diabetes(as_frame=True)
X = diabetes.data
y = diabetes.target

print("Dataset: Diabetes")
print("Target: Disease progression measure")
print(X.head())

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Define the models
models = {
    "Linear Regression": make_pipeline(
        StandardScaler(), LinearRegression()
    ),
    "Ridge Regression": make_pipeline(
        StandardScaler(), Ridge(alpha=1.0)
    ),
    "Lasso Regression": make_pipeline(
        StandardScaler(), Lasso(alpha=0.01, max_iter=10000)
    ),
}

results = []
predictions = {}

# 4. Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    })

    print(f"\n{name}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R2:   {r2:.4f}")

# 5. Compare model performance
results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
results_df.to_csv("regression_results.csv", index=False)

# 6. Print Linear Regression coefficients
linear_model = models["Linear Regression"].named_steps["linearregression"]

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_,
})

print("\nLINEAR REGRESSION COEFFICIENTS")
print(coefficients.to_string(index=False))
print("Intercept:", round(linear_model.intercept_, 4))

# 7. Actual versus predicted plot
y_pred_linear = predictions["Linear Regression"]

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_linear, alpha=0.3)
plt.xlabel("Predicted Disease Progression")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Linear Regression: Residual Plot")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.close()

# 8. Residual plot
residuals = y_test - y_pred_linear

plt.figure(figsize=(8, 6))
plt.scatter(y_pred_linear, residuals, alpha=0.3)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted House Value")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Linear Regression: Residual Plot")
plt.tight_layout()
plt.savefig("residuals.png", dpi=150)
plt.close()

print("\nSUCCESS: Training, evaluation, comparison, and plots completed.")
