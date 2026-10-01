
# AI/ML 1M — Week 3

## W3D4: SVM & KNN — When to Use What

### Objective
Compare Support Vector Machine (SVM) and K-Nearest Neighbors (KNN) classification models using Scikit-learn.

### Dataset
Breast Cancer Wisconsin dataset from Scikit-learn.
- Samples: 569
- Features: 30

### Implementation
- Loaded and inspected data using Pandas and NumPy.
- Split the dataset into training and testing sets.
- Standardized features using StandardScaler.
- Trained an SVM classifier with an RBF kernel.
- Trained a KNN classifier with k=5.
- Evaluated accuracy, precision, recall, and F1-score.
- Compared different KNN neighbor counts and SVM C values.

### Results
| Model | Accuracy |
|---|---:|
| SVM (C=1) | 98.25% |
| KNN (k=5) | 95.61% |
| KNN (k=3) | 98.25% |

### Key Concepts
- SVM finds a decision boundary between classes.
- KNN classifies samples based on nearby training examples.
- Feature scaling is important for both algorithms.
- Hyperparameters affect model performance.

### Technologies
Python, NumPy, Pandas, Scikit-learn
