# -*- coding: utf-8 -*-
"""Gradient Descent for Logistic Regression

Logistic regression implemented from scratch in NumPy and trained with batch and
mini-batch gradient descent on the South African Heart Disease dataset, then
compared against scikit-learn.

Author: Feroz Naeem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
url = "https://raw.githubusercontent.com/tofighi/MachineLearning/master/datasets/heart.csv"
data = pd.read_csv(url)

# ------------------------------------------
# Step 1: Exploratory Data Analysis (EDA)
# ------------------------------------------
# Class distribution
class_dist = data['chd'].value_counts(normalize=True) * 100
print("Class Distribution (%):\n", class_dist)

# Missing values
missing_values = data.isnull().sum()
print("\nMissing Values:\n", missing_values)

# Categorical variables
categorical_vars = data.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical Variables:", categorical_vars)

# Correlation matrix (including chd)
numerical_data = data.select_dtypes(include=[np.number])  # Keep chd
corr_matrix = numerical_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f",
            xticklabels=False, yticklabels=False)  # Remove labels
plt.title("Correlation Matrix (Including CHD)")
plt.show()

max_corr = corr_matrix.abs().unstack().sort_values(ascending=False)
max_corr_pairs = max_corr[max_corr < 1.0].head(1)
print("\nMaximum Absolute Correlation Pair:\n", max_corr_pairs)

# ------------------------------------------
# Step 2: Preprocessing
# ------------------------------------------
# One-Hot Encoding
data = pd.get_dummies(data, columns=['famhist'], drop_first=True)

# Standardization
numerical_features = numerical_data.columns.drop('chd')  # Exclude target
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Add intercept term
data.insert(0, 'intercept', 1.0)

# Prepare arrays
X = data.drop('chd', axis=1).values.astype(np.float64)
y = data['chd'].values.astype(np.float64)

# ------------------------------------------
# Step 3: Model Implementation
# ------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5
    return -(1/m) * (y @ np.log(h + epsilon) + (1 - y) @ np.log(1 - h + epsilon))

def batch_gradient_descent(X, y, theta, alpha, epochs):
    m = len(y)
    cost_history = []
    for _ in range(epochs):
        h = sigmoid(X @ theta)
        gradient = (1/m) * X.T @ (h - y)
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

def mini_batch_gradient_descent(X, y, theta, alpha, epochs, batch_size=50):
    m = len(y)
    cost_history = []
    for _ in range(epochs):
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            h = sigmoid(X_batch @ theta)
            gradient = (1/len(X_batch)) * X_batch.T @ (h - y_batch)
            theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# ------------------------------------------
# Step 4: Training (12 graphs)
# ------------------------------------------
np.random.seed(42)
theta_initial = np.random.rand(X.shape[1]).astype(np.float64)

alphas = [0.001, 0.0001]
epochs_list = [500, 1000, 10000]

# Batch GD plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Batch GD Learning Curves", fontsize=16)
plot_idx = 0
for alpha in alphas:
    for epochs in epochs_list:
        row, col = divmod(plot_idx, 3)
        theta = theta_initial.copy()
        _, cost_history = batch_gradient_descent(X, y, theta, alpha, epochs)
        axes[row, col].plot(cost_history, color='blue' if alpha==0.001 else 'red')
        axes[row, col].set_title(f"α={alpha}, epochs={epochs}")
        plot_idx += 1
plt.tight_layout()
plt.show()

# Mini-Batch GD plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Mini-Batch GD Learning Curves", fontsize=16)
plot_idx = 0
for alpha in alphas:
    for epochs in epochs_list:
        row, col = divmod(plot_idx, 3)
        theta = theta_initial.copy()
        _, cost_history = mini_batch_gradient_descent(X, y, theta, alpha, epochs)
        axes[row, col].plot(cost_history, color='green' if alpha==0.001 else 'purple')
        axes[row, col].set_title(f"α={alpha}, epochs={epochs}")
        plot_idx += 1
plt.tight_layout()
plt.show()

# ------------------------------------------
# Step 5: Comparison with sklearn
# ------------------------------------------
model = LogisticRegression(penalty=None, max_iter=10000)
model.fit(X, y)

theta_final, _ = batch_gradient_descent(X, y, theta_initial.copy(), alpha=0.001, epochs=1000)

print("\nCustom Coefficients:")
print(theta_final)
print("\nSklearn Coefficients:")
print(np.concatenate([model.intercept_, model.coef_[0]]))

"""#### **1. EDA Questions**  
**Q1: What is the percentage of each class (0 and 1)?**  

```
Class Distribution (%):
0    65.367965
1    34.632035
```  

---

**Q2: How many missing values do we have?**  

```
Missing Values: 0 for all columns.
```  

---

**Q3: How many categorical variables are in the features?**  

```
Categorical Variables: ['famhist']
```  

---

**Q4: What features have the maximum correlation?**  

```
Maximum Correlation Pair: adiposity  obesity    0.716556
```  
---

### **Q5 (Batch Gradient Descent): Based on the learning curves, what value for epoch does seem reasonable to stop training for each learning rate (cost will not decrease significantly anymore)?**  
- Learning Rate (alpha = 0.001): Stops at 1,000 epochs.  
  - **Reason**: The cost stabilizes (plateaus) after 1,000 epochs, suggesting minimal decline. If trained too long, larger learning rates converge faster but risk overshooting.  
- Learning Rate (alpha = 0.0001): Stop at 10,000 epochs.  
  - **Reason: The smaller learning rate requires significantly more epochs to converge. The cost gradually drops up to 10,000 epochs. After  improvements are negligible.   

---

### **Q6 (Mini-Batch Gradient Descent): Based on the learning curves, what value for epoch does seem reasonable to stop training for each learning rate (cost will not decrease significantly anymore)?**  
- Learning Rate (alpha = 0.001): Stop at 1,000 epochs.  
  - Reason: The cost stabilizes after about 1,000 epochs, even with the noise from mini-batches (batch size = 50). Minimal progress is seen with additional training.    
- Learning Rate (alpha = 0.0001): Stop at 10,000 epochs.  
  - Reason: Extended training is necessary due to the noisy updates from mini-batches and the lower learning rate. 10,000 epochs later, the cost reaches a plateau.    

---


### Compare custom coefficients with sklearn’s LogisticRegression**  
**Custom Coefficients**:  
```python  
[ 0.187, 0.790, 0.620, 0.575, 0.166, 0.043, 0.118, 0.724, 0.496, 0.655, -0.027 ]  
```  

**Sklearn Coefficients**:  
```python  
[-0.635, -0.635, -0.118, 0.157, 0.358, 0.352, 0.162, 0.369, -0.278, 0.010, 0.645, 0.939]  
```  


"""