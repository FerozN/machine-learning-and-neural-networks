# Machine Learning Algorithms & Neural Networks

Machine learning algorithms implemented in Python. Most are built from scratch in NumPy and then compared against scikit-learn or TensorFlow/Keras. The projects cover supervised learning (logistic regression), unsupervised learning (K-Means, nearest-neighbour clustering, DBSCAN), and neural networks (Keras regression networks and hand-written backpropagation).

I built these for a machine learning course in the Computer Engineering program at Toronto Metropolitan University. I originally wrote them as Google Colab notebooks and exported them to `.py` scripts.

## Contents

| File | Topic | What it does |
|------|-------|--------------|
| `logistic regression gradient descent.py` | Logistic regression | Exploratory data analysis (EDA) and preprocessing. Logistic regression written in NumPy and trained with batch and mini-batch gradient descent. Compares the coefficients with scikit-learn. |
| `clustering.py` | Clustering | K-Means, threshold-based nearest-neighbour (NN) clustering, and DBSCAN on a small 2-D dataset and on the heart disease data. Checks the results against scikit-learn. |
| `neural network regression and xor.py` | Neural networks | Keras feedforward networks that fit a polynomial in 10 experiments. Also a 2-2-1 network that learns XOR, with backpropagation written in NumPy. |

## Dataset

The logistic regression and clustering scripts use the **South African Heart Disease** dataset: 462 patients, with a binary `chd` (coronary heart disease) label. The scripts load it at runtime from:

```
https://raw.githubusercontent.com/tofighi/MachineLearning/master/datasets/heart.csv
```

You need an internet connection to run them. The neural network script generates its own synthetic data, and the clustering script also hard-codes a small 8-point dataset.

## Project details

### Gradient descent for logistic regression

1. **EDA.** Class balance (about 65% / 35%), a check for missing values, identification of categorical variables (`famhist`), and a correlation heatmap. The strongest correlation is between `adiposity` and `obesity` (≈ 0.72).
2. **Preprocessing.** One-hot encodes `famhist`, standardizes the numeric features, and adds an intercept column.
3. **Model.** `sigmoid`, a cross-entropy cost function, `batch_gradient_descent`, and `mini_batch_gradient_descent` (batch size 50), written in NumPy.
4. **Experiments.** Learning curves for learning rates of 0.001 and 0.0001 at 500, 1,000, and 10,000 epochs, for both optimizers (12 plots).
5. **Comparison.** Compares the learned coefficients with an unregularized scikit-learn `LogisticRegression`.

### Clustering

- **Part A, K-Means.** Runs K-Means by hand on 8 points (A1–A8) with fixed starting centres, iterating until it converges. Then applies K-Means (k = 2) to `sbp` and `tobacco`: it trains on the first 300 rows, maps each cluster to a class by majority vote, and reports test accuracy with plots of the actual and predicted classes.
- **Part B, nearest-neighbour clustering.** Builds a graph that connects points within a distance threshold and treats each connected component as a cluster. Compares the result with K-Means, then predicts test points from their nearest training point.
- **Part C, DBSCAN.** Runs DBSCAN on the 8-point set (eps = 2 and eps = √10) and on the heart data, with noise points handled explicitly.
- **Part D, verification.** Compares the results with scikit-learn's `KMeans`, `DBSCAN`, and `AgglomerativeClustering`, which acts as a stand-in for NN clustering.
- Ends with a short write-up on clustering evaluation metrics: silhouette score, Davies-Bouldin, Calinski-Harabasz, ARI (Adjusted Rand Index), NMI (Normalized Mutual Information), and Fowlkes-Mallows.

### Neural networks

- Generates 30,000 samples of `y = 0.2x⁴ + 2x³ + 0.1x² + 10` on [-1, 1], with helper functions to shuffle, split the data 30/20/50 (train/validation/test), min-max scale it, and compute MAE, MSE, RMSE, and R².
- Trains Keras models (Adam optimizer, MSE loss, 20 epochs, batch size 12) under five configurations, each run on both shuffled and unshuffled data:

  | Case | Architecture | Activation | Scaled |
  |------|--------------|------------|--------|
  | 1 | 12-8-4-1 | ReLU | No |
  | 2 | 24-1 | ReLU | No |
  | 3 | 12-8-4-1 | tanh | No |
  | 4 | 12-8-4-1 | ReLU | Yes |
  | 5 | 12-8-4-1 | tanh | Yes |

  The unshuffled runs show what goes wrong when you split ordered data: the model trains on one region of x and is tested on another.
- **XOR from scratch.** A 2-2-1 sigmoid network trained for 10,000 epochs. The forward pass, backpropagation, and gradient descent updates are written by hand.

## How to run

**Requirements:** Python 3.9–3.12. TensorFlow does not support newer versions yet.

```bash
git clone https://github.com/FerozN/<repo-name>.git
cd <repo-name>

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

Run any script. The file names contain spaces, so keep the quotes:

```bash
python "logistic regression gradient descent.py"
python clustering.py
python "neural network regression and xor.py"
```

Notes:
- Results print to the console, and each plot opens in its own window. **Close a plot window to let the script continue.** To run without windows, set the matplotlib backend first: `MPLBACKEND=Agg` on macOS/Linux, or `$env:MPLBACKEND="Agg"` in PowerShell.
- The neural network script trains 10 Keras models, which takes a few minutes on a CPU. The 10,000-epoch gradient descent runs also take a moment.
- **Google Colab:** Upload a script (or paste it into cells) and run it. Colab already has all the dependencies installed.

## Known limitations

- **Colab exports.** Each file is a flattened notebook, so it has repeated imports and text cells kept as string literals.
- **Scaling in the neural network script.** Min-max scaling is fitted separately on the train, validation, and test splits. It is not fitted on the training split and then reused for the others. As a result, the scaled metrics describe each split on its own scale.
- **Test predictions in clustering.** For NN clustering and DBSCAN, test points take the cluster of their nearest training point. These algorithms have no native `predict` method, so this is an approximation.

## Tech stack

Python · NumPy · pandas · scikit-learn · TensorFlow/Keras · Matplotlib · seaborn

## Author

**Feroz Naeem**, B.Eng. Computer Engineering (Software Engineering), Toronto Metropolitan University
[github.com/FerozN](https://github.com/FerozN) · [linkedin.com/in/FerozN](https://linkedin.com/in/FerozN)
