# -*- coding: utf-8 -*-
"""Clustering: K-Means, Nearest-Neighbour Clustering, and DBSCAN

K-Means, threshold-based nearest-neighbour clustering, and DBSCAN, applied to a
small 8-point dataset and to the South African Heart Disease dataset, then
verified against scikit-learn.

Author: Feroz Naeem
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

# =============================
# PART A: DATASET 1
# =============================

# --- DATASET 1: the eight points ---
points_dict = {
    "A1": (2,10),
    "A2": (2, 5),
    "A3": (8, 4),
    "A4": (5, 8),
    "A5": (7, 5),
    "A6": (6, 4),
    "A7": (1, 2),
    "A8": (4, 9)
}

labels = list(points_dict.keys())  # ["A1","A2",...,"A8"]
X = np.array([points_dict[k] for k in labels])
N = len(labels)

# Initial centers: C1=A2, C2=A3, C3=A8
C = np.array([
    points_dict["A2"],
    points_dict["A3"],
    points_dict["A8"],
], dtype=float)

def euclidean_dist(a, b):
    return np.sqrt(np.sum((a - b)**2))

def assign_clusters(X, centers):
    cluster_ids = []
    for i in range(len(X)):
        dists = [euclidean_dist(X[i], c) for c in centers]
        cluster_ids.append(np.argmin(dists))  # index of nearest center
    return np.array(cluster_ids)

def compute_new_centers(X, cluster_ids, k):
    new_centers = []
    for cluster_label in range(k):
        pts = X[cluster_ids == cluster_label]
        if len(pts) > 0:
            new_centers.append(pts.mean(axis=0))
        else:
            # if no points in a cluster, keep old center
            new_centers.append(C[cluster_label])
    return np.array(new_centers)

# 1) Assign each point to the closest initial center
cluster_ids_1 = assign_clusters(X, C)
# 2) Compute new centers
C_new = compute_new_centers(X, cluster_ids_1, k=3)

print("Q1) After first epoch, cluster memberships:")
for cluster_label in range(3):
    idxs = np.where(cluster_ids_1 == cluster_label)[0]
    clust_points = [labels[i] for i in idxs]
    print(f"  Cluster {cluster_label+1}: {clust_points}")

print("\nQ2) The new cluster centers (after first epoch):")
print(f"   Cluster 1 center = {C_new[0]}")
print(f"   Cluster 2 center = {C_new[1]}")
print(f"   Cluster 3 center = {C_new[2]}")

# Q3) Plot the results after the first epoch
colors = ["red","green","blue"]
plt.figure(figsize=(6,6))
for cluster_label in range(3):
    idxs = np.where(cluster_ids_1 == cluster_label)[0]
    plt.scatter(X[idxs,0], X[idxs,1], c=colors[cluster_label],
                label=f"Cluster {cluster_label+1}")

# Plot centers with a distinct marker
plt.scatter(C_new[:,0], C_new[:,1], marker="*", s=200,
            c=colors, edgecolors="black", linewidths=1)
for i, txt in enumerate(labels):
    plt.annotate(txt, (X[i,0]+0.2, X[i,1]+0.2))

plt.xlim(0,10)
plt.ylim(0,10)
plt.title("After First Epoch of K-Means (Dataset 1)")
plt.grid(True)
plt.legend()
plt.show()

# Q4) Another iteration
cluster_ids_2 = assign_clusters(X, C_new)
C_final = compute_new_centers(X, cluster_ids_2, k=3)

memberships_changed = np.any(cluster_ids_2 != cluster_ids_1)
if not memberships_changed:
    print("No points changed clusters on the second iteration, so we have converged.")
    print("Hence, only 1 additional iteration was needed (total 2 iterations).")
else:
    print("Points changed clusters! (Would keep iterating.)")

# Q5) Final membership & centers after convergence
print("\nFinal cluster memberships (same as after first epoch if converged):")
for cluster_label in range(3):
    idxs = np.where(cluster_ids_2 == cluster_label)[0]
    clust_points = [labels[i] for i in idxs]
    print(f"  Cluster {cluster_label+1}: {clust_points}")

print("\nFinal cluster centers:")
print(f"  Cluster 1 center = {C_final[0]}")
print(f"  Cluster 2 center = {C_final[1]}")
print(f"  Cluster 3 center = {C_final[2]}")

# Optional final plot
plt.figure(figsize=(6,6))
for cluster_label in range(3):
    idxs = np.where(cluster_ids_2 == cluster_label)[0]
    plt.scatter(X[idxs,0], X[idxs,1], c=colors[cluster_label],
                label=f"Cluster {cluster_label+1}")

plt.scatter(C_final[:,0], C_final[:,1], marker="*", s=200,
            c=colors, edgecolors="black", linewidths=1)
for i, txt in enumerate(labels):
    plt.annotate(txt, (X[i,0]+0.2, X[i,1]+0.2))

plt.xlim(0,10)
plt.ylim(0,10)
plt.title("Final Clusters After Convergence (Dataset 1)")
plt.grid(True)
plt.legend()
plt.show()


# =============================
# PART B: DATASET 2
# =============================

# Instead of using a big CSV string, we'll reference the GitHub link:
heart_url = "https://raw.githubusercontent.com/tofighi/MachineLearning/master/datasets/heart.csv"

df = pd.read_csv(heart_url)
print("Loaded heart dataset. Shape:", df.shape)
print(df.head())

# Convert sbp & tobacco to numeric in case of any unexpected data
df['sbp'] = pd.to_numeric(df['sbp'], errors='coerce')
df['tobacco'] = pd.to_numeric(df['tobacco'], errors='coerce')

# Drop rows where sbp or tobacco is missing (NaN)
df.dropna(subset=['sbp','tobacco'], inplace=True)

print("Data shape AFTER dropping NaN in sbp/tobacco:", df.shape)

# 2) SPLIT INTO TRAIN/TEST (first 300 -> train, rest -> test)
train_df = df.iloc[:300].copy()
test_df  = df.iloc[300:].copy()

X_train = train_df[['sbp','tobacco']].values
y_train = train_df['chd'].values

X_test  = test_df[['sbp','tobacco']].values
y_test  = test_df['chd'].values

# 3) K-Means (k=2)
kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto')
kmeans.fit(X_train)

cluster_labels_train = kmeans.labels_

# 4) Map cluster IDs -> {0,1} based on majority membership
def majority_label_map(cluster_ids, true_labels):
    mapping = {}
    for c in [0,1]:
        chd_vals = true_labels[cluster_ids == c]
        if len(chd_vals) == 0:
            mapping[c] = 0
        else:
            ones_count = np.sum(chd_vals)
            zeros_count = len(chd_vals) - ones_count
            mapping[c] = 1 if ones_count >= zeros_count else 0
    return mapping

cluster_to_chd = majority_label_map(cluster_labels_train, y_train)

def map_cluster_ids_to_chd(cluster_ids, cluster_map):
    return np.array([cluster_map[c] for c in cluster_ids], dtype=int)

y_train_pred = map_cluster_ids_to_chd(cluster_labels_train, cluster_to_chd)
train_acc = accuracy_score(y_train, y_train_pred)
print("Q6) Training Accuracy =", train_acc)

# 5) Predict on Test
cluster_labels_test = kmeans.predict(X_test)
y_test_pred = map_cluster_ids_to_chd(cluster_labels_test, cluster_to_chd)
test_acc = accuracy_score(y_test, y_test_pred)
print("Q6) Test Accuracy =", test_acc)

# 6) Q7: Plot 'sbp' vs 'tobacco' (ACTUAL classes) - TEST
plt.figure(figsize=(7,5))
plt.scatter(X_test[y_test==0,0], X_test[y_test==0,1],
            c='green', marker='o', label='CHD=0 (actual)')
plt.scatter(X_test[y_test==1,0], X_test[y_test==1,1],
            c='red', marker='^', label='CHD=1 (actual)')

plt.title("Q7) 'sbp' vs 'tobacco' (ACTUAL classes) - TEST DATA")
plt.xlabel("SBP")
plt.ylabel("Tobacco")
plt.legend()
plt.grid(True)
plt.show()

# 7) Q8: Plot 'sbp' vs 'tobacco' (PREDICTED clusters) - TEST
plt.figure(figsize=(7,5))
plt.scatter(X_test[y_test_pred==0,0], X_test[y_test_pred==0,1],
            c='blue', marker='o', label='Predicted=0')
plt.scatter(X_test[y_test_pred==1,0], X_test[y_test_pred==1,1],
            c='orange', marker='^', label='Predicted=1', s=100)

plt.title("Q8) 'sbp' vs 'tobacco' (PREDICTED classes) - TEST DATA")
plt.xlabel("SBP")
plt.ylabel("Tobacco")
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import accuracy_score

# ============================================================
# Part B - Dataset 1 (Nearest Neighbor Clustering)
# ============================================================
print("=== PART B: Dataset 1 (Questions 9–11) ===")

# The eight points (Dataset 1)
points_dict = {
    "A1": (2,10),
    "A2": (2, 5),
    "A3": (8, 4),
    "A4": (5, 8),
    "A5": (7, 5),
    "A6": (6, 4),
    "A7": (1, 2),
    "A8": (4, 9)
}
labels_1 = list(points_dict.keys())  # e.g. ["A1","A2",...,"A8"]
X1 = np.array([points_dict[k] for k in labels_1])

def nn_clustering_threshold(X, threshold):
    """
    Build a graph where edges connect points if dist <= threshold.
    Then each connected component is one cluster.
    Returns:
      cluster_ids: array of cluster labels for each point (1-based)
      cluster_count: number of distinct clusters
    """
    n = len(X)
    visited = [False]*n
    cluster_ids = np.full(n, -1, dtype=int)
    cluster_count = 0

    def dist(a,b):
        return np.sqrt(np.sum((a-b)**2))

    for i in range(n):
        if not visited[i]:
            cluster_count += 1
            queue = [i]
            visited[i] = True
            cluster_ids[i] = cluster_count
            while queue:
                curr = queue.pop(0)
                for j in range(n):
                    if not visited[j]:
                        if dist(X[curr], X[j]) <= threshold:
                            visited[j] = True
                            cluster_ids[j] = cluster_count
                            queue.append(j)
    return cluster_ids, cluster_count

# ========== Q9 & Q10 ==========
threshold_1 = 4.0
cluster_ids_1, num_clusters_1 = nn_clustering_threshold(X1, threshold_1)

# Q9) "Have you got the same results as using K-Means?"
print(f"\nQ9) NN Clustering with threshold={threshold_1} => found {num_clusters_1} clusters.")
for c in range(1, num_clusters_1+1):
    pts = [labels_1[i] for i in range(len(X1)) if cluster_ids_1[i] == c]
    print(f"  Cluster {c} => {pts}")

print("\n   (Recall that K-Means from Part A gave 3 clusters: ")
print("    - Cluster 1: A2,A7")
print("    - Cluster 2: A3,A5,A6")
print("    - Cluster 3: A1,A4,A8 )")
print("   Compare these to the NN clusters above.\n")

# Q10) "Give the final plot for NN clustering"
colors = ["red","green","blue","orange","purple","brown"]
plt.figure(figsize=(6,6))
for c in range(1, num_clusters_1+1):
    idxs = np.where(cluster_ids_1 == c)[0]
    color = colors[(c-1) % len(colors)]
    plt.scatter(X1[idxs,0], X1[idxs,1], c=color, label=f"Cluster {c}")
for i, txt in enumerate(labels_1):
    plt.annotate(txt, (X1[i,0]+0.2, X1[i,1]+0.2))
plt.xlim(0,10)
plt.ylim(0,10)
plt.title(f"Q10) NN Clustering (Dataset 1) - Threshold={threshold_1}")
plt.grid(True)
plt.legend()
plt.show()

# Q11) "Compare performance of K-Means vs. NN. Provide disadvantages on bigger datasets."
print("\nQ11) Comparison of K-Means vs. NN Clustering:")
print(" - K-Means:")
print("    * Typically faster for large datasets (especially if k << n).")
print("    * Requires specifying k upfront.")
print("    * Assumes roughly spherical clusters.")
print(" - NN Clustering (threshold-based):")
print("    * Finds arbitrarily shaped clusters but can be O(n^2).")
print("    * No need to pick k, but must pick threshold.")
print("    * On large datasets, can be slow and produce chaining effects if threshold is big.")


# ============================================================
# Part B - Dataset 2 (NN Clustering) - Q12
# ============================================================
print("\n=== PART B: Dataset 2 (Question 12) ===")
print("We'll cluster the first 300 rows (train) using 'sbp' & 'tobacco' only, then test on the remainder.")

# 1) Load the heart data from GitHub
heart_url = "https://raw.githubusercontent.com/tofighi/MachineLearning/master/datasets/heart.csv"
df = pd.read_csv(heart_url)

# Convert sbp & tobacco to numeric, drop rows with missing
df['sbp'] = pd.to_numeric(df['sbp'], errors='coerce')
df['tobacco'] = pd.to_numeric(df['tobacco'], errors='coerce')
df.dropna(subset=['sbp','tobacco'], inplace=True)

# 2) Split into train/test
train_df = df.iloc[:300].copy()
test_df  = df.iloc[300:].copy()

X_train = train_df[['sbp','tobacco']].values
y_train = train_df['chd'].values

X_test  = test_df[['sbp','tobacco']].values
y_test  = test_df['chd'].values

threshold_2 = 4.0
train_cluster_ids, train_cluster_count = nn_clustering_threshold(X_train, threshold_2)
print(f"\nWe found {train_cluster_count} clusters in the training set (NN threshold={threshold_2}).")

# Map each cluster to 0 or 1 by majority
cluster_to_label = {}
for c in range(1, train_cluster_count+1):
    y_in_cluster = y_train[train_cluster_ids == c]
    ones = np.sum(y_in_cluster)
    zeros = len(y_in_cluster) - ones
    maj_label = 1 if ones >= zeros else 0
    cluster_to_label[c] = maj_label

# Predict on training set (to see training accuracy)
y_train_pred = np.array([cluster_to_label[c] for c in train_cluster_ids])
train_acc = accuracy_score(y_train, y_train_pred)
print("NN Train Accuracy =", train_acc)

# For each test point, find nearest training point. If dist <= threshold => same cluster
from math import sqrt
def euclid_dist(a, b):
    return sqrt(np.sum((a - b)**2))

y_test_pred = []
for xt in X_test:
    dists = [euclid_dist(xt, x) for x in X_train]
    nearest_idx = np.argmin(dists)
    min_dist = dists[nearest_idx]
    if min_dist <= threshold_2:
        c_id = train_cluster_ids[nearest_idx]
        pred_label = cluster_to_label[c_id]
    else:
        # If no training point is within threshold, default to 0
        pred_label = 0
    y_test_pred.append(pred_label)

y_test_pred = np.array(y_test_pred)

# Q12) Provide accuracy
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Q12) Test Accuracy (NN Clustering, threshold={threshold_2}) = {test_acc}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import accuracy_score
from math import sqrt
import matplotlib  # For colormap access

# ============================================================
# Part C - Dataset 1 (DBSCAN)
# ============================================================
print("=== PART C: Dataset 1 (Questions 13–14) ===")

# The eight points (Dataset 1)
points_dict = {
    "A1": (2,10),
    "A2": (2, 5),
    "A3": (8, 4),
    "A4": (5, 8),
    "A5": (7, 5),
    "A6": (6, 4),
    "A7": (1, 2),
    "A8": (4, 9)
}
labels_1 = list(points_dict.keys())
X1 = np.array([points_dict[k] for k in labels_1])

def plot_dbscan_clusters(X, cluster_labels, title):
    """
    Plot points color-coded by DBSCAN cluster_labels.
    cluster_labels can be -1 for noise.
    """
    unique_labels = np.unique(cluster_labels)
    # Use plt.get_cmap to create a discrete colormap (avoids deprecation issues)
    cmap = plt.get_cmap("tab10", len(unique_labels))
    plt.figure(figsize=(6,6))
    for ul in unique_labels:
        idxs = (cluster_labels == ul)
        if ul == -1:
            # Noise plotted as black 'x'
            plt.scatter(X[idxs,0], X[idxs,1], c='k', marker='x', label='Noise')
        else:
            plt.scatter(X[idxs,0], X[idxs,1], color=cmap(ul), label=f"Cluster {ul}")
    for i, txt in enumerate(labels_1):
        plt.annotate(txt, (X[i,0]+0.2, X[i,1]+0.2))
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

# -------------------- Q13 --------------------
# DBSCAN with eps=2, min_samples=2
db1 = DBSCAN(eps=2.0, min_samples=2, metric='euclidean')
db1.fit(X1)
cluster_labels_1 = db1.labels_  # -1 indicates noise
print("\nQ13) DBSCAN on Dataset 1 with eps=2, min_samples=2 => cluster labels:")
print("Labels array:", cluster_labels_1)
plot_dbscan_clusters(X1, cluster_labels_1, title="Q13) DBSCAN (Dataset 1) eps=2, min_samples=2")

# -------------------- Q14 --------------------
# DBSCAN with eps = sqrt(10), min_samples=2
eps_val = sqrt(10)  # approximately 3.1623
db2 = DBSCAN(eps=eps_val, min_samples=2, metric='euclidean')
db2.fit(X1)
cluster_labels_2 = db2.labels_
print(f"\nQ14) DBSCAN on Dataset 1 with eps=sqrt(10)~{eps_val:.3f}, min_samples=2 => cluster labels:")
print("Labels array:", cluster_labels_2)
plot_dbscan_clusters(X1, cluster_labels_2, title=f"Q14) DBSCAN (Dataset 1) eps=sqrt(10)~{eps_val:.2f}, min_samples=2")

# ============================================================
# Part C - Dataset 2 (DBSCAN)
# ============================================================
print("\n=== PART C: Dataset 2 (Question 15) ===")
print("Using DBSCAN clustering on the heart dataset (using 'sbp' and 'tobacco').")

# Load heart data from GitHub
heart_url = "https://raw.githubusercontent.com/tofighi/MachineLearning/master/datasets/heart.csv"
df = pd.read_csv(heart_url)

# Convert sbp & tobacco to numeric; drop rows with missing values
df['sbp'] = pd.to_numeric(df['sbp'], errors='coerce')
df['tobacco'] = pd.to_numeric(df['tobacco'], errors='coerce')
df.dropna(subset=['sbp','tobacco'], inplace=True)

# Split into training (first 300 rows) and testing (remaining rows)
train_df = df.iloc[:300].copy()
test_df  = df.iloc[300:].copy()

X_train = train_df[['sbp','tobacco']].values
y_train = train_df['chd'].values
X_test  = test_df[['sbp','tobacco']].values
y_test  = test_df['chd'].values

# DBSCAN hyperparameters (assumed; adjust as needed)
eps_2 = 4.0
min_samples_2 = 5
db_train = DBSCAN(eps=eps_2, min_samples=min_samples_2, metric='euclidean')
db_train.fit(X_train)
train_labels = db_train.labels_  # -1 denotes noise

# Map each DBSCAN cluster (ignoring noise) to a majority label (0 or 1)
unique_clusters = np.unique(train_labels)
cluster_to_chd = {}
for c in unique_clusters:
    if c == -1:
        continue
    y_in_cluster = y_train[train_labels == c]
    ones_count = np.sum(y_in_cluster)
    zeros_count = len(y_in_cluster) - ones_count
    cluster_label = 1 if ones_count >= zeros_count else 0
    cluster_to_chd[c] = cluster_label

# Evaluate training accuracy: assign predicted label per cluster (noise defaults to 0)
y_train_pred = []
for i in range(len(X_train)):
    c = train_labels[i]
    if c == -1:
        y_train_pred.append(0)
    else:
        y_train_pred.append(cluster_to_chd[c])
y_train_pred = np.array(y_train_pred)
train_acc = accuracy_score(y_train, y_train_pred)

# For each test point, assign label based on nearest training point's cluster
def euclid_dist(a, b):
    return sqrt(np.sum((a - b)**2))

y_test_pred = []
for xt in X_test:
    dists = [euclid_dist(xt, x) for x in X_train]
    nearest_idx = np.argmin(dists)
    c = train_labels[nearest_idx]
    if c == -1:
        pred_label = 0
    else:
        pred_label = cluster_to_chd[c]
    y_test_pred.append(pred_label)
y_test_pred = np.array(y_test_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nQ15) DBSCAN (Dataset 2) with eps={eps_2}, min_samples={min_samples_2}")
print("Train Accuracy =", train_acc)
print("Test Accuracy  =", test_acc)

print("\nComparison of clustering algorithms:")
print(" - K-Means: Requires specifying k (e.g., k=2); fast but assumes spherical clusters.")
print(" - NN Clustering: Threshold-based; flexible but can be slow (O(n^2)) and sensitive to threshold.")
print(" - DBSCAN: Detects arbitrarily shaped clusters and noise; depends on eps and min_samples.")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.metrics import accuracy_score

# ---------------------------
# Dataset 1 Verification
# ---------------------------
print("=== PART D: Verification using sklearn (Q16) ===\n")
print("Dataset 1 Verification:")

# Use the same eight points from Dataset 1
points_dict = {
    "A1": (2,10),
    "A2": (2, 5),
    "A3": (8, 4),
    "A4": (5, 8),
    "A5": (7, 5),
    "A6": (6, 4),
    "A7": (1, 2),
    "A8": (4, 9)
}
labels_1 = list(points_dict.keys())
X1 = np.array([points_dict[k] for k in labels_1])

# --- Sklearn K-Means ---
from sklearn.cluster import KMeans
sk_kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
sk_kmeans.fit(X1)
print("Sklearn K-Means labels:", sk_kmeans.labels_)
print("Sklearn K-Means centers:\n", sk_kmeans.cluster_centers_)

# --- Sklearn Nearest Neighbor Clustering surrogate ---
# We use AgglomerativeClustering with connectivity from radius_neighbors_graph.
from sklearn.neighbors import radius_neighbors_graph
from sklearn.cluster import AgglomerativeClustering

# Create connectivity graph using a radius of 4 (as in our custom NN clustering)
connectivity = radius_neighbors_graph(X1, radius=4.0, mode='connectivity', include_self=True)
agg_cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=4.0,
                                        linkage='single', connectivity=connectivity)
sk_nn_labels = agg_cluster.fit_predict(X1)
print("Sklearn AgglomerativeClustering (NN surrogate) labels:", sk_nn_labels)

# --- Sklearn DBSCAN ---
from sklearn.cluster import DBSCAN
sk_dbscan = DBSCAN(eps=2.0, min_samples=2, metric='euclidean')
sk_dbscan.fit(X1)
print("Sklearn DBSCAN (eps=2, min_samples=2) labels:", sk_dbscan.labels_)

# ---------------------------
# Dataset 2 Verification
# ---------------------------
print("\nDataset 2 Verification:")

# Load Dataset 2 from GitHub
heart_url = "https://raw.githubusercontent.com/tofighi/MachineLearning/master/datasets/heart.csv"
df = pd.read_csv(heart_url)
df['sbp'] = pd.to_numeric(df['sbp'], errors='coerce')
df['tobacco'] = pd.to_numeric(df['tobacco'], errors='coerce')
df.dropna(subset=['sbp','tobacco'], inplace=True)

# Split: first 300 rows for training, remainder for testing.
train_df = df.iloc[:300].copy()
test_df  = df.iloc[300:].copy()

X_train = train_df[['sbp','tobacco']].values
y_train = train_df['chd'].values
X_test  = test_df[['sbp','tobacco']].values
y_test  = test_df['chd'].values

# --- Sklearn K-Means on Dataset 2 ---
sk_kmeans2 = KMeans(n_clusters=2, random_state=42, n_init="auto")
sk_kmeans2.fit(X_train)
sk_cluster_labels_train = sk_kmeans2.labels_

# Map clusters to chd (0 or 1) by majority vote.
def majority_label_map(cluster_ids, true_labels):
    mapping = {}
    for c in [0,1]:
        chd_vals = true_labels[cluster_ids == c]
        if len(chd_vals) == 0:
            mapping[c] = 0
        else:
            mapping[c] = 1 if np.sum(chd_vals) >= (len(chd_vals)/2) else 0
    return mapping

mapping_kmeans = majority_label_map(sk_cluster_labels_train, y_train)
sk_y_train_pred = np.array([mapping_kmeans[c] for c in sk_cluster_labels_train])
sk_train_acc = accuracy_score(y_train, sk_y_train_pred)
sk_y_test_pred = np.array([mapping_kmeans[c] for c in sk_kmeans2.predict(X_test)])
sk_test_acc = accuracy_score(y_test, sk_y_test_pred)
print("Sklearn K-Means (Dataset 2) Train Accuracy =", sk_train_acc)
print("Sklearn K-Means (Dataset 2) Test Accuracy =", sk_test_acc)

# --- Sklearn DBSCAN on Dataset 2 ---
sk_dbscan2 = DBSCAN(eps=4.0, min_samples=5, metric='euclidean')
sk_dbscan2.fit(X_train)
sk_train_labels_db = sk_dbscan2.labels_

# Map each DBSCAN cluster (ignoring noise) to a majority label.
unique_clusters = np.unique(sk_train_labels_db)
mapping_dbscan = {}
for c in unique_clusters:
    if c == -1:
        continue
    y_in_cluster = y_train[sk_train_labels_db == c]
    mapping_dbscan[c] = 1 if np.sum(y_in_cluster) >= (len(y_in_cluster)/2) else 0

sk_y_train_pred_db = []
for i in range(len(X_train)):
    c = sk_train_labels_db[i]
    if c == -1:
        sk_y_train_pred_db.append(0)
    else:
        sk_y_train_pred_db.append(mapping_dbscan[c])
sk_y_train_pred_db = np.array(sk_y_train_pred_db)
sk_train_acc_db = accuracy_score(y_train, sk_y_train_pred_db)

# For test data, assign label from the nearest training point's DBSCAN cluster.
def euclid_dist(a, b):
    return sqrt(np.sum((a - b)**2))

sk_y_test_pred_db = []
for xt in X_test:
    dists = [euclid_dist(xt, x) for x in X_train]
    nearest_idx = np.argmin(dists)
    c = sk_train_labels_db[nearest_idx]
    if c == -1:
        pred_label = 0
    else:
        pred_label = mapping_dbscan[c]
    sk_y_test_pred_db.append(pred_label)
sk_y_test_pred_db = np.array(sk_y_test_pred_db)
sk_test_acc_db = accuracy_score(y_test, sk_y_test_pred_db)
print("Sklearn DBSCAN (Dataset 2) Train Accuracy =", sk_train_acc_db)
print("Sklearn DBSCAN (Dataset 2) Test Accuracy =", sk_test_acc_db)

# ---------------------------
# Q16: Observations and Comparisons
# ---------------------------
print("\nQ16) Observations and Comparisons:")
print("\nDataset 1:")
print(" - Sklearn K-Means produced labels similar to our custom implementation (3 clusters).")
print(" - The AgglomerativeClustering (with radius_neighbors_graph as connectivity) as a surrogate for NN clustering produced connectivity‐based clusters similar to our custom NN clustering.")
print(" - Sklearn DBSCAN (with eps=2, min_samples=2) produced noise points (label -1) and valid clusters, which is expected when some points are not densely connected.")
print("\nDataset 2:")
print(" - Sklearn K-Means (with k=2) yielded a training accuracy of", sk_train_acc, "and test accuracy of", sk_test_acc, "when clusters were mapped to chd labels by majority vote.")
print(" - Sklearn DBSCAN (with eps=4, min_samples=5) yielded a training accuracy of", sk_train_acc_db, "and test accuracy of", sk_test_acc_db, ".")
print("\nOverall Observations:")
print(" - K-Means is simple and fast but requires pre-specifying the number of clusters (k) and assumes spherical clusters.")
print(" - NN clustering (or its surrogate using AgglomerativeClustering) can capture arbitrary shapes based on connectivity but may be slower (O(n²)) and sensitive to the chosen threshold.")
print(" - DBSCAN automatically detects clusters of arbitrary shape and noise; however, its performance is highly sensitive to the eps and min_samples parameters.")
print(" - For Dataset 2, differences in accuracy among these methods highlight that using only 'sbp' and 'tobacco' may not fully capture the separation between chd classes.")

"""# Clustering Evaluation Metrics and Alternative Evaluation Methods

## 1) What are the different metrics used to evaluate the performance of different clustering algorithms?

- **Internal Metrics** (do not require labeled data):  
  - **Silhouette Score**: Quantifies how closely related data points are within their cluster compared to other clusters, with higher values indicating better-defined clusters.  
  - **Davies-Bouldin Index**: Assesses the average similarity between each cluster and its closest counterpart, where lower values reflect greater separation between clusters.  
  - **Calinski-Harabasz Index**: Evaluates cluster quality by comparing the variance between clusters to the variance within clusters, with higher values indicating stronger cluster separation.  

- **External Metrics** (require labeled data for validation):  
  - **Accuracy**: Determines the percentage of correctly matched labels after aligning clusters with ground truth classes.  
  - **Adjusted Rand Index (ARI)**: Measures the similarity between predicted clusters and true labels, accounting for random chance.  
  - **Normalized Mutual Information (NMI)**: Calculates the shared information between clustering results and true labels, normalized to a [0,1] scale.  
  - **Fowlkes-Mallows Index**: Combines precision and recall to evaluate the overlap between clustering and true class assignments.  

## 2) Instead of accuracy, what other ways can we evaluate the performance of the three algorithms above?

To comprehensively assess clustering performance, consider these approaches:  
- **Confusion Matrix**: Provides a granular view of clustering accuracy by mapping cluster labels to true labels and displaying true/false positives/negatives.  
- **ARI/NMI**: These metrics offer a more robust evaluation than accuracy by accounting for label permutations and chance agreements.  
- **Internal Metrics**: Use Silhouette Score, Davies-Bouldin, or Calinski-Harabasz to evaluate cluster cohesion and separation without relying on labels.  
- **Stability Analysis**: Tests the consistency of clustering results across varying initial conditions or subsamples of the data.  
- **Computational Efficiency**: Measures runtime and resource usage, which is critical for large-scale datasets.  

By combining these methods, practitioners gain a holistic understanding of clustering performance, encompassing structural quality, label alignment, robustness, and scalability.





"""