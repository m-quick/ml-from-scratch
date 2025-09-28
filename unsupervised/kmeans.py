import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

from utils import euclidean_distance


class KMeans:
    def __init__(
        self,
        k: int,
        max_iter: int = 300,
        random_seed: int | None = None,
        tolerance: float = 0.0001,
    ) -> None:
        self.k = k
        self.max_iter = max_iter  # follows sklearn
        self.tolerance = tolerance  # follows sklearn
        if random_seed:
            np.random.seed(random_seed)

    def init_centroids(self, X: np.ndarray) -> np.ndarray:
        dimensions_min = np.amin(X, axis=0)
        dimensions_max = np.amax(X, axis=0)
        return np.random.uniform(
            dimensions_min, dimensions_max, size=(self.k, X.shape[1])
        )

    def assign_label(self, centroids: np.ndarray, point: np.ndarray) -> int:
        distances = euclidean_distance(centroids, point)
        return np.argmin(distances)

    def get_all_labels(self, centroids: np.ndarray, X: np.ndarray) -> np.ndarray:
        labels = list(map(lambda point: self.assign_label(centroids, point), X))
        return np.array(labels)

    @staticmethod
    def get_new_centroids(
        X: np.ndarray, label_idx: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        new_centroids = [
            np.mean(X[idx], axis=0)[0] if any(idx) else centroids[i]
            for i, idx in enumerate(label_idx)
        ]
        return np.array(new_centroids)  # to do array operations later

    def fit(self, X: np.ndarray) -> None:
        centroids = self.init_centroids(X)
        for i in range(self.max_iter):
            labels = self.get_all_labels(centroids, X)
            label_idx = [np.argwhere(labels == label) for label in range(self.k)]
            new_centroids = self.get_new_centroids(X, label_idx, centroids)
            if np.max(abs(new_centroids - centroids)) < self.tolerance:
                print("Stopped after", i + 1, "iterations")
                break
            centroids = new_centroids
        self.centroids = centroids

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "centroids"):
            raise Exception("Centroids don't exist. Call .fit() first")
        return self.get_all_labels(self.centroids, X)


data = make_blobs(n_samples=1_000, n_features=2, centers=3, random_state=42)
X = data[0]

scores = []

for i in range(2, 6):
    kmeans = KMeans(k=i, random_seed=42)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)

best_score = max(scores)
best_split_k = scores.index(best_score)
print("Best split with K:", best_split_k + 2, "Silhouette score:", round(best_score, 2))

plt.plot(list(range(2, 6)), scores)
plt.show()
