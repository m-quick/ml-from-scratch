from statistics import mode

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from utils import euclidean_distance


class KNN:
    def __init__(self, k: int) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        labels = set(y)
        print(f"Training data contains {len(labels)} labels")
        self.label_points_map = {
            label: np.array([i for i, j in zip(X, y) if j == label]) for label in labels
        }

    def assign_label(self, new_point: np.ndarray) -> int:
        all_distances = []
        for label, points in self.label_points_map.items():
            distances = euclidean_distance(points, new_point)
            all_distances.extend([[i, label] for i in distances])
        neighbour_labels = [i[1] for i in sorted(all_distances)[: self.k]]
        return mode(neighbour_labels)

    def predict(self, new_points: np.ndarray) -> list[int]:
        if not hasattr(self, "label_points_map"):
            raise Exception("Model has not been fitted. Call the .fit() method first")
        return list(map(self.assign_label, new_points))


X, y = make_classification(
    n_samples=1_000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


clf = KNN(k=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
ax.set_xlabel("Pred")
ax.set_ylabel("True")
plt.show()
