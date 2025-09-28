import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class GNB:
    def __init__(self) -> None:
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.labels = np.unique(y)
        n_features = X.shape[1]
        self.means = np.zeros((len(self.labels), n_features))
        self.variances = np.zeros((len(self.labels), n_features))
        self.priors = np.zeros(len(self.labels))

        for i, label in enumerate(self.labels):
            X_lab = X[y == label]
            self.means[i] = X_lab.mean(axis=0)
            self.variances[i] = X_lab.var(axis=0)
            self.priors[i] = len(X_lab) / len(X)

    @staticmethod
    def gaussian_pdf(x: float, mean: float, variance: float):
        coefficient = 1.0 / np.sqrt(2 * np.pi * variance)
        exponent = -0.5 * ((x - mean) ** 2) / variance
        return coefficient * np.exp(exponent)

    def get_label_proba(self, point: np.ndarray) -> np.ndarray:
        label_proba = []
        for i in range(len(self.labels)):
            posterior = self.priors[i]
            for j, feature in enumerate(point):
                density = self.gaussian_pdf(
                    feature, self.means[i, j], self.variances[i, j]
                )
                posterior *= density
            label_proba.append(posterior)
        return np.asarray(label_proba) / sum(label_proba)

    def predict(self, X: np.ndarray, show_proba: bool) -> np.ndarray:
        point_proba = list(map(self.get_label_proba, X))
        if show_proba:
            return point_proba
        return [self.labels[np.argmax(point)] for point in point_proba]


X, y = make_classification(
    n_samples=1_000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

clf = GNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test, show_proba=False)

ax = sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
ax.set_xlabel("Pred")
ax.set_ylabel("True")
plt.show()
