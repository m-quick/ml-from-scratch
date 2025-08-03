import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, precision_score, recall_score


class Logit:
    def __init__(self, learning_rate: float = 0.001, epochs: int = 10_000) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs

    @staticmethod
    def sigmoid(x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
        n_samples, n_params = X_with_bias.shape
        params = np.zeros(n_params)
        for _ in range(self.epochs):
            dot_prod = np.dot(X_with_bias, params)
            estimates = self.sigmoid(dot_prod)
            error = estimates - y
            gradient = (X_with_bias.T @ error) / n_samples
            params -= gradient * self.learning_rate
        self.params = params

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "params"):
            raise Exception("Params have not been fitted")
        X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
        dot_prod = np.dot(X_with_bias, self.params)
        return self.sigmoid(dot_prod)

    def predict(self, X: np.ndarray, threshold: int = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)
        return [int(i > threshold) for i in proba]


X, y = make_classification(
    n_samples=1_000,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    random_state=42,
)

values = []

for n in range(10_000, 500_000, 50_000):
    clf = Logit(epochs=n)
    clf.fit(X, y)
    preds = clf.predict(X)

    scores = {"n": n}

    for label, metric in zip(
        ["Accuracy", "Recall", "Precision"],
        [accuracy_score, recall_score, precision_score],
    ):
        scores.update({label: metric(y, preds)})

    values.append(scores)

fig, ax = plt.subplots(figsize=(15, 6))
pd.DataFrame(values).set_index("n").plot.line(ax=ax)

plt.show()
