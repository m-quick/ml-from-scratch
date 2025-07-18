import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class OLS:
    def __init__(self, learning_rate: float = 0.0001, epochs: int = 10_000):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        weights = np.zeros(X.shape) if len(X.shape) == 2 else 0
        bias = 0
        scale = 1 / X.shape[0]
        for _ in range(self.epochs):
            y_hat = bias + np.dot(X, weights)
            dw = scale * np.dot(X, (y_hat - y))
            db = scale * np.sum(y_hat - y)
            weights -= dw * self.learning_rate
            bias -= db * self.learning_rate
        self.weights = weights
        self.bias = bias

    def predict(self, X: np.ndarray) -> float:
        if not all([hasattr(self, attr) for attr in ["weights", "bias"]]):
            raise Exception("Parameters have not been set")
        return self.bias + np.dot(X, self.weights)

    def r_squared(self, y_hat: np.ndarray, y: np.ndarray) -> float:
        tss = sum([(np.mean(y) - i) ** 2 for i in y])
        rss = sum([(pred - true) ** 2 for pred, true in zip(y_hat, y)])
        return 1 - (rss / tss)


np.random.seed(42)
n_samples = 100
x = np.random.uniform(0, 10, n_samples)
noise = np.random.normal(0, 0.5, n_samples)
b0, b1 = 5, 0.4
y = b0 + (b1 * x) + noise

fig, ax = plt.subplots(figsize=(15, 6))

ax.scatter(x, y)

df = pd.DataFrame({"x": x})

n_epochs = [i for i in range(10_000, 500_000, 50_000)]

for n in n_epochs:
    ols = OLS(epochs=n)
    start = time.time()
    ols.fit(x, y)
    end = time.time()
    y_hat = ols.predict(x)
    print(
        "Run duration:",
        round((end - start), 2),
        f"N epochs: {n:_}",
        f"R2: {ols.r_squared(y_hat, y):.2f}",
    )
    df[f"{n:_}"] = y_hat

df.set_index("x").plot.line(ax=ax)

plt.show()
