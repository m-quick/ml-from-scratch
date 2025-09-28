import numpy as np


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    squared_diffs = (a - b) ** 2
    return np.sqrt(np.sum(squared_diffs, axis=1))
