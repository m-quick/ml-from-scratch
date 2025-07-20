from typing import Self

import numpy as np


class Node:
    def __init__(
        self,
        split_feature: str | None,
        split_threshold: int | float | str | None,
        samples: list[int] | None,
        left: Self | None,
        right: Self | None,
    ) -> None:
        self.split_feature = split_feature
        self.split_treshold = split_threshold
        self.samples = samples
        self.left = left
        self.right = right

    def prediction(self, show_proba: bool) -> float | int:
        if self.samples is None:
            raise ValueError("Not a leaf node")
        proba = np.mean(self.samples)
        n_digits = int(show_proba) * 3
        return round(proba, n_digits)
