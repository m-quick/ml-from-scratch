import numpy as np
from node import Node
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class DecisionTree:
    def __init__(
        self, min_samples_split: int = 2, max_depth: int | None = None
    ) -> None:
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    @staticmethod
    def calculate_entropy(samples: list[int]) -> float:
        class_shares = np.bincount(samples) / len(samples)
        return -sum([i * np.log2(i) for i in class_shares if i > 0])

    def stop_growing(self, depth: int, samples: list[int]) -> bool:
        n_unique_samples = len(np.unique(samples))
        if n_unique_samples == 1:
            return True
        if len(samples) < self.min_samples_split:
            return True
        if self.max_depth is not None:
            if depth >= self.max_depth:
                return True
        return False

    @staticmethod
    def split_feature(
        values: np.ndarray, threshold: int | float | str
    ) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(threshold, (float, int)):
            left_idx = np.argwhere(values < threshold)
            right_idx = np.argwhere(values >= threshold)
        elif isinstance(threshold, str):
            left_idx = np.argwhere(values == threshold)
            right_idx = np.argwhere(values != threshold)
        else:
            raise ValueError(
                f"Threshold must be either a number or a string but a {type(threshold)} was passed"
            )
        return left_idx.flatten(), right_idx.flatten()

    def get_information_gain(
        self,
        samples: list[int],
        feature_values: np.ndarray,
        threshold: int | float | str,
    ) -> float:
        parent_entropy = self.calculate_entropy(samples)
        left_idx, right_idx = self.split_feature(feature_values, threshold)
        left_entropy = self.calculate_entropy(samples[left_idx])
        right_entropy = self.calculate_entropy(samples[right_idx])
        children_entropy = np.average(
            [left_entropy, right_entropy], weights=[len(left_idx), len(right_idx)]
        )
        return parent_entropy - children_entropy

    def find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        random_col_order = np.random.choice(
            X.shape[1], X.shape[1], replace=False
        )  # avoids always splitting by same feature if gain is identical across features
        biggest_gain = -1
        best_split_feature, best_split_threshold = None, None
        for col in random_col_order:
            unique_vals = np.unique(X[:, col])
            for val in unique_vals:
                gain = self.get_information_gain(y, X[:, col], val)
                if gain > biggest_gain:
                    biggest_gain = gain
                    best_split_feature = col
                    best_split_threshold = val
        return best_split_feature, best_split_threshold

    def add_node(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        if self.stop_growing(depth, y):
            return Node(
                split_feature=None,
                split_threshold=None,
                samples=y,
                left=None,
                right=None,
            )
        split_feature, split_threshold = self.find_best_split(X, y)
        left_idx, right_idx = self.split_feature(X[:, split_feature], split_threshold)
        left_branch = self.add_node(X[left_idx,], y[left_idx], depth + 1)
        right_branch = self.add_node(X[right_idx,], y[right_idx], depth + 1)
        return Node(
            split_feature=split_feature,
            split_threshold=split_threshold,
            samples=None,
            left=left_branch,
            right=right_branch,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.tree = self.add_node(X, y, depth=0)

    def traverse_tree(self, X: np.ndarray, node: Node):
        if node.samples is not None:
            return np.mean(node.samples)

        if isinstance(node.split_treshold, (float, int)):
            go_left = X[node.split_feature] < node.split_treshold
        elif isinstance(node.split_treshold, str):
            go_left = X[node.split_feature] == node.split_treshold

        next_node = node.left if go_left else node.right
        return self.traverse_tree(X, next_node)

    def predict(self, X: np.ndarray, show_proba: bool) -> None:
        n_digits = int(show_proba) * 3
        return [round(self.traverse_tree(row, self.tree), n_digits) for row in X]


if __name__ == "__main__":
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    clf = DecisionTree(min_samples_split=2, max_depth=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test, show_proba=False)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
