import numpy as np
from decision_tree import DecisionTree
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class RandomForest:
    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    @staticmethod
    def boostrap_samples(X: np.ndarray) -> tuple:
        n_samples, n_features = X.shape
        keep_n_features = int(np.sqrt(n_features))  # follows sklearn
        row_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        col_idx = np.random.choice(
            n_features, size=max(2, keep_n_features), replace=False
        )
        return row_idx, col_idx

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        forest: list[DecisionTree] = []
        for _ in range(self.n_trees):
            row_idx, col_idx = self.boostrap_samples(X)
            X_subset = X[np.ix_(row_idx, col_idx)]
            y_subset = y[row_idx]
            tree = DecisionTree(self.min_samples_split, self.max_depth)
            tree.fit(X_subset, y_subset)
            forest.append(tree)
        self.forest = forest

    def predict(self, X: np.ndarray, show_proba: bool) -> float | int:
        tree_votes = [tree.predict(X, show_proba=False) for tree in self.forest]
        tree_votes_swapped = np.swapaxes(tree_votes, 0, 1)
        n_digits = int(show_proba) * 3
        return [round(np.mean(i), n_digits) for i in tree_votes_swapped]


X, y = make_classification(
    n_samples=1_000,
    n_features=3,
    n_informative=3,
    n_redundant=0,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.25
)

clf = RandomForest()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test, show_proba=False)

print(accuracy_score(y_test, y_pred))
