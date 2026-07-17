"""
CART decision tree — refactored into a class from src/decision_tree/model.ipynb.

This is the *base learner* for the random forest. Wrapping it in a scikit-learn
style estimator (`fit` / `predict`) lets the forest notebook focus on the
ensemble logic (bagging + aggregation) rather than re-deriving the tree.

One addition for the forest: `max_features` restricts each split to a random
subset of the columns, redrawn at every node. That per-node randomness is the
"feature subsampling" that decorrelates the trees. Leave it as None and this
behaves exactly like the plain decision tree from the original notebook.
"""
import numpy as np

class CARTDecisionTree:

    def __init__(self, max_depth: int = 5,
                 min_samples_split: int = 2,
                 max_features: int | None = None,
                 random_state: int | np.random.Generator | None = None):
        # random_state takes a Generator as well as a seed — np.random.default_rng
        # returns a Generator unchanged. That lets a forest hand every tree the one
        # shared generator, so each tree's per-node column draws differ. Handing
        # every tree the same *int* would give them all identical draws instead.
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.tree_ = None

    def _gini_impurity(self, labels: np.ndarray) -> float:
        """Gini impurity: 0 when pure, up to 0.5 for an even 2-class mix."""
        if len(labels) == 0:
            return 0.0

        label_counts = np.unique(labels, return_counts=True)[1]
        proportions = label_counts / np.sum(label_counts)
        return 1 - np.sum(proportions ** 2)


    def _split_dataset(self, X: np.ndarray, y: np.ndarray, feature_index: int, threshold: float):
        """Samples with feature <= threshold go LEFT, the rest go RIGHT."""
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]


    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
        """
        Greedily search for the split with the lowest weighted child impurity.

        Called exactly once per node, so anything random in here is redrawn at
        every node — which is precisely how a forest gets a fresh feature
        subset per split, for free, via the recursion in `_build_tree`.

        Args:
            X: 2D array of features, shape (n_samples, n_features).
            y: 1D array of labels.

        Returns:
            {'feature_index', 'threshold', 'impurity'} or None if no valid split.
        """
        n_samples, n_features = X.shape

        if self.max_features is None:
            # Plain decision tree: search every column, deterministically.
            feature_indices = range(n_features)
        else:
            # Never ask for more columns than exist (drawing without replacement
            # would raise), and never ask for zero (nothing left to split on).
            n_draw = max(1, min(self.max_features, n_features))

            feature_indices = self.rng.choice(n_features, size=n_draw, replace=False)

        best = None
        for feature_index in feature_indices:
            for threshold in np.unique(X[:, feature_index]):
                X_left, y_left, X_right, y_right = self._split_dataset(X, y, feature_index, threshold)

                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                weighted_impurity = (
                    (len(y_left) / n_samples) * self._gini_impurity(y_left)
                    + (len(y_right) / n_samples) * self._gini_impurity(y_right)
                )

                if best is None or weighted_impurity < best['impurity']:
                    best = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'impurity': weighted_impurity,
                    }

        return best


    def _majority_class(self, y: np.ndarray):
        """Most common label in y — a leaf predicts the majority vote."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]


    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """
        Recursively grow a decision tree.

        Returns a node dict:
        - Leaf:     {'leaf': True,  'prediction': <label>}
        - Internal: {'leaf': False, 'feature_index', 'threshold', 'left', 'right'}
        """
        if self._gini_impurity(y) == 0 or depth >= self.max_depth or len(y) < self.min_samples_split:
            return {'leaf': True, 'prediction': self._majority_class(y)}

        split = self._find_best_split(X, y)
        if split is None:
            return {'leaf': True, 'prediction': self._majority_class(y)}

        X_left, y_left, X_right, y_right = self._split_dataset(X, y, split['feature_index'], split['threshold'])
        left_tree = self._build_tree(X_left, y_left, depth + 1)
        right_tree = self._build_tree(X_right, y_right, depth + 1)

        return {
            'leaf': False,
            'feature_index': split['feature_index'],
            'threshold': split['threshold'],
            'left': left_tree,
            'right': right_tree,
        }


    def _predict_one(self, node, x: np.ndarray):
        """Route one sample down the tree to a leaf and return its prediction."""
        while not node['leaf']:
            if x[node['feature_index']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['prediction']


    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit a decision tree to the data."""
        self.tree_ = self._build_tree(X, y)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict a label for every row of X."""
        return np.array([self._predict_one(self.tree_, row) for row in X])
