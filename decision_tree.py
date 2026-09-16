# -*- coding: utf-8 -*-
"""
decision_tree.py — CART decision tree reference implementation (provided source)
================================================================================

This file implements a CART classification tree based on Gini impurity. The API
style matches scikit-learn (fit / predict / predict_proba) so students can
hand-write a random forest on top of it.

【Important】Pay close attention to the max_features parameter:
    - Default max_features=None: each best-split search considers 【all features】
      — a standard decision tree.
    - When max_features is an integer m: before each split search, 【m features
      are randomly sampled】 and the best split is chosen only among them.
      This is the source of "feature randomness" in Random Forest!
      When implementing RF, pass an appropriate max_features when constructing
      each tree, and combine with "sample randomness" (bootstrap), to reuse
      DecisionTreeClassifier as the base learner.

Class / method overview:
    DecisionTreeClassifier(max_depth=6, min_samples_split=2,
                            min_samples_leaf=1, max_features=None,
                            random_state=None)
        .fit(X, y)                -> self
        .predict(X)                -> np.ndarray, predicted class
        .predict_proba(X)          -> np.ndarray, shape (n_samples, 2);
                                       positive class (larger label) probability in column 1
        .feature_importances_      -> attribute available after training
"""

from __future__ import annotations

import numpy as np


class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value", "n_samples")

    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None  # leaf node: [P(class0), P(class1)]
        self.n_samples = 0

    @property
    def is_leaf(self):
        return self.value is not None


def _gini(y):
    """Compute Gini impurity: Gini = 1 - sum(p_k^2)"""
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - np.sum(p ** 2)


class DecisionTreeClassifier:
    """Binary CART decision tree based on Gini impurity."""

    def __init__(self, max_depth=6, min_samples_split=2, min_samples_leaf=1,
                 max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.root_ = None
        self.classes_ = None
        self.n_features_ = None
        self.feature_importances_ = None
        self._rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("This implementation only supports binary classification.")

        self.n_features_ = X.shape[1]
        self._importance_accum = np.zeros(self.n_features_)

        self.root_ = self._build_tree(X, y, depth=0)

        total = self._importance_accum.sum()
        if total > 0:
            self.feature_importances_ = self._importance_accum / total
        else:
            self.feature_importances_ = self._importance_accum
        return self

    def _build_tree(self, X, y, depth):
        node = _Node()
        node.n_samples = len(y)

        n_pos = np.sum(y == self.classes_[1])
        n_neg = np.sum(y == self.classes_[0])
        proba = np.array([n_neg, n_pos]) / max(len(y), 1)

        # Stopping criteria: max depth / too few samples / pure node
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            node.value = proba
            return node

        split = self._best_split(X, y)
        if split is None:
            node.value = proba
            return node

        feature, threshold, gain = split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            node.value = proba
            return node

        self._importance_accum[feature] += gain * len(y)

        node.feature = feature
        node.threshold = threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X, y):
        n_samples, n_features = X.shape
        parent_gini = _gini(y)
        if parent_gini == 0.0:
            return None

        # Feature randomness: key switch for random forest (see file header)
        if self.max_features is None:
            candidate_features = np.arange(n_features)
        else:
            m = min(self.max_features, n_features)
            candidate_features = self._rng.choice(n_features, size=m, replace=False)

        best_gain = 0.0
        best_feature, best_threshold = None, None

        for feature in candidate_features:
            values = np.unique(X[:, feature])
            if len(values) <= 1:
                continue
            # Candidate thresholds: midpoints between adjacent values (cap count for speed)
            thresholds = (values[:-1] + values[1:]) / 2.0
            if len(thresholds) > 32:
                idx = np.linspace(0, len(thresholds) - 1, 32).astype(int)
                thresholds = thresholds[idx]

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                n_left, n_right = left_mask.sum(), (~left_mask).sum()
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue

                gini_left = _gini(y[left_mask])
                gini_right = _gini(y[~left_mask])
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_feature is None:
            return None
        return best_feature, best_threshold, best_gain

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def predict(self, X):
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        return self.classes_[pred_idx]

    def _predict_one(self, x):
        node = self.root_
        while not node.is_leaf:
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)


if __name__ == "__main__":
    # Minimal self-test: ensure the tree fits a simple separable dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=300, n_features=6, n_informative=4,
                                random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_train, y_train)
    acc = tree.score(X_test, y_test)
    print(f"Self-test: single decision tree accuracy on random data = {acc:.3f}")
    print("Feature importances:", np.round(tree.feature_importances_, 3))
