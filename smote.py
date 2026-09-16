# -*- coding: utf-8 -*-
"""
smote.py — SMOTE (Synthetic Minority Over-sampling Technique) reference implementation
=======================================================================================

This file is 【provided source code】. Students only need to import and call it
in their own scripts; do not modify the internal implementation.

Method recap (covered in class; key steps here for reading the code):
    For each minority-class sample x_i:
        1. Find its k nearest neighbors within the minority-class set;
        2. Randomly pick one (or more) neighbor(s) x_nn from those k;
        3. Pick a random point on the line between x_i and x_nn as a synthetic sample:
                x_new = x_i + gap * (x_nn - x_i),   gap ~ Uniform(0, 1)
    Repeat until the minority class reaches the target count.

Usage example:
    from smote import SMOTE

    smote = SMOTE(k_neighbors=5, sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

Parameters:
    k_neighbors : int, default 5
        The k in the paper — number of neighbors for each minority sample.

    sampling_strategy : 'auto' or float, default 'auto'
        - 'auto': oversample the minority class to 【equal】 the majority (1:1).
        - float (between 0 and 1, e.g. 0.5): after oversampling,
              minority count = float * majority count.
          e.g. sampling_strategy=0.5 means minority becomes half of majority.

    random_state : int or None
        Random seed for reproducibility.

Returns:
    fit_resample(X, y) returns (X_resampled, y_resampled),
    keeping all original samples and adding synthetic minority samples.
    Note: use this only on the 【training set】 — never apply SMOTE to the test set!
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class SMOTE:
    """SMOTE oversampler (binary classification; continuous/numeric features)."""

    def __init__(self, k_neighbors: int = 5, sampling_strategy="auto", random_state=None):
        self.k_neighbors = k_neighbors
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state

    def fit_resample(self, X, y):
        """Apply SMOTE oversampling to (X, y); return (X_resampled, y_resampled).

        X : array-like, shape (n_samples, n_features), must be numeric
        y : array-like, shape (n_samples,), binary labels (0/1 or any two values)
        """
        rng = np.random.RandomState(self.random_state)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        if len(classes) != 2:
            raise ValueError("This implementation only supports binary classification (minority vs majority).")

        minority_class = classes[np.argmin(counts)]
        majority_class = classes[np.argmax(counts)]
        n_minority = counts.min()
        n_majority = counts.max()

        # How many new samples to synthesize
        if self.sampling_strategy == "auto":
            n_target = n_majority
        else:
            ratio = float(self.sampling_strategy)
            n_target = int(round(ratio * n_majority))

        n_synthetic = max(n_target - n_minority, 0)
        if n_synthetic == 0:
            return X.copy(), y.copy()

        X_minority = X[y == minority_class]

        k = min(self.k_neighbors, len(X_minority) - 1)
        if k < 1:
            raise ValueError("Too few minority samples to compute neighbors; reduce k_neighbors.")

        # Find k neighbors within the minority class (as in the paper)
        nn_model = NearestNeighbors(n_neighbors=k + 1)  # +1 because the point itself is included
        nn_model.fit(X_minority)
        _, neighbor_indices = nn_model.kneighbors(X_minority)
        neighbor_indices = neighbor_indices[:, 1:]  # drop self

        synthetic_samples = np.zeros((n_synthetic, X.shape[1]))
        for idx in range(n_synthetic):
            # 1) Randomly pick a minority sample i
            i = rng.randint(0, len(X_minority))
            # 2) Randomly pick one of its k neighbors
            nn_choice = neighbor_indices[i, rng.randint(0, k)]
            x_i = X_minority[i]
            x_nn = X_minority[nn_choice]
            # 3) Random point on the line segment
            gap = rng.rand()
            synthetic_samples[idx] = x_i + gap * (x_nn - x_i)

        X_resampled = np.vstack([X, synthetic_samples])
        y_resampled = np.concatenate(
            [y, np.full(n_synthetic, minority_class, dtype=y.dtype)]
        )

        # Shuffle so synthetic samples are not all at the end
        shuffle_idx = rng.permutation(len(X_resampled))
        return X_resampled[shuffle_idx], y_resampled[shuffle_idx]


if __name__ == "__main__":
    # Minimal self-test: create an imbalanced dataset and check oversampling.
    rng = np.random.RandomState(0)
    X_majority = rng.normal(loc=0.0, scale=1.0, size=(200, 2))
    X_minority = rng.normal(loc=3.0, scale=1.0, size=(20, 2))
    X_demo = np.vstack([X_majority, X_minority])
    y_demo = np.concatenate([np.zeros(200), np.ones(20)])

    print("Class distribution before oversampling:", dict(zip(*np.unique(y_demo, return_counts=True))))
    smote = SMOTE(k_neighbors=5, sampling_strategy="auto", random_state=42)
    X_new, y_new = smote.fit_resample(X_demo, y_demo)
    print("Class distribution after oversampling:", dict(zip(*np.unique(y_new, return_counts=True))))
