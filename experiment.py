# -*- coding: utf-8 -*-
"""Compare course-provided trees and SMOTE with a completed random forest.

Adapted from the course student_template.py; see README.md credits.
Run python experiment.py with the course loan.csv in this folder.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from smote import SMOTE
from decision_tree import DecisionTreeClassifier
from utils import evaluate_model, print_metrics_table, plot_metrics_comparison

RANDOM_STATE = 42


# ======================================================================
# Part 1: Data loading and preprocessing (adapted from course scaffold)
# ======================================================================
TARGET_COLUMN = "Personal Loan"  # Target: whether the customer accepted the personal loan offer (1=yes, 0=no)
DROP_COLUMNS = ["ID", "ZIP Code"]  # ID has no predictive value; ZIP Code is high-cardinality geo code — dropped for this lab


def load_and_preprocess_data(csv_path="loan.csv"):
    """Load loan.csv, apply basic cleaning, and split train/test.

    See Lab Manual Section 4 for field descriptions. Only two cleaning steps here:
        1. Drop ID and ZIP Code, which do not help prediction directly;
        2. Correct a few negative Experience values (data-entry errors in the raw data).
    Other fields (Family, Education, etc.) are small-range integer encodings and can be
    used directly as numeric features for threshold-based trees/forests — no one-hot needed.
    """
    df = pd.read_csv(csv_path)

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df["Experience"] = df["Experience"].abs()

    y = df[TARGET_COLUMN].values
    X = df.drop(columns=[TARGET_COLUMN]).values
    feature_names = df.drop(columns=[TARGET_COLUMN]).columns.tolist()

    X = X.astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    # Simple standardization (not required for tree models, but good practice; does not change tree results)
    mean, std = X_train.mean(axis=0), X_train.std(axis=0)
    std[std == 0] = 1.0
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test, y_train, y_test, feature_names


# ======================================================================
# Part 2 : Call provided SMOTE to oversample the training set
# ======================================================================
def apply_smote(X_train, y_train):
    """Balance only training data using the supplied SMOTE implementation."""
    smote = SMOTE(k_neighbors=5, sampling_strategy='auto', random_state=RANDOM_STATE)

    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print("Before SMOTE:", np.unique(y_train, return_counts=True))
    print("After SMOTE:", np.unique(y_train_smote, return_counts=True))
    return X_train_smote, y_train_smote

    # -----------------------------------------------------------------


# ======================================================================
# Part 3 : Hand-write Random Forest
# ======================================================================
class RandomForestScratch:
    """Random Forest (bootstrapped trees + probability averaging); base learner reused from decision_tree.py.

    Completed forest built on the supplied tree implementation.
    """

    def __init__(self, n_estimators=50, max_depth=6, max_features="sqrt",
                 random_state=None):
        """
        Parameters:
            n_estimators : number of trees
            max_depth    : max depth of each tree (passed to DecisionTreeClassifier)
            max_features : number of features randomly selected at each split; can be:
                             - 'sqrt': use sqrt(n_features) (common RF setting)
                             - int: specify the feature count directly
            random_state : random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state

        self.trees_ = []          # Fitted trees
        self.classes_ = None

    def _resolve_max_features(self, n_features):
        """Compute the actual number of features (int) from self.max_features."""
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        return n_features

    def fit(self, X, y):
        """Fit independently bootstrapped trees with reproducible sampling."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.trees_ = []
        self.classes_ = np.unique(y)

        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        for i in range(self.n_estimators):
            indices = rng.choice(
	            n_samples,
	            size=n_samples,
	            replace=True
	        )

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                max_features=self._resolve_max_features(n_features),
                random_state=None if self.random_state is None else self.random_state + i,
            )
            tree.fit(X[indices], y[indices])
            self.trees_.append(tree)

        return self
        # -----------------------------------------------------------------

    def predict_proba(self, X):
        """Average class probabilities across fitted trees."""
        X = np.asarray(X, dtype=float)
        if not self.trees_:
            raise ValueError("Fit the forest before predicting.")
        
        total_proba = 0
        for tree in self.trees_:
            total_proba += tree.predict_proba(X)
        return total_proba / len(self.trees_)
        # -----------------------------------------------------------------

    def predict(self, X):
        """
        Choose the class with the largest averaged probability.
        This need not equal a majority of hard tree predictions.
        """
        proba = self.predict_proba(X)
        pred_idx = np.argmax(proba, axis=1)
        return self.classes_[pred_idx]

    def score(self, X, y):
        y = np.asarray(y)
        return np.mean(self.predict(X) == y)


# ======================================================================
# Part 4: Train + evaluate + compare (adapted from course scaffold)
# ======================================================================
def run_experiment():
    print("=" * 70)
    print("Step 1: Load and preprocess loan.csv")
    X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_data()
    print(f"Train set size: {X_train.shape}, Test set size: {X_test.shape}")
    print("Train set class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))

    print("\nStep 2: Call SMOTE to oversample the training set (Task 1)")
    X_train_smote, y_train_smote = apply_smote(X_train, y_train)

    results = []

    print("\nStep 3: Train [Original data + single decision tree]")
    tree_raw = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
    tree_raw.fit(X_train, y_train)
    proba = tree_raw.predict_proba(X_test)[:, 1]
    pred = tree_raw.predict(X_test)
    results.append(evaluate_model(y_test, pred, proba, "Decision Tree (raw)"))

    print("Step 4: Train [Original data + random forest] (Task 2)")
    rf_raw = RandomForestScratch(n_estimators=50, max_depth=6,
                                  max_features="sqrt", random_state=RANDOM_STATE)
    rf_raw.fit(X_train, y_train)
    proba = rf_raw.predict_proba(X_test)[:, 1]
    pred = rf_raw.predict(X_test)
    results.append(evaluate_model(y_test, pred, proba, "Random Forest (raw)"))

    print("Step 5: Train [SMOTE data + single decision tree]")
    tree_smote = DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)
    tree_smote.fit(X_train_smote, y_train_smote)
    proba = tree_smote.predict_proba(X_test)[:, 1]
    pred = tree_smote.predict(X_test)
    results.append(evaluate_model(y_test, pred, proba, "Decision Tree (SMOTE)"))

    print("Step 6: Train [SMOTE data + random forest]")
    rf_smote = RandomForestScratch(n_estimators=50, max_depth=6,
                                    max_features="sqrt", random_state=RANDOM_STATE)
    rf_smote.fit(X_train_smote, y_train_smote)
    proba = rf_smote.predict_proba(X_test)[:, 1]
    pred = rf_smote.predict(X_test)
    results.append(evaluate_model(y_test, pred, proba, "Random Forest (SMOTE)"))

    print("\n" + "=" * 70)
    print("Four-way experiment comparison:")
    print_metrics_table(results)

    fig = plot_metrics_comparison(results, save_path="metrics_comparison.png")
    print("\nComparison bar chart saved as metrics_comparison.png")

    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    return results


if __name__ == "__main__":
    run_experiment()
