# -*- coding: utf-8 -*-
"""
utils.py — Evaluation and visualization helpers (provided source)
=================================================================

This file provides metric computation and plotting functions used repeatedly
in the lab. Students should call them directly; no modifications needed.

Functions:
    evaluate_model(y_true, y_pred, y_prob=None, model_name="Model")
        Compute Accuracy / Precision / Recall / F1 / AUC and return as a dict.

    print_metrics_table(results)
        Print evaluation results of multiple models (list of dicts) as a table.

    plot_confusion_matrix(y_true, y_pred, title="")
        Draw a confusion-matrix heatmap.

    plot_metrics_comparison(results, save_path=None)
        Draw grouped bar charts of metrics across models for comparison.

    set_chinese_font()
        Try to set a CJK-capable matplotlib font (legacy helper; charts use English labels).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def set_chinese_font():
    """Try common CJK fonts (Windows/Mac). Failure does not affect experiment results."""
    candidates = ["Microsoft YaHei", "SimHei", "PingFang SC", "Arial Unicode MS"]
    for font in candidates:
        try:
            plt.rcParams["font.sans-serif"] = [font]
            plt.rcParams["axes.unicode_minus"] = False
            return
        except Exception:
            continue


def evaluate_model(y_true, y_pred, y_prob=None, model_name="Model"):
    """Compute common binary classification metrics.

    Parameters:
        y_true : true labels
        y_pred : predicted labels (0/1)
        y_prob : predicted probability of the positive class (label 1), used for AUC;
                 if None, AUC is not computed.
        model_name : model name for display
    Returns:
        dict with accuracy / precision / recall / f1 / auc
    """
    result = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        try:
            result["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            result["auc"] = float("nan")
    else:
        result["auc"] = float("nan")
    return result


def print_metrics_table(results):
    """Print a comparison table of evaluation results for multiple models.

    results : list[dict], each dict is the return value of evaluate_model
    """
    header = f"{'Model':<28}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>9}{'AUC':>9}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<28}"
            f"{r['accuracy']:>10.3f}"
            f"{r['precision']:>11.3f}"
            f"{r['recall']:>9.3f}"
            f"{r['f1']:>9.3f}"
            f"{r['auc']:>9.3f}"
        )


def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", save_path=None):
    set_chinese_font()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_metrics_comparison(results, save_path=None):
    """Draw grouped bar charts of metrics across models for side-by-side comparison.

    results : list[dict], each dict is the return value of evaluate_model
    """
    set_chinese_font()
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    n_models = len(results)
    x = np.arange(len(metrics))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(results):
        values = [r[m] for m in metrics]
        ax.bar(x + i * width, values, width=width, label=r["model"])

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "AUC"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
