"""Plotting functions"""

import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, y, ax=None):
    """Plot data with labels"""
    if not ax:
        fig, ax = plt.subplots(1,1,figsize=(6,4))
    ax.scatter(X[:,0], X[:,1], c=y, cmap="viridis", s=10, alpha=0.8)
    classes = len(np.unique(y))
    ax.set(xlabel="x1", ylabel="x2", title=f'Full dataset ({classes} classes)')
    ax.grid(alpha=0.3)
    return fig, ax

def plot_split(X_split, y_split, title, ax):
    ax.scatter(X_split[:,0], X_split[:,1], c=y_split, cmap="viridis", s=10, alpha=0.8)
    ax.set(xlabel="x1", ylabel="x2", title=title)
    ax.grid(alpha=0.3)
    return ax

def plot_split_all(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
):
    fig, axs = plt.subplots(1,3, figsize=(14, 4))
    plot_split(X_train, y_train, "Train Set", axs[0])
    plot_split(X_val,   y_val,   "Validation Set", axs[1])
    plot_split(X_test,  y_test,  "Test Set", axs[2])
    n_classes = len(np.unique(y_train))
    plt.suptitle(f"{n_classes}-Class Dataset Splits", fontsize=14)
    plt.show()
    return fig

def plot_class_counts(train_counts, val_counts, test_counts, n_classes):
    fig, ax = plt.subplots(figsize=(6,4))

    labels = [f"Class {i}" for i in range(n_classes)]
    x = np.arange(n_classes)

    train_vals = [train_counts[i] for i in range(n_classes)]
    val_vals   = [val_counts[i]   for i in range(n_classes)]
    test_vals  = [test_counts[i]  for i in range(n_classes)]

    width = 0.22
    ax.bar(x - width, train_vals, width, label='Train')
    ax.bar(x,         val_vals,   width, label='Validation')
    ax.bar(x + width, test_vals,  width, label='Test')

    ax.set_ylabel("Samples")
    ax.set_title("Class Distribution Across Splits")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.show()
    return fig