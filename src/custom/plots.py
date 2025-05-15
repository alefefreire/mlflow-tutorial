from itertools import cycle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from sklearn.preprocessing import label_binarize


def plot_roc_curves_after_cv(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    target_names: List[str],
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Plot ROC curves after using cross-validation.

    This function plots ROC curves for each class, as well as micro-average
    and macro-average ROC curves, after performing cross-validation.

    Parameters
    ----------
    y_true : array-like
        True labels. Can be either in one-hot encoded format or multi-class format.
    y_pred_proba : array-like
        Predicted probabilities from model for each class.
    target_names : list
        Names of the classes.
    figsize : tuple, default=(12, 8)
        Size of the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the ROC curves.
    """
    # Check if y_true is already in one-hot format
    if len(y_true.shape) == 1:
        # Convert to one-hot format
        classes = np.unique(y_true)
        n_classes = len(classes)
        y_onehot = label_binarize(y_true, classes=classes)
    else:
        # Already in one-hot format
        y_onehot = y_true
        n_classes = y_onehot.shape[1]

    # Calculate ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Calculate micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Calculate macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Calculate average and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot micro-average ROC curve
    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    # Plot macro-average ROC curve
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    # Plot ROC curves for each class
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red", "purple"])
    for class_id, color in zip(range(n_classes), colors):
        RocCurveDisplay.from_predictions(
            y_onehot[:, class_id],
            y_pred_proba[:, class_id],
            name=f"ROC curve for {target_names[class_id]}",
            color=color,
            ax=ax,
            plot_chance_level=(
                class_id == n_classes - 1
            ),  # Plot chance line only for the last class
            despine=True,
        )

    # Set plot labels and title
    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="One-vs-Rest multiclass",
    )
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    return fig
