from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import svm
from sklearn.metrics import RocCurveDisplay, auc, classification_report
from sklearn.model_selection import StratifiedKFold

from src.models.classifier import ClassifierModel


def plot_cross_validated_roc(
    X: np.array,
    y: np.array,
    n_splits: int = 6,
    random_state: Optional[int] = None,
    classifier: Optional[ClassifierModel] = None,
    target_names: Optional[str] = None,
) -> Tuple[plt.Figure, float, float]:
    """
    Plot ROC curves with cross-validation for a binary classifier.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data.
    y : array-like of shape (n_samples,)
        Target labels.
    n_splits : int, default=6
        Number of folds for cross-validation.
    random_state : int, default=None
        Random seed for reproducibility.
    classifier : estimator, default=None
        Classifier to use. If None, a linear SVM will be used.
    target_names : list, default=None
        Names of target classes for plot title. Should have at least two elements.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    mean_auc : float
        The mean AUC across all folds.
    std_auc : float
        The standard deviation of AUC across all folds.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    if classifier is None:
        classifier = svm.SVC(
            kernel="linear", probability=True, random_state=random_state
        )

    if target_names is None:
        target_names = ["Class 0", "Class 1"]

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X.iloc[train], y.iloc[train])
        viz = RocCurveDisplay.from_estimator(
            classifier,
            X.iloc[test],
            y.iloc[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
            plot_chance_level=(fold == n_splits - 1),
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability\n(Positive label '{target_names[1]}')",
    )
    ax.legend(loc="lower right")

    return fig, mean_auc, std_auc


def plot_classification_report(
    y_true: np.array,
    y_pred: np.array,
    figsize: Tuple[int, int] = (10, 6),
    output_dict: bool = True,
) -> plt.Figure:
    """
    Plot classification report as a heatmap.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    figsize : tuple, default=(10, 6)
        Figure size
    output_dict : bool, default=True
        Whether to return the classification report as a dict

    Returns:
    --------
    fig : matplotlib figure
        The figure containing the heatmap
    """
    # Get classification report as dictionary
    report = classification_report(y_true, y_pred, output_dict=output_dict)

    df = pd.DataFrame(report).T

    # Drop the 'support' column and the 'accuracy' row for the heatmap
    df_heat = df.drop("support", axis=1).drop("accuracy", axis=0)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        df_heat,
        annot=True,
        cmap="Blues",
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    # Set title and labels
    plt.title("Classification Report", fontsize=14)
    plt.tight_layout()

    return fig
