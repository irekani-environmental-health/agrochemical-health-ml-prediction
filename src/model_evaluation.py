"""
Model Evaluation and Visualization Module
=========================================

This module provides functions for evaluating ML models and generating
visualizations including ROC curves, feature importance, and SHAP analysis.

Author: Luis José Yudico-Anaya
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    classification_report, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Comprehensive evaluation of a trained model.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    model_name : str
        Name of the model for display

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics
    """
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'cohen_kappa': cohen_kappa_score(y_test, y_pred)
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)

    # Print report
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {model_name}")
    print(f"{'='*60}")
    for metric, value in metrics.items():
        if metric != 'model_name':
            print(f"{metric.upper():15s}: {value:.4f}")
    print(f"{'='*60}\n")

    return metrics


def plot_roc_curve(models_dict, X_test, y_test, figsize=(10, 8)):
    """
    Plot ROC curves for multiple models.

    Parameters
    ----------
    models_dict : dict
        Dictionary with model names as keys and trained models as values
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    figsize : tuple
        Figure size (width, height)

    Returns
    -------
    matplotlib.figure.Figure
        ROC curve figure
    """
    plt.figure(figsize=figsize)

    for name, model in models_dict.items():
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)

    # Plot diagonal
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def plot_confusion_matrix(y_test, y_pred, model_name="Model", figsize=(8, 6)):
    """
    Plot confusion matrix heatmap.

    Parameters
    ----------
    y_test : array-like
        True labels
    y_pred : array-like
        Predicted labels
    model_name : str
        Name of the model
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Confusion matrix figure
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['No Problems', 'Has Problems'],
                yticklabels=['No Problems', 'Has Problems'])
    plt.title(f'Confusion Matrix: {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    return plt.gcf()


def plot_feature_importance(model, feature_names, top_n=15, figsize=(10, 8)):
    """
    Plot feature importance for tree-based models.

    Parameters
    ----------
    model : sklearn estimator
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of features
    top_n : int
        Number of top features to display
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Feature importance figure
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return None

    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=figsize)
    plt.barh(range(top_n), importances[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    return plt.gcf()


def generate_shap_summary(model, X, feature_names, max_display=15):
    """
    Generate SHAP summary plot for model interpretability.

    Parameters
    ----------
    model : sklearn estimator
        Trained model
    X : array-like
        Feature data
    feature_names : list
        Names of features
    max_display : int
        Maximum number of features to display

    Returns
    -------
    matplotlib.figure.Figure
        SHAP summary figure
    """
    try:
        import shap

        # Create explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Create summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names,
                         max_display=max_display, show=False)
        plt.tight_layout()

        return plt.gcf()

    except ImportError:
        print("SHAP library not installed. Install with: pip install shap")
        return None


if __name__ == "__main__":
    print("Model evaluation utilities loaded successfully.")
    print("Functions available:")
    print("  - evaluate_model()")
    print("  - plot_roc_curve()")
    print("  - plot_confusion_matrix()")
    print("  - plot_feature_importance()")
    print("  - generate_shap_summary()")
