"""
Model Training Module for Agrochemical Health Prediction
=========================================================

This module implements training pipelines for 13 machine learning algorithms
to predict health problems in agricultural workers exposed to agrochemicals.

Author: Luis José Yudico-Anaya
License: MIT
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def get_model_configs():
    """
    Get configuration dictionary for all 13 models.

    Returns
    -------
    dict
        Dictionary with model names and configured instances
    """
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        ),
        'SVM': SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        ),
        'Naive Bayes': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=50,
            random_state=42
        ),
        'Ridge Classifier': RidgeClassifier(
            random_state=42
        )
    }

    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

    # Add LightGBM if available
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    return models


def get_scoring_metrics():
    """
    Define scoring metrics for model evaluation.

    Returns
    -------
    dict
        Dictionary of scoring functions
    """
    return {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='binary', zero_division=0),
        'recall': make_scorer(recall_score, average='binary', zero_division=0),
        'f1': make_scorer(f1_score, average='binary', zero_division=0),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True)
    }


def train_with_cross_validation(model, X, y, cv_folds=5, random_state=42):
    """
    Train model using stratified k-fold cross-validation.

    Parameters
    ----------
    model : sklearn estimator
        Machine learning model to train
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv_folds : int
        Number of cross-validation folds (default: 5)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    dict
        Cross-validation results with mean and std for each metric
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = get_scoring_metrics()

    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    # Calculate mean and std for each metric
    results = {}
    for metric in scoring.keys():
        results[f'{metric}_mean'] = np.mean(cv_results[f'test_{metric}'])
        results[f'{metric}_std'] = np.std(cv_results[f'test_{metric}'])

    return results


def compare_all_models(X, y, cv_folds=5, random_state=42):
    """
    Compare all 13 models using cross-validation.

    Parameters
    ----------
    X : array-like
        Feature matrix
    y : array-like
        Target vector
    cv_folds : int
        Number of cross-validation folds
    random_state : int
        Random seed

    Returns
    -------
    dict
        Dictionary with results for each model
    """
    models = get_model_configs()
    all_results = {}

    print("Training and evaluating models...")
    print("-" * 80)

    for name, model in models.items():
        print(f"Training {name}...")
        results = train_with_cross_validation(model, X, y, cv_folds, random_state)
        all_results[name] = results

        print(f"  Accuracy: {results['accuracy_mean']:.4f} (±{results['accuracy_std']:.4f})")
        print(f"  F1-Score: {results['f1_mean']:.4f} (±{results['f1_std']:.4f})")
        print()

    return all_results


if __name__ == "__main__":
    print("Model training utilities loaded successfully.")
    print(f"Available models: {len(get_model_configs())}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
