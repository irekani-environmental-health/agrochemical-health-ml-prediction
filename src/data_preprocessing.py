"""
Data Preprocessing Pipeline for Agrochemical Health Prediction
===============================================================

This module handles data cleaning, feature engineering, and preprocessing
for the Irekani Environmental Health Project dataset.

Author: Luis José Yudico-Anaya
License: MIT
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


def load_data(filepath):
    """
    Load the Irekani dataset from CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file

    Returns
    -------
    pd.DataFrame
        Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df, strategy='mean'):
    """
    Handle missing values using specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    strategy : str
        Imputation strategy ('mean', 'median', 'most_frequent')

    Returns
    -------
    pd.DataFrame
        Dataframe with imputed values
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # Impute numeric columns
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy=strategy)
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Impute categorical columns
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    return df


def encode_categorical_variables(df, categorical_cols):
    """
    Encode categorical variables using Label Encoding.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    categorical_cols : list
        List of categorical column names

    Returns
    -------
    pd.DataFrame
        Dataframe with encoded categorical variables
    dict
        Dictionary of LabelEncoders for each column
    """
    encoders = {}
    df_encoded = df.copy()

    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le

    return df_encoded, encoders


def create_binary_target(df, target_col, threshold=0):
    """
    Create binary target variable from multi-class health problem indicator.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    threshold : int
        Threshold for binarization (default: 0)

    Returns
    -------
    pd.Series
        Binary target variable (0: no problems, 1: has problems)
    """
    return (df[target_col] > threshold).astype(int)


def scale_features(X_train, X_test=None):
    """
    Standardize features using StandardScaler.

    Parameters
    ----------
    X_train : array-like
        Training features
    X_test : array-like, optional
        Test features

    Returns
    -------
    array-like
        Scaled training features
    array-like (optional)
        Scaled test features if provided
    StandardScaler
        Fitted scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler

    return X_train_scaled, scaler


def get_feature_names(df, exclude_cols=['health_problems', 'target']):
    """
    Get list of feature names excluding target and ID columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    exclude_cols : list
        Columns to exclude

    Returns
    -------
    list
        List of feature column names
    """
    return [col for col in df.columns if col not in exclude_cols]


if __name__ == "__main__":
    # Example usage
    print("Data preprocessing utilities loaded successfully.")
    print("Functions available:")
    print("  - load_data()")
    print("  - handle_missing_values()")
    print("  - encode_categorical_variables()")
    print("  - create_binary_target()")
    print("  - scale_features()")
    print("  - get_feature_names()")
