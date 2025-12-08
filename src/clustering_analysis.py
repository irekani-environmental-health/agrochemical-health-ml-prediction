"""
K-Means Clustering Analysis Module
==================================

This module implements K-means clustering for agricultural/environmental
population segmentation in the Irekani dataset.

Author: Luis José Yudico-Anaya
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def select_agricultural_features(df):
    """
    Select 8 agricultural/environmental features for clustering.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    list
        List of selected feature names
    """
    features = [
        'tiempo_dedicado_campo',  # Years of agricultural experience
        'hectareas_trabajadas',   # Farm size (hectares)
        'fertilidad_suelo',       # Soil fertility perception
        'calidad_agua_regar',     # Water quality for irrigation
        'cantidad_agua',          # Water availability
        'temperatura',            # Temperature perception
        'arboles',                # Tree density
        'zonas_bosques'           # Forest/woodland areas
    ]
    return features


def prepare_clustering_data(df, features, scale=True):
    """
    Prepare data for clustering analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    features : list
        List of feature names to use
    scale : bool
        Whether to standardize features (default: True)

    Returns
    -------
    array-like
        Prepared feature matrix
    StandardScaler or None
        Fitted scaler if scale=True, else None
    """
    X = df[features].copy()

    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        return X.values, None


def find_optimal_k(X, k_range=range(2, 11), random_state=42):
    """
    Find optimal number of clusters using elbow method and silhouette score.

    Parameters
    ----------
    X : array-like
        Feature matrix
    k_range : range
        Range of k values to test
    random_state : int
        Random seed

    Returns
    -------
    dict
        Dictionary with inertia and silhouette scores for each k
    """
    results = {
        'k': [],
        'inertia': [],
        'silhouette': [],
        'davies_bouldin': [],
        'calinski_harabasz': []
    }

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)

        results['k'].append(k)
        results['inertia'].append(kmeans.inertia_)
        results['silhouette'].append(silhouette_score(X, labels))
        results['davies_bouldin'].append(davies_bouldin_score(X, labels))
        results['calinski_harabasz'].append(calinski_harabasz_score(X, labels))

    return results


def perform_kmeans_clustering(X, n_clusters=4, random_state=42):
    """
    Perform K-means clustering with specified number of clusters.

    Parameters
    ----------
    X : array-like
        Feature matrix
    n_clusters : int
        Number of clusters (default: 4)
    random_state : int
        Random seed

    Returns
    -------
    KMeans
        Fitted KMeans model
    array-like
        Cluster labels for each sample
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    print(f"\nK-Means Clustering (k={n_clusters})")
    print("=" * 60)
    print(f"Silhouette Score: {silhouette_score(X, labels):.3f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_score(X, labels):.3f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz_score(X, labels):.2f}")
    print("\nCluster Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"  Cluster {cluster}: {count} samples ({count/len(labels)*100:.1f}%)")
    print("=" * 60)

    return kmeans, labels


def plot_elbow_curve(results, figsize=(12, 5)):
    """
    Plot elbow curve and silhouette scores.

    Parameters
    ----------
    results : dict
        Results from find_optimal_k()
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        Elbow plot figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Elbow plot
    ax1.plot(results['k'], results['inertia'], marker='o', linewidth=2)
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia', fontsize=12)
    ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)

    # Silhouette plot
    ax2.plot(results['k'], results['silhouette'], marker='o', color='green', linewidth=2)
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cluster_pca(X, labels, figsize=(10, 8)):
    """
    Plot clusters in 2D using PCA projection.

    Parameters
    ----------
    X : array-like
        Feature matrix
    labels : array-like
        Cluster labels
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
        PCA projection figure
    """
    # PCA to 2 components
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=figsize)
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                         cmap='viridis', s=50, alpha=0.6, edgecolors='k')
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    plt.title('K-Means Clusters (PCA Projection)', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return plt.gcf()


def analyze_cluster_profiles(df, labels, features):
    """
    Analyze characteristics of each cluster.

    Parameters
    ----------
    df : pd.DataFrame
        Original dataframe
    labels : array-like
        Cluster labels
    features : list
        List of feature names used for clustering

    Returns
    -------
    pd.DataFrame
        Summary statistics for each cluster
    """
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = labels

    # Calculate mean values per cluster
    cluster_profiles = df_with_clusters.groupby('cluster')[features].mean()

    # Calculate overall mean for comparison
    overall_mean = df[features].mean()

    # Calculate percentage difference from overall mean
    profile_comparison = pd.DataFrame()
    for cluster in cluster_profiles.index:
        profile_comparison[f'Cluster {cluster}'] = (
            (cluster_profiles.loc[cluster] - overall_mean) / overall_mean * 100
        ).round(1)

    print("\nCluster Profiles (% difference from overall mean):")
    print("=" * 80)
    print(profile_comparison)
    print("=" * 80)

    return cluster_profiles


if __name__ == "__main__":
    print("Clustering analysis utilities loaded successfully.")
    print("Functions available:")
    print("  - select_agricultural_features()")
    print("  - prepare_clustering_data()")
    print("  - find_optimal_k()")
    print("  - perform_kmeans_clustering()")
    print("  - plot_elbow_curve()")
    print("  - plot_cluster_pca()")
    print("  - analyze_cluster_profiles()")
