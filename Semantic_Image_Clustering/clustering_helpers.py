"""
Helper functions for semantic image clustering workflows.

Provides utilities for:
- Loading and preprocessing images
- Evaluating KMeans for different k values
- Computing clustering quality metrics
- Visualization of results
"""

from PIL import Image
import numpy as np
import os
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, adjusted_mutual_info_score
import matplotlib.pyplot as plt


def load_images_as_vectors(folder, target_size=(64, 64), normalize=True):
    """Load all JPG images from folder, resize, and flatten into vectors.
    
    Args:
        folder (str): Path to folder containing images
        target_size (tuple): Target size for resize (default 64x64)
        normalize (bool): Whether to normalize pixel values to [0, 1]
    
    Returns:
        tuple: (image_vectors, filenames)
            - image_vectors (np.ndarray): Array of flattened image vectors (N, features)
            - filenames (list): Sorted list of filenames
    """
    filenames = sorted([f for f in os.listdir(folder) if f.lower().endswith('.jpg')])
    vectors = []
    
    for fname in filenames:
        img = Image.open(os.path.join(folder, fname)).convert('RGB')
        img = img.resize(target_size)
        arr = np.array(img)
        if normalize:
            arr = arr / 255.0
        vectors.append(arr.flatten())
    
    return np.array(vectors), filenames


def extract_true_labels(filenames):
    """Extract true class labels from filenames.
    
    Assumes filename format: class017_003.jpg -> label 17
    
    Args:
        filenames (list): List of filenames
    
    Returns:
        np.ndarray: Array of true class labels
    """
    labels = np.array([int(f.split('_')[0].replace('class', '')) for f in filenames])
    return labels


def evaluate_k_range(X, k_range, random_state=42):
    """Evaluate KMeans clustering for a range of k values.
    
    Computes silhouette scores and inertias for each k value, useful for 
    the elbow method and silhouette analysis.
    
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        k_range (range or list): Range of k values to test
        random_state (int): Random seed for reproducibility (default 42)
    
    Returns:
        dict: Dictionary containing:
            - 'silhouette_scores': list of silhouette scores
            - 'inertias': list of inertia values (WCSS)
            - 'k_range': list of k values tested
    """
    sil_scores = []
    inertias = []
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil_scores.append(silhouette_score(X, labels))
        inertias.append(km.inertia_)
    
    return {
        'silhouette_scores': sil_scores,
        'inertias': inertias,
        'k_range': list(k_range)
    }


def plot_evaluation_metrics(eval_results, figsize=(12, 5)):
    """Plot elbow method and silhouette score curves side-by-side.
    
    Args:
        eval_results (dict): Output from evaluate_k_range()
        figsize (tuple): Figure size (default 12x5)
    """
    k_range = eval_results['k_range']
    inertias = eval_results['inertias']
    sil_scores = eval_results['silhouette_scores']
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Elbow plot
    axes[0].plot(k_range, inertias, 'b-o', linewidth=2.5, markersize=10)
    axes[0].set_xlabel('K (Number of Clusters)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Inertia (WCSS)', fontsize=12, fontweight='bold')
    axes[0].set_title('Elbow Method for Optimal K', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(k_range)
    
    # Silhouette plot
    axes[1].plot(k_range, sil_scores, 'ro-', linewidth=2.5, markersize=10)
    axes[1].set_xlabel('K (Number of Clusters)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Silhouette Score vs K', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(k_range)
    
    plt.tight_layout()
    plt.show()


def perform_kmeans_clustering(X, n_clusters, random_state=42):
    """Fit KMeans and return model and labels.
    
    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features)
        n_clusters (int): Number of clusters
        random_state (int): Random seed (default 42)
    
    Returns:
        tuple: (fitted_model, predicted_labels)
            - fitted_model: Trained KMeans model
            - predicted_labels: Cluster assignments for each sample
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    return km, labels


def compute_clustering_metrics(true_labels, predicted_labels, verbose=True):
    """Compute Adjusted Rand Index (ARI) and Adjusted Mutual Information (AMI).
    
    Both metrics measure agreement between predicted and true labels:
    - ARI: Measures structural similarity (range: -1 to 1)
    - AMI: Measures information overlap (range: 0 to 1)
    
    Higher values indicate better clustering quality.
    
    Args:
        true_labels (np.ndarray): Ground truth cluster labels
        predicted_labels (np.ndarray): Predicted cluster labels
        verbose (bool): Whether to print results (default True)
    
    Returns:
        dict: Dictionary with keys 'ari' and 'ami'
    """
    ari = adjusted_rand_score(true_labels, predicted_labels)
    ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    
    if verbose:
        print(f"ARI: {ari:.3f}  |  AMI: {ami:.3f}")
    
    return {'ari': ari, 'ami': ami}


def plot_clusters_scatter(coords, predicted_labels, method_name='', figsize=(8, 6)):
    """Plot 2D cluster visualization with scatter plot.
    
    Args:
        coords (np.ndarray): 2D coordinates of shape (n_samples, 2)
        predicted_labels (np.ndarray): Cluster assignment for each sample
        method_name (str): Name of dimensionality reduction method (e.g., 't-SNE', 'PCA')
        figsize (tuple): Figure size (default 8x6)
    """
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        coords[:, 0], 
        coords[:, 1], 
        c=predicted_labels, 
        cmap='tab20', 
        s=50, 
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    plt.xlabel('Component 1', fontsize=11, fontweight='bold')
    plt.ylabel('Component 2', fontsize=11, fontweight='bold')
    title = f'Cluster Visualization ({method_name})' if method_name else 'Cluster Visualization'
    plt.title(title, fontsize=12, fontweight='bold')
    cbar = plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.show()
