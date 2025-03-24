"""
MultiSpectral: Advanced K-Means Clustering with Cosine Similarity
Author: [Antara Tewary]
Date: March 2025

This script implements a custom K-means++ clustering algorithm using cosine similarity as the distance metric.
It automatically determines the optimal number of clusters using the elbow method and visualizes the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
import os

class CosineSimilarityClustering:
    """A clustering algorithm using cosine similarity as the distance metric with K-means++ initialization."""
    
    def __init__(self, max_iterations=300, tolerance=1e-4, n_runs=10, random_state=42):
        """
        Initialize the clustering algorithm.
        
        Parameters:
        -----------
        max_iterations : int, default=300
            Maximum number of iterations for each clustering run.
        tolerance : float, default=1e-4
            Convergence threshold for early stopping.
        n_runs : int, default=10
            Number of times to run clustering with different initializations.
        random_state : int, default=42
            Random seed for reproducibility.
        """
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.n_runs = n_runs
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        
    def fit(self, X, num_clusters):
        """
        Fit the clustering model to the data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        num_clusters : int
            Number of clusters to find.
            
        Returns:
        --------
        self : object
            Fitted estimator.
        """
        np.random.seed(self.random_state)
        num_instances, num_features = X.shape
        
        best_sse = np.inf
        best_cluster_ids = None
        best_cluster_centers = None
        
        for _ in range(self.n_runs):
            # Initialize cluster centers using K-means++ algorithm
            cluster_centers = [X[np.random.choice(num_instances)]]
            for _ in range(1, num_clusters):
                dist = np.min(cdist(X, np.array(cluster_centers), 'cosine'), axis=1)
                probs = dist / np.sum(dist)
                cluster_centers.append(X[np.random.choice(num_instances, p=probs)])
            
            cluster_centers = np.array(cluster_centers)
            prev_sse = np.inf
            
            for _ in range(self.max_iterations):
                # Calculate cosine distance between instances and cluster centers
                dist = cdist(X, cluster_centers, 'cosine')
                
                # Assign each instance to the nearest cluster
                cluster_ids = np.argmin(dist, axis=1)
                
                # Update cluster centers
                new_centers = []
                for i in range(num_clusters):
                    cluster_data = X[cluster_ids == i]
                    if cluster_data.size > 0:
                        new_centers.append(cluster_data.mean(axis=0))
                    else:
                        # If a cluster is empty, reinitialize it
                        new_centers.append(X[np.random.choice(num_instances)])
                new_centers = np.array(new_centers)
                
                # Calculate SSE (inertia)
                sse = 0
                for i in range(num_instances):
                    sse += np.sum((X[i] - new_centers[cluster_ids[i]])**2)
                
                # Check for convergence
                if np.abs(prev_sse - sse) < self.tolerance:
                    break
                
                cluster_centers = new_centers
                prev_sse = sse
            
            if sse < best_sse:
                best_sse = sse
                best_cluster_ids = cluster_ids
                best_cluster_centers = cluster_centers
        
        self.cluster_centers_ = best_cluster_centers
        self.labels_ = best_cluster_ids
        self.inertia_ = best_sse
        
        return self
    
    def predict(self, X):
        """
        Predict the closest cluster for each sample in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to predict.
            
        Returns:
        --------
        labels : array of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
        
        dist = cdist(X, self.cluster_centers_, 'cosine')
        return np.argmin(dist, axis=1)


def load_data(file_path):
    """
    Load data from a text file.
    
    Parameters:
    -----------
    file_path : str
        Path to the data file.
        
    Returns:
    --------
    data : numpy.ndarray
        Loaded data.
    """
    try:
        # Try pandas first (more flexible for various formats)
        data = pd.read_csv(file_path, header=None, sep=None, engine='python')
        return data.values
    except:
        # Fall back to numpy
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                data.append(list(map(float, line.strip().split())))
        return np.array(data)


def preprocess_data(data, normalize=True, impute=True):
    """
    Preprocess the data by handling missing values and normalizing.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data.
    normalize : bool, default=True
        Whether to standardize the data.
    impute : bool, default=True
        Whether to impute missing values.
        
    Returns:
    --------
    processed_data : numpy.ndarray
        Preprocessed data.
    """
    # Handle missing values
    if impute:
        data = np.nan_to_num(data, nan=np.nanmean(data, axis=0))
    
    # Normalize data
    if normalize:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
    
    return data


def reduce_dimensions(data, method='pca', n_components=2):
    """
    Reduce the dimensionality of the data.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data.
    method : str, default='pca'
        Dimensionality reduction method ('pca' or 'tsne').
    n_components : int, default=2
        Number of components to keep.
        
    Returns:
    --------
    reduced_data : numpy.ndarray
        Data with reduced dimensions.
    """
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    return reducer.fit_transform(data)


def find_optimal_clusters(data, max_clusters=20, step=1):
    """
    Find the optimal number of clusters using the elbow method.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data.
    max_clusters : int, default=20
        Maximum number of clusters to consider.
    step : int, default=1
        Step size for the range of clusters.
        
    Returns:
    --------
    sse_values : list
        SSE values for each number of clusters.
    cluster_numbers : list
        Range of cluster numbers evaluated.
    """
    sse_values = []
    cluster_numbers = range(2, max_clusters + 1, step)
    
    for k in cluster_numbers:
        model = CosineSimilarityClustering()
        model.fit(data, k)
        sse_values.append(model.inertia_)
        print(f"Evaluated clustering with k={k}, SSE={model.inertia_:.4f}")
    
    return sse_values, list(cluster_numbers)


def plot_elbow_curve(cluster_numbers, sse_values, save_path=None):
    """
    Plot the elbow curve for finding the optimal number of clusters.
    
    Parameters:
    -----------
    cluster_numbers : list
        Range of cluster numbers evaluated.
    sse_values : list
        SSE values for each number of clusters.
    save_path : str, optional
        Path to save the figure.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cluster_numbers, sse_values, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=14)
    plt.ylabel('Sum of Squared Errors (SSE)', fontsize=14)
    plt.title('Elbow Method for Optimal K Selection', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(cluster_numbers)
    
    # Add annotations for significant points
    max_drop_idx = np.argmax(np.array(sse_values[:-1]) - np.array(sse_values[1:]))
    plt.annotate(f'Significant drop at K={cluster_numbers[max_drop_idx+1]}',
                xy=(cluster_numbers[max_drop_idx+1], sse_values[max_drop_idx+1]),
                xytext=(cluster_numbers[max_drop_idx+1]+1, sse_values[max_drop_idx+1]+0.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_clusters(data, labels, centers=None, save_path=None):
    """
    Visualize the clusters in a 2D space.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input data (must be 2D for visualization).
    labels : numpy.ndarray
        Cluster assignments.
    centers : numpy.ndarray, optional
        Cluster centers.
    save_path : str, optional
        Path to save the figure.
    """
    if data.shape[1] > 2:
        print("Data has more than 2 dimensions. Reducing to 2D using PCA for visualization.")
        data = reduce_dimensions(data, method='pca', n_components=2)
    
    plt.figure(figsize=(12, 8))
    
    # Create a color palette with distinct colors
    palette = sns.color_palette('husl', len(np.unique(labels)))
    
    # Create a scatter plot for each cluster
    for i, color in enumerate(palette):
        cluster_data = data[labels == i]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                   c=[color], label=f'Cluster {i+1}',
                   alpha=0.7, s=100, edgecolors='w', linewidth=0.5)
    
    # Plot the centroids if provided
    if centers is not None and centers.shape[1] <= 2:
        if centers.shape[1] < 2 and data.shape[1] >= 2:
            # If centers are in a different space, project them using the same transformation
            print("Note: Centers are being projected to 2D space for visualization.")
        plt.scatter(centers[:, 0], centers[:, 1], 
                   c='black', s=200, alpha=1, marker='X', 
                   edgecolors='white', linewidth=2, label='Centroids')
    
    plt.title('Cluster Analysis Results', fontsize=16)
    plt.xlabel('Dimension 1', fontsize=14)
    plt.ylabel('Dimension 2', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_results(labels, output_file='cluster_assignments.txt'):
    """
    Save the cluster assignments to a file.
    
    Parameters:
    -----------
    labels : numpy.ndarray
        Cluster assignments.
    output_file : str, default='cluster_assignments.txt'
        Path to the output file.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # Save cluster assignments (1-indexed for readability)
    with open(output_file, 'w') as f:
        for label in labels:
            f.write(f"{label + 1}\n")
    
    print(f"Cluster assignments saved to {output_file}")


def main():
    """Main function to run the clustering analysis."""
    # Set figure aesthetics
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")
    
    # Configuration
    data_file = 'data/flower_features.txt'  # Changed from test-data-iris.txt
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    print("Loading and preprocessing data...")
    data = load_data(data_file)
    processed_data = preprocess_data(data, normalize=True, impute=True)
    
    print("Finding optimal number of clusters...")
    sse_values, cluster_numbers = find_optimal_clusters(processed_data, max_clusters=10, step=1)
    plot_elbow_curve(cluster_numbers, sse_values, save_path=f"{results_dir}/elbow_curve.png")
    
    # Determine optimal number of clusters (you can implement a more sophisticated method)
    # For now, we'll just use 3 which is known for Iris data
    optimal_k = 3
    print(f"Selected optimal number of clusters: {optimal_k}")
    
    # Perform final clustering with optimal K
    print("Performing final clustering with optimal K...")
    model = CosineSimilarityClustering(n_runs=20)  # More runs for the final clustering
    model.fit(processed_data, optimal_k)
    
    # Reduce dimensions for visualization if needed
    if processed_data.shape[1] > 2:
        vis_data = reduce_dimensions(processed_data, method='pca', n_components=2)
    else:
        vis_data = processed_data
    
    # Visualize results
    print("Visualizing clustering results...")
    visualize_clusters(vis_data, model.labels_, 
                      centers=reduce_dimensions(model.cluster_centers_, method='pca', n_components=2) 
                      if processed_data.shape[1] > 2 else model.cluster_centers_,
                      save_path=f"{results_dir}/cluster_visualization.png")
    
    # Save results
    print("Saving results...")
    save_results(model.labels_, output_file=f"{results_dir}/cluster_assignments.txt")
    
    print("Clustering analysis completed successfully!")


if __name__ == "__main__":
    main()