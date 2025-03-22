import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

class CustomKNNClassifier:
    """
    A custom implementation of the K-Nearest Neighbors algorithm for classification
    """
    
    def __init__(self, n_neighbors=5):
        """Initialize the classifier with the number of neighbors to consider"""
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Store the training data and labels"""
        self.X_train = X
        self.y_train = y
        return self
    
    def _euclidean_distance(self, vector1, vector2):
        """Calculate Euclidean distance between two vectors"""
        return np.sqrt(np.sum((vector1 - vector2) ** 2))
    
    def predict(self, X):
        """Predict class labels for samples in X"""
        predictions = []
        
        for sample in X:
            # Calculate distances from current sample to all training samples
            distances = [self._euclidean_distance(sample, x_train) for x_train in self.X_train]
            
            # Get indices of k nearest neighbors
            nearest_indices = np.argsort(distances)[:self.n_neighbors]
            
            # Get labels of the nearest neighbors
            nearest_labels = [self.y_train[i] for i in nearest_indices]
            
            # Predict the most common class among the nearest neighbors
            most_common = np.bincount(nearest_labels).argmax()
            predictions.append(most_common)
            
        return np.array(predictions)


def perform_cross_validation(X, y, classifier, n_folds=5):
    """
    Perform k-fold cross-validation to evaluate model performance
    
    Parameters:
    X : feature matrix
    y : target vector
    classifier : classifier object with fit and predict methods
    n_folds : number of folds for cross-validation
    
    Returns:
    mean_accuracy : average accuracy across all folds
    """
    # Calculate fold size
    fold_size = len(X) // n_folds
    accuracies = []
    
    # Perform cross-validation
    for i in range(n_folds):
        # Define validation fold indices
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size
        
        # Split data into training and validation sets
        X_val = X[start_idx:end_idx]
        y_val = y[start_idx:end_idx]
        
        # Training data excludes validation fold
        X_train = np.concatenate([X[:start_idx], X[end_idx:]])
        y_train = np.concatenate([y[:start_idx], y[end_idx:]])
        
        # Train model and make predictions
        model = classifier(n_neighbors=classifier.n_neighbors)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        # Calculate accuracy for current fold
        accuracy = np.mean(y_pred == y_val)
        accuracies.append(accuracy)
    
    # Return mean accuracy across all folds
    return np.mean(accuracies)


def optimize_hyperparameters(X, y, estimator_class, param_grid, n_folds=5):
    """
    Find the best hyperparameters using cross-validation
    
    Parameters:
    X : feature matrix
    y : target vector
    estimator_class : classifier class
    param_grid : dictionary with parameter names as keys and lists of values to try
    n_folds : number of folds for cross-validation
    
    Returns:
    best_params : dictionary with best parameter values
    best_score : best cross-validation score
    """
    best_score = 0
    best_params = {}
    
    # For this simplified version, we're only optimizing n_neighbors
    for n_neighbors in param_grid['n_neighbors']:
        # Create model with current parameters
        model = estimator_class(n_neighbors=n_neighbors)
        
        # Evaluate model using cross-validation
        score = perform_cross_validation(X, y, estimator_class, n_folds)
        
        # Update best parameters if current score is better
        if score > best_score:
            best_score = score
            best_params = {'n_neighbors': n_neighbors}
    
    return best_params, best_score


def main():
    """
    Main function to run the student performance prediction pipeline
    """
    # Set figure style for plots
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Load datasets
    print("Loading datasets...")
    try:
        train_df = pd.read_csv('training.csv')
        test_df = pd.read_csv('test.csv')
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
    except FileNotFoundError:
        print("Error: Dataset files not found. Please ensure 'training.csv' and 'test.csv' are in the current directory.")
        return
    
    # Check for missing values
    print("\nChecking for missing values...")
    missing_train = train_df.isnull().sum()
    missing_test = test_df.isnull().sum()
    
    if missing_train.sum() > 0:
        print("Warning: Training data contains missing values")
        print(missing_train[missing_train > 0])
    else:
        print("Training data is complete (no missing values)")
    
    if missing_test.sum() > 0:
        print("Warning: Test data contains missing values")
        print(missing_test[missing_test > 0])
    else:
        print("Test data is complete (no missing values)")
    
    # Prepare data
    print("\nPreparing data...")
    X_train = train_df.drop('Target', axis=1)
    y_train = train_df['Target'].values
    X_test = test_df
    
    # Feature names for later reference
    feature_names = X_train.columns
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Target', data=train_df)
    plt.title('Class Distribution', fontsize=14)
    plt.xlabel('Performance Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Class distribution plot saved as 'class_distribution.png'")
    
    # Generate correlation heatmap
    plt.figure(figsize=(20, 16))
    correlation_matrix = X_train.corr()
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                          fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=18)
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Correlation matrix saved as 'correlation_matrix.png'")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply PCA for dimensionality reduction
    print("\nPerforming PCA analysis...")
    # First, explore explained variance
    pca_explorer = PCA()
    pca_explorer.fit(X_train_scaled)
    explained_variance_ratio = pca_explorer.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    # Plot explained variance
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             marker='o', linestyle='-', markersize=5)
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Explained Variance')
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('Cumulative Explained Variance', fontsize=12)
    plt.title('PCA Explained Variance Analysis', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('pca_variance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("PCA variance analysis plot saved as 'pca_variance_analysis.png'")
    
    # Determine optimal number of components (95% variance explained)
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"Selecting {n_components} components for 95% variance retention")
    
    # Apply PCA with selected number of components
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # Model training and hyperparameter optimization
    print("\nOptimizing KNN hyperparameters...")
    param_grid = {'n_neighbors': range(1, 31)}
    best_params, best_score = optimize_hyperparameters(
        X_train_pca, y_train, CustomKNNClassifier, param_grid, n_folds=10)
    
    optimal_k = best_params['n_neighbors']
    print(f"Best number of neighbors (k): {optimal_k}")
    print(f"Cross-validation accuracy: {best_score:.4f}")
    
    # Train final model with best parameters
    print("\nTraining final model...")
    final_model = CustomKNNClassifier(n_neighbors=optimal_k)
    final_model.fit(X_train_pca, y_train)
    
    # Make predictions on test set
    print("Making predictions on test data...")
    test_predictions = final_model.predict(X_test_pca)
    
    # Display prediction statistics
    unique_classes, class_counts = np.unique(test_predictions, return_counts=True)
    print("\nPrediction class distribution:")
    for class_label, count in zip(unique_classes, class_counts):
        print(f"Class {class_label}: {count} instances ({count/len(test_predictions)*100:.2f}%)")
    
    # Save predictions to file
    output_file = "student_performance_predictions.txt"
    with open(output_file, "w") as f:
        for prediction in test_predictions:
            f.write(f"{prediction}\n")
    print(f"\nPredictions saved to '{output_file}'")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()