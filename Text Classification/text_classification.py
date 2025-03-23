import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif, f_classif
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, learning_curve
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import f1_score
from imblearn.over_sampling import ADASYN, RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek
from collections import Counter

def load_sparse_data(filename):
    """
    Load data in sparse format from a file where each line contains:
    class_label feature_index1 feature_index2 ...
    """
    data = []
    with open(filename, 'r') as file:
        for line in file:
            data.append(list(map(int, line.strip().split())))
    return data

def convert_to_sparse_matrix(data, is_training=True):
    """Convert loaded data into a sparse matrix format suitable for ML algorithms"""
    row, col, values = [], [], []
    labels = []
    num_features = 0

    for i, line_data in enumerate(data):
        # First value is class label
        label = line_data[0] 
        labels.append(label)
        
        # Remaining values are feature indices
        for col_idx in line_data[1:]:
            row.append(i)
            col.append(col_idx)
            values.append(1)  # Binary features
            num_features = max(num_features, col_idx + 1)

    # Only return labels for training data
    Y = labels if is_training else None
    X = csr_matrix((values, (row, col)), shape=(len(data), num_features))
    
    return Y, X

def visualize_class_distribution(labels):
    """Create a bar chart showing the distribution of classes in the dataset"""
    class_counts = dict(Counter(labels))
    
    # Extracting class labels and counts
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Class Label')
    plt.ylabel('Number of Instances')
    plt.title('Class Distribution in Training Data')
    plt.xticks(classes)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('class_distribution.png')
    plt.close()

def handle_imbalanced_data(X_train, Y_train, method='ADASYN'):
    """
    Apply resampling techniques to handle class imbalance using k-fold cross-validation
    to ensure robust resampling.
    """
    print(f"Handling imbalanced data using {method}...")
    
    # Initialize k-fold splitting
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    Y_train = np.array(Y_train)
    
    X_resampled = []
    Y_resampled = []
    
    # Apply resampling technique to each fold
    for train_idx, val_idx in skf.split(X_train, Y_train):
        train_idx = train_idx.astype(int)
        val_idx = val_idx.astype(int)
        
        X_fold, Y_fold = X_train[train_idx], Y_train[train_idx]
        
        # Choose resampling method
        if method == 'ADASYN':
            resampler = ADASYN(sampling_strategy='minority', n_neighbors=5, random_state=42)
        elif method == 'RandomOverSampler':
            resampler = RandomOverSampler(random_state=42)
        elif method == 'ClusterCentroids':
            resampler = ClusterCentroids(random_state=42)
        elif method == 'SMOTETomek':
            resampler = SMOTETomek(random_state=42)
        else:
            raise ValueError(f"Unsupported resampling method: {method}")
            
        # Apply resampling
        X_fold_resampled, Y_fold_resampled = resampler.fit_resample(X_fold, Y_fold)
        
        # Store resampled fold
        X_resampled.append(csr_matrix(X_fold_resampled))
        Y_resampled.append(Y_fold_resampled)
    
    # Combine resampled folds
    X_resampled_combined = vstack(X_resampled)
    Y_resampled_combined = np.concatenate(Y_resampled)
    
    # Display class distribution after resampling
    print("Class distribution after resampling:")
    print(sorted(Counter(Y_resampled_combined).items()))
    
    return X_resampled_combined, Y_resampled_combined

def reduce_dimensions(X_train, X_test, n_components=20):
    """Apply TruncatedSVD for dimensionality reduction"""
    print(f"Reducing dimensions using TruncatedSVD with {n_components} components...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_reduced = svd.fit_transform(X_train)
    X_test_reduced = svd.transform(X_test)
    
    # Report explained variance
    explained_variance = svd.explained_variance_ratio_.sum()
    print(f"Explained variance with {n_components} components: {explained_variance:.2%}")
    
    return X_train_reduced, X_test_reduced

def train_and_select_classifier(X_train, Y_train):
    """Train and select the best classifier using cross-validation"""
    print("Training and selecting classifier...")
    
    # Setup cross-validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    
    # Define classifier with hyperparameter grid
    classifier = BernoulliNB()
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        classifier, 
        param_grid, 
        cv=cv, 
        scoring='f1_macro',
        verbose=1
    )
    
    grid_search.fit(X_train, Y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation F1 score: {best_score:.4f}")
    
    return best_model

def visualize_learning_curve(model, X, y, title="Learning Curve"):
    """Plot learning curve to visualize model performance as training data increases"""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, train_sizes=train_sizes, scoring='f1_macro', n_jobs=-1
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("F1 Score")
    plt.grid(True)
    
    plt.fill_between(
        train_sizes, 
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std, 
        alpha=0.1, color="blue"
    )
    plt.fill_between(
        train_sizes, 
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std, 
        alpha=0.1, color="orange"
    )
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="blue", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="orange", label="Cross-validation score")
    
    plt.legend(loc="best")
    plt.savefig('learning_curve.png')
    plt.close()

def main():
    # Load and prepare data
    print("Loading data...")
    train_data = load_sparse_data('train_data.txt')
    test_data = load_sparse_data('test_data.txt')
    
    Y_train, X_train = convert_to_sparse_matrix(train_data, is_training=True)
    _, X_test = convert_to_sparse_matrix(test_data, is_training=False)
    
    # Visualize original class distribution
    visualize_class_distribution(Y_train)
    
    # Remove constant features
    print("Removing constant features...")
    constant_filter = VarianceThreshold()
    X_train = constant_filter.fit_transform(X_train)
    X_test = constant_filter.transform(X_test)
    
    # Handle imbalanced data
    X_train_resampled, Y_train_resampled = handle_imbalanced_data(X_train, Y_train, method='ADASYN')
    
    # Reduce dimensions
    X_train_reduced, X_test_reduced = reduce_dimensions(X_train_resampled, X_test, n_components=20)
    
    # Train and select best model
    best_model = train_and_select_classifier(X_train_reduced, Y_train_resampled)
    
    # Visualize model learning curve
    visualize_learning_curve(best_model, X_train_reduced, Y_train_resampled)
    
    # Make predictions on test data
    print("Making predictions on test data...")
    test_predictions = best_model.predict(X_test_reduced)
    
    # Count predictions by class
    num_zeros = np.sum(test_predictions == 0)
    num_ones = np.sum(test_predictions == 1)
    print(f"Class 0 predictions: {num_zeros}")
    print(f"Class 1 predictions: {num_ones}")
    
    # Save predictions to file
    output_file = "binary_text_predictions.txt"
    np.savetxt(output_file, test_predictions, fmt='%d')
    print(f"Predictions saved to {output_file}")
    
if __name__ == "__main__":
    main()