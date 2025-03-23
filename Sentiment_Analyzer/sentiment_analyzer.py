#!/usr/bin/env python3
"""
SentimentScope: Text Sentiment Analysis Using KNN
Author: Your Name
Date: March 2025

This script implements a sentiment analysis system using a custom K-Nearest Neighbors
algorithm with cosine similarity on text data.
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Text Sentiment Analysis using KNN')
    parser.add_argument('--train', required=True, help='Path to training data CSV')
    parser.add_argument('--test', required=True, help='Path to test data CSV')
    parser.add_argument('--output', default='sentiment_predictions.txt', 
                        help='Output file for predictions')
    parser.add_argument('--k', type=int, default=None, 
                        help='K value for KNN (if not provided, will be tuned)')
    parser.add_argument('--folds', type=int, default=5, 
                        help='Number of folds for cross-validation')
    parser.add_argument('--max-k', type=int, default=20, 
                        help='Maximum K value to try during tuning')
    parser.add_argument('--components', type=int, default=100, 
                        help='Number of components for dimensionality reduction')
    return parser.parse_args()


def load_data(train_path, test_path):
    """Load and prepare training and test datasets."""
    # Load training data
    train_data = pd.read_csv(train_path, names=["Sentiment", "Reviews"])
    
    # Load test data
    test_data = pd.read_csv(test_path, names=["Reviews"])
    
    # Extract features and target
    X_train = train_data.drop(['Sentiment'], axis=1)
    y_train = train_data['Sentiment']
    X_test = test_data
    
    return X_train, y_train, X_test


def plot_sentiment_distribution(sentiments):
    """Plot the distribution of sentiment labels."""
    sentiment_counts = sentiments.value_counts()
    
    plt.figure(figsize=(8, 6))
    plt.bar(sentiment_counts.index, sentiment_counts.values, color=['coral', 'skyblue'])
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution in Training Data')
    plt.xticks(sentiment_counts.index, labels=['Negative (-1)', 'Positive (+1)'])
    plt.tight_layout()
    plt.savefig('sentiment_distribution.png')
    plt.close()


def plot_word_frequencies(text_data, title, filename, n=20):
    """Plot the most common words in the text data."""
    # Tokenize the text and count word frequencies
    tokens = ' '.join(text_data).split()
    word_freq = Counter(tokens)
    
    # Get the top N common words
    common_words = word_freq.most_common(n)
    
    # Plot the common words
    common_words_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Frequency', y='Word', data=common_words_df)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def preprocess_text(X_train, X_test):
    """Apply text preprocessing to both training and test data."""
    # Download required NLTK resources
    for resource in ['stopwords', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'corpora/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Create a set of custom stopwords (excluding negation words)
    common_stopwords = set(stopwords.words('english'))
    negation_words = {"but", "not", "never", "no", "none", "neither", "no one", 
                     "nobody", "none", "nor", "nothing", "nowhere"}
    custom_stopwords = common_stopwords - negation_words
    
    # Define preprocessing functions
    def to_lowercase(text_series):
        return text_series.str.lower()
    
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def tokenize(text_series):
        return text_series.str.split()
    
    def remove_custom_stopwords(tokens):
        return [word for word in tokens if word not in custom_stopwords]
    
    def lemmatize_tokens(tokens):
        return [lemmatizer.lemmatize(word) for word in tokens]
    
    def remove_numbers(tokens):
        return [word for word in tokens if not word.isdigit()]
    
    # Apply preprocessing to both datasets
    for dataset in [X_train, X_test]:
        # Convert to lowercase
        dataset['Reviews'] = to_lowercase(dataset['Reviews'])
        
        # Remove punctuation
        dataset['Reviews'] = dataset['Reviews'].apply(remove_punctuation)
        
        # Tokenize
        dataset['Reviews'] = tokenize(dataset['Reviews'])
        
        # Remove stopwords
        dataset['Reviews'] = dataset['Reviews'].apply(remove_custom_stopwords)
        
        # Lemmatize
        dataset['Reviews'] = dataset['Reviews'].apply(lemmatize_tokens)
        
        # Remove numeric tokens
        dataset['Reviews'] = dataset['Reviews'].apply(remove_numbers)
        
        # Join tokens back into strings (for vectorization)
        dataset['Reviews'] = dataset['Reviews'].apply(lambda x: ' '.join(x))
    
    return X_train, X_test


def vectorize_text(X_train, X_test, n_components=100):
    """
    Transform text data into numerical features using TF-IDF vectorization
    and dimensionality reduction.
    """
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train['Reviews'])
    X_test_tfidf = tfidf_vectorizer.transform(X_test['Reviews'])
    
    # Normalize the vectors
    scaler = MaxAbsScaler()
    X_train_tfidf_scaled = scaler.fit_transform(X_train_tfidf)
    X_test_tfidf_scaled = scaler.transform(X_test_tfidf)
    
    # Dimensionality reduction
    svd = TruncatedSVD(n_components=n_components)
    X_train_svd = svd.fit_transform(X_train_tfidf_scaled)
    X_test_svd = svd.transform(X_test_tfidf_scaled)
    
    # Convert to arrays
    X_train_features = pd.DataFrame(X_train_svd).values
    X_test_features = pd.DataFrame(X_test_svd).values
    
    # Print variance explained
    variance_explained = svd.explained_variance_ratio_.sum()
    print(f"Variance explained by {n_components} components: {variance_explained:.2%}")
    
    return X_train_features, X_test_features


class CosineSimilarityKNN:
    """
    K-Nearest Neighbors classifier using cosine similarity for text sentiment analysis.
    """
    
    def __init__(self, k=5):
        """Initialize the KNN classifier with k neighbors."""
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """Fit the model with training data."""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X_test):
        """
        Predict the sentiment of test samples using the k nearest neighbors.
        
        Args:
            X_test: Test features
            
        Returns:
            List of predicted sentiment labels (-1 or 1)
        """
        predictions = []
        
        for test_sample in X_test:
            # Calculate cosine similarities between test sample and all training samples
            similarities = cosine_similarity([test_sample], self.X_train)[0]
            
            # Find indices of k nearest neighbors (highest similarities)
            k_neighbors_indices = np.argsort(similarities)[-self.k:]
            
            # Extract labels of k nearest neighbors
            neighbor_labels = [self.y_train.iloc[i] if isinstance(self.y_train, pd.Series) 
                              else self.y_train[i] for i in k_neighbors_indices]
            
            # Determine prediction by majority vote
            # Use np.sign to convert sum to -1 or 1 (or 0 if exactly balanced)
            vote_sum = np.sum(neighbor_labels)
            prediction = np.sign(vote_sum) if vote_sum != 0 else 1  # Default to positive if tied
            
            predictions.append(prediction)
        
        return predictions


def cross_validate(X, y, k, n_folds=5):
    """
    Perform cross-validation to evaluate model performance.
    
    Args:
        X: Feature matrix
        y: Target labels
        k: Number of neighbors for KNN
        n_folds: Number of folds for cross-validation
        
    Returns:
        Mean accuracy across all folds
    """
    # Convert pandas Series to numpy array if needed
    if isinstance(y, pd.Series):
        y = y.values
    
    # Calculate fold size
    fold_size = len(X) // n_folds
    
    # Store accuracy for each fold
    fold_accuracies = []
    
    # Loop through each fold
    for i in range(n_folds):
        # Define fold indices
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size
        
        # Split data into training and validation sets
        X_val = X[start_idx:end_idx]
        y_val = y[start_idx:end_idx]
        
        X_train = np.concatenate([X[:start_idx], X[end_idx:]])
        y_train = np.concatenate([y[:start_idx], y[end_idx:]])
        
        # Create and train KNN model
        knn = CosineSimilarityKNN(k=k)
        knn.fit(X_train, y_train)
        
        # Make predictions on validation set
        y_pred = knn.predict(X_val)
        
        # Calculate accuracy
        accuracy = np.mean(np.array(y_pred) == y_val)
        fold_accuracies.append(accuracy)
    
    # Return mean accuracy across all folds
    return np.mean(fold_accuracies)


def tune_hyperparameters(X, y, max_k=20, n_folds=5):
    """
    Find the optimal value of k for KNN using cross-validation.
    
    Args:
        X: Feature matrix
        y: Target labels
        max_k: Maximum value of k to try
        n_folds: Number of folds for cross-validation
        
    Returns:
        Tuple of (best_k, best_accuracy)
    """
    print("Tuning KNN hyperparameter...")
    best_k = None
    best_accuracy = 0.0
    
    # Try odd values of k from 3 to max_k
    for k in range(3, max_k + 1, 2):
        # Use cross-validation to evaluate this k value
        mean_accuracy = cross_validate(X, y, k, n_folds)
        print(f"  k={k}, accuracy={mean_accuracy:.4f}")
        
        # Update best k if this one is better
        if mean_accuracy > best_accuracy:
            best_k = k
            best_accuracy = mean_accuracy
    
    print(f"Best k value: {best_k} (accuracy: {best_accuracy:.4f})")
    return best_k, best_accuracy


def main():
    """Main function to run the sentiment analysis pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    print("SentimentScope: Text Sentiment Analysis")
    print("======================================")
    
    # Load data
    print("\nLoading data...")
    X_train, y_train, X_test = load_data(args.train, args.test)
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Plot sentiment distribution
    print("\nAnalyzing sentiment distribution...")
    plot_sentiment_distribution(y_train)
    
    # Plot initial word frequencies
    print("Analyzing word frequencies...")
    plot_word_frequencies(X_train['Reviews'], 
                         "Top 20 Words Before Preprocessing", 
                         "word_freq_before.png")
    
    # Preprocess text
    print("\nPreprocessing text data...")
    X_train, X_test = preprocess_text(X_train, X_test)
    
    # Plot word frequencies after preprocessing
    plot_word_frequencies(X_train['Reviews'], 
                         "Top 20 Words After Preprocessing", 
                         "word_freq_after.png")
    
    # Vectorize text
    print("\nVectorizing text data...")
    X_train_features, X_test_features = vectorize_text(
        X_train, X_test, n_components=args.components
    )
    
    # Find optimal k or use provided value
    if args.k is None:
        print("\nPerforming hyperparameter tuning...")
        k, _ = tune_hyperparameters(X_train_features, y_train, 
                                   max_k=args.max_k, n_folds=args.folds)
    else:
        k = args.k
        print(f"\nUsing provided k value: {k}")
    
    # Train final model
    print("\nTraining final model...")
    final_model = CosineSimilarityKNN(k=k)
    final_model.fit(X_train_features, y_train)
    
    # Generate predictions
    print("\nGenerating predictions for test data...")
    test_predictions = final_model.predict(X_test_features)
    
    # Write predictions to file
    print(f"Writing predictions to {args.output}...")
    with open(args.output, "w") as f:
        for prediction in test_predictions:
            f.write(f"{prediction}\n")
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()