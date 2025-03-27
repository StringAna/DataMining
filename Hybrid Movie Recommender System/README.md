# MovieMind: A Hybrid Machine Learning Framework for Film Recommendation

## Overview

MovieMind is a sophisticated recommendation system that combines collaborative filtering and content-based approaches to predict user movie ratings. The system leverages film metadata (actors, directors, genres) alongside user behavior patterns to generate highly personalized recommendations.

## Technical Approach

This project implements three different recommendation approaches:

- **Random Forest Regression**: Using actor, director, and genre features
- **Collaborative Filtering with K-Means**: Clustering similar users for improved predictions
- **Hybrid Model**: Combining collaborative and content-based methods with optimized weighting

## Business Value

MovieMind demonstrates my ability to deliver tangible business value through:

- **Personalization at Scale**: Increasing user engagement by providing tailored recommendations
- **Data-Driven Decision Making**: Leveraging multiple data sources to inform predictions
- **Algorithmic Optimization**: Implementing model tuning techniques to maximize accuracy
- **Cross-Domain Knowledge**: Combining user behavior analysis with content metadata

## Skills Demonstrated

- **Machine Learning**: Random Forest Regression, Matrix Factorization, K-Means Clustering
- **Data Engineering**: Data preprocessing, merging, and handling missing values
- **Feature Engineering**: Converting categorical features to numeric representations
- **Model Evaluation**: Implemented proper train/validation splits and RMSE evaluation
- **Optimization Techniques**: Hyperparameter tuning and model weighting strategies

## Alternative Datasets

If you want to try this code with publicly available data, you can use:

- **MovieLens**: The [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) contains 100,000 ratings from 943 users on 1,682 movies, with user demographics and movie genres.
- **The Movies Dataset**: Available on [Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset), includes metadata for 45,000 movies with 26 million ratings.

## Installation

```
pip install -r requirements.txt
```

## Usage

Run the Jupyter notebook to see the three models in action:

```
jupyter notebook MovieMind.ipynb
```
