#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run the sentiment analyzer with default parameters
python sentiment_analyzer.py --train data/reviews_train.csv --test data/reviews_test.csv --output predictions.txt

# Example with custom parameters
# python sentiment_analyzer.py --train data/reviews_train.csv --test data/reviews_test.csv --output predictions.txt --k 11 --components 150