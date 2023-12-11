#!/usr/bin/python3

# -*- Author: Ali (Real0x0a1) -*-
# -*- Info: This script demonstrates a professional approach to using K-Nearest Neighbors (KNN) for classification. -*-
# -*- It loads and preprocesses a car evaluation dataset, trains a KNN classifier, and evaluates its performance. -*-

import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsClassifier as knc
from sklearn import preprocessing as pr

# Initialize counters for correct and incorrect predictions
CORRECT = 0
INCORRECT = 0


def load_and_preprocess_data(file_path):
    """Load and preprocess data from CSV file."""
    data = pd.read_csv(file_path)

    # Encode categorical features
    label_encoder = pr.LabelEncoder()
    features_to_encode = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']

    for feature in features_to_encode:
        data[feature] = label_encoder.fit_transform(data[feature])

    return data, label_encoder, features_to_encode


def train_knn_model(X_train, y_train, n_neighbors=7):
    """Train KNN classifier."""
    model = knc(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, classes):
    """Evaluate the model and print results."""
    accuracy = model.score(X_test, y_test)
    print(f'Model Accuracy: {accuracy:.2%}')

    # Make predictions and compare with actual labels
    predicted = model.predict(X_test)

    global CORRECT, INCORRECT
    for i, (pred, actual) in enumerate(zip(predicted, y_test)):
        predicted_class = classes[pred]
        actual_class = classes[actual]
        print(f'Example {i + 1}: Predicted: {predicted_class} | Actual: {actual_class}')

        # Check correctness of the prediction
        if predicted_class == actual_class:
            CORRECT += 1
        else:
            INCORRECT += 1

    print(f'Correct Predictions: {CORRECT} | Incorrect Predictions: {INCORRECT}')


if __name__ == "__main__":
    # Load and preprocess data
    data, label_encoder, features_to_encode = load_and_preprocess_data(
        'Car-Evaluation/car.data')

    # Create feature matrix (X) and target labels (y)
    X = data[features_to_encode[:-1]].values
    y = data['class'].values

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = tts(X, y, test_size=0.1)

    # Train the KNN model
    model = train_knn_model(X_train, y_train)

    # Evaluate the model
    classes = ['unacc', 'acc', 'good', 'vgood']
    evaluate_model(model, X_test, y_test, classes)

