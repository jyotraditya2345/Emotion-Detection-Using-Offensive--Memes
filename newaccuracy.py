import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load trained model & vectorizer
model = joblib.load("/Users/jyotiradityachopra/Desktop/new major proect/filename_classifier.pkl")
vectorizer = joblib.load("/Users/jyotiradityachopra/Desktop/new major proect/filename_vectorizer.pkl")

# Define actual labels for dataset paths
dataset_paths = {
    "women_leaders": "/Users/jyotiradityachopra/Desktop/new major proect/women_leadership_memes",
    "women_shopping": "/Users/jyotiradityachopra/Desktop/new major proect/women shopping memes",
    "women_working": "/Users/jyotiradityachopra/Desktop/new major proect/working women memes",
    "women_kitchen": "/Users/jyotiradityachopra/Desktop/new major proect/women_kitchen_memes"
}

# Reverse mapping of dataset paths to labels
label_map = {v: k for k, v in dataset_paths.items()}

# Function to evaluate model accuracy
def evaluate_model(test_folder, model, vectorizer, label_map):
    if not os.path.exists(test_folder):
        print(f"Error: Folder '{test_folder}' not found.")
        return

    all_filenames = []
    true_labels = []
    pred_labels = []

    # Iterate through each labeled dataset folder
    for folder, label in label_map.items():
        if not os.path.exists(folder):
            continue
        
        filenames = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for filename in filenames:
            file_path = os.path.join(folder, filename)
            all_filenames.append(filename)
            true_labels.append(label)  # Actual label
            
            # Predict category
            X_new = vectorizer.transform([filename])
            predicted_label = model.predict(X_new)[0]
            pred_labels.append(predicted_label)  # Predicted label

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, pred_labels)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(true_labels, pred_labels, target_names=label_map.keys()))

    # Fix for confusion matrix issue
    unique_labels = sorted(set(true_labels) | set(pred_labels))  # Get only present labels

    # Generate confusion matrix with only valid labels
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=unique_labels)

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    return accuracy

# Example Usage
if __name__ == "__main__":
    test_folder = "/Users/jyotiradityachopra/Desktop/new major proect/test image folder"  # Update this path
    model_accuracy = evaluate_model(test_folder, model, vectorizer, label_map)
