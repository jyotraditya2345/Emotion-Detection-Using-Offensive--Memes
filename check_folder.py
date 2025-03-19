import os
import joblib
import pandas as pd

# Load trained model & vectorizer
model = joblib.load("filename_classifier.pkl")
vectorizer = joblib.load("filename_vectorizer.pkl")

# Dataset paths (UPDATE THESE)
dataset_paths = {
    "women_leaders": "/Users/jyotiradityachopra/Desktop/new major proect/women_leadership_memes",
    "women_shopping": "/Users/jyotiradityachopra/Desktop/new major proect/women shopping memes",
    "women_working": "/Users/jyotiradityachopra/Desktop/new major proect/working women memes",
    "women_kitchen": "/Users/jyotiradityachopra/Desktop/new major proect/women_kitchen_memes"
}

# Function to classify a batch of filenames in a folder
def check_folder(folder_path, model, vectorizer, dataset_paths):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return
    
    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not filenames:
        print("No images found in the folder.")
        return
    
    predictions = []
    
    for filename in filenames:
        X_new = vectorizer.transform([filename])
        predicted_label = model.predict(X_new)[0]
        predictions.append((filename, predicted_label))
    
    # Get actual folder name
    actual_label = None
    for label, path in dataset_paths.items():
        if folder_path == path:
            actual_label = label
            break
    
    print("\n=== Image Classification Report ===")
    correct = 0
    misplaced = []

    for filename, predicted_label in predictions:
        if predicted_label == actual_label:
            correct += 1
            print(f"[CORRECT] {filename} -> {predicted_label}")
        else:
            misplaced.append((filename, predicted_label))
            print(f"[MISPLACED] {filename} -> Should be in '{predicted_label}'")

    print(f"\nTotal Images Checked: {len(predictions)}")
    print(f"Correctly Placed: {correct}")
    print(f"Misplaced: {len(misplaced)}")

    return misplaced

# Example Usage
if __name__ == "__main__":
    folder_to_check = "/Users/jyotiradityachopra/Desktop/new major proect/test image folder"  # Update this path
    misplaced_files = check_folder(folder_to_check, model, vectorizer, dataset_paths)
    
    if misplaced_files:
        print("\n=== Suggested Folder Moves ===")
        for file, correct_folder in misplaced_files:
            print(f"Move '{file}' to '{correct_folder}' folder.")
