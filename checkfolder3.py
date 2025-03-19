import os
import joblib
import pandas as pd

# Load trained model & vectorizer
model = joblib.load("filename_classifier.pkl")
vectorizer = joblib.load("filename_vectorizer.pkl")

# Dataset paths (Update these paths if needed)
dataset_paths = {
    "women_leaders": "/Users/jyotiradityachopra/Desktop/new major proect/women_leadership_memes",
    "women_shopping": "/Users/jyotiradityachopra/Desktop/new major proect/women shopping memes",
    "women_working": "/Users/jyotiradityachopra/Desktop/new major proect/working women memes",
    "women_kitchen": "/Users/jyotiradityachopra/Desktop/new major proect/women_kitchen_memes"
}

# Path to save the CSV for misplaced images
csv_save_path = "/Users/jyotiradityachopra/Desktop/new major proect/misplaced_images.csv"

# Function to classify filenames and detect misplacements
def check_folder(folder_path, model, vectorizer, dataset_paths):
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return
    
    filenames = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not filenames:
        print("No images found in the folder.")
        return
    
    predictions = []
    misplaced = []

    for filename in filenames:
        X_new = vectorizer.transform([filename])
        predicted_label = model.predict(X_new)[0]

        # Get actual folder name
        actual_label = None
        for label, path in dataset_paths.items():
            if folder_path == path:
                actual_label = label
                break
        
        # Check if the file is misplaced
        if predicted_label != actual_label:
            misplaced.append((filename, actual_label, predicted_label))

    # Save misplaced images to CSV
    if misplaced:
        df = pd.DataFrame(misplaced, columns=["Filename", "Correct Folder", "Predicted Folder"])
        df.to_csv(csv_save_path, index=False)
        print(f"\n=== Misplaced Image Report Saved: {csv_save_path} ===")
    else:
        print("\n✅ No misplaced images found!")

# Example Usage
if __name__ == "__main__":
    folder_to_check = "/Users/jyotiradityachopra/Desktop/new major proect/test image folder"  # Update this path
    check_folder(folder_to_check, model, vectorizer, dataset_paths)
