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

# Function to classify filenames and suggest correct folders
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

    # Store predictions in a DataFrame
    df = pd.DataFrame(predictions, columns=["Filename", "Predicted Folder"])

    # Save the CSV file with suggested folder placements
    csv_path = os.path.join(folder_path, "classification_report.csv")
    df.to_csv(csv_path, index=False)

    print(f"\n=== Classification Complete! Results saved in: {csv_path} ===")
    return df

# Example Usage
if __name__ == "__main__":
    folder_to_check = "/Users/jyotiradityachopra/Desktop/new major proect/test image folder"  # Update this path
    classification_results = check_folder(folder_to_check, model, vectorizer, dataset_paths)
    
    if classification_results is not None:
        print("\n=== Suggested Folder Moves ===")
        print(classification_results.to_string(index=False))
