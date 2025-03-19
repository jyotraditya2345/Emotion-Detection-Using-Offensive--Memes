import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Define dataset paths (UPDATE THESE PATHS)
dataset_paths = {
    "women_leaders": "/Users/jyotiradityachopra/Desktop/new major proect/women_leadership_memes",
    "women_shopping": "/Users/jyotiradityachopra/Desktop/new major proect/women shopping memes",
    "women_working": "/Users/jyotiradityachopra/Desktop/new major proect/working women memes",
    "women_kitchen": "/Users/jyotiradityachopra/Desktop/new major proect/women_kitchen_memes"
}

# Function to extract filenames and labels
def get_filenames_and_labels(dataset_paths):
    filenames = []
    labels = []
    for label, path in dataset_paths.items():
        if not os.path.exists(path):
            print(f"Warning: Path not found - {path}")
            continue
        for filename in os.listdir(path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Only image files
                filenames.append(filename)
                labels.append(label)
    return filenames, labels

# Get filenames and labels
filenames, labels = get_filenames_and_labels(dataset_paths)

# Convert to DataFrame
df = pd.DataFrame({"filename": filenames, "label": labels})

# Convert text (filenames) into numerical features using TF-IDF
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))  # Character-level n-grams
X = vectorizer.fit_transform(df["filename"])
y = df["label"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM Classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Save Model and Vectorizer
joblib.dump(model, "filename_classifier.pkl")
joblib.dump(vectorizer, "filename_vectorizer.pkl")

print("Model and vectorizer saved successfully!")
