Emotion Detection Using Offensive Memes
This project focuses on detecting and classifying offensive memes based on the emotions they evoke, using machine learning techniques. By analyzing meme filenames and associated metadata, the system automatically identifies emotional tones such as anger, humor, sarcasm, or hate. The goal is to support better content moderation and understanding of the emotional impact of offensive media.

Project Title and Description
Project Title: Emotion Detection Using Offensive Memes
Description: This is an AI-based system for categorizing memes into emotional categories by analyzing filenames and associated text data. The model uses TF-IDF vectorization and an SVM classifier to automate classification tasks and help organize meme datasets based on the emotions they express.

Folder Structure
emotion-detection-offensive-memes/
│
├── dataset/
│   ├── Angry Memes/
│   ├── Sarcastic Memes/
│   ├── Humorous Memes/
│   └── Hateful Memes/
│
├── model/
│   └── emotion_classifier.pkl
│
├── src/
│   ├── train_emotion_model.py
│   ├── classify_emotions.py
│   └── evaluate_emotions.py
│
├── reports/
│   └── emotion_evaluation_report.html

Source Code

train_emotion_model.py – Preprocesses meme filenames and trains the SVM classifier using TF-IDF features.

classify_emotions.py – Loads the trained model and classifies memes in the test set.

evaluate_emotions.py – Evaluates the classifier and generates detailed performance reports.

Documentation

Key Features
Automated classification of memes into emotional categories.

Uses TF-IDF vectorization to extract meaningful features from filenames.

Employs SVM classifier for high-accuracy emotion prediction.

Identifies incorrectly categorized images for review.

Generates comprehensive reports, including confusion matrix and metrics.

Dataset Structure

The dataset is organized into clearly labeled folders, each representing a specific emotional category:

Angry Memes

Sarcastic Memes

Humorous Memes

Hateful Memes

Installation Instructions

Clone the repository
git clone https://github.com/yourusername/emotion-detection-offensive-memes.git
cd emotion-detection-offensive-memes

Install dependencies
pip install -r requirements.txt

Train the emotion detection model
python src/train_emotion_model.py

Classify new memes
python src/classify_emotions.py

Evaluate the model
python src/evaluate_emotions.py



