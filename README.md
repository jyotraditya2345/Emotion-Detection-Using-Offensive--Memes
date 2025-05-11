# 🧠 Emotion Detection on Offensive Memes

A deep learning project to detect emotions in offensive memes using image and text analysis. This project leverages CNNs, OCR, and NLP techniques to classify memes based on the emotions they evoke, especially in offensive contexts.

## 📌 Project Objective

To create an AI system capable of:
- Detecting **emotions** (e.g., anger, sadness, fear, etc.) present in **offensive memes**.
- Classifying memes using both **visual content** and **text extracted** from them.
- Providing useful insights to researchers and moderation systems in handling toxic content online.

---

## 🧩 Features

- OCR extraction from meme images using Tesseract.
- CNN model to analyze visual content of memes.
- Text classification using pretrained NLP models.
- Final emotion classification using a fusion of image and text features.
- Clean dataset structure with preprocessing utilities.
- CSV output mapping image files to their emotion categories.

---

## 🛠️ Tech Stack

- Python
- TensorFlow / Keras
- OpenCV
- Tesseract OCR
- NLTK / SpaCy
- Pandas / NumPy
- Matplotlib / Seaborn

---

## 📁 Folder Structure

emotion-detection-memes/
│
├── data/
│ ├── offensive_memes/
│ │ ├── meme1.jpg
│ │ ├── ...
│
├── ocr_extraction.py # Extract text from memes
├── image_preprocessing.py # Resize, normalize meme images
├── cnn_model.py # CNN model for image-based classification
├── text_classification.py # NLP-based emotion detection from text
├── fusion_model.py # Combine image and text for final prediction
├── utils.py # Common utility functions
├── predict.py # Run inference and generate output
├── results/
│ └── predictions.csv # Output mapping of meme to emotion
│
└── README.md


---

## 🚀 How to Run

Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Download Tesseract OCR

Tesseract Installation Guide

Place your dataset

Store your meme images inside the data/offensive_memes/ folder.

Run the pipeline

bash
Copy
Edit
python predict.py
Check results

Outputs saved in results/predictions.csv

📊 Results
Meme Image	Detected Emotion
meme1.jpg	Anger
meme2.jpg	Sadness
meme3.jpg	Fear


👨‍💻 Contributors
Jyotiraditya (Team Lead, Code and Integration)

Raman Sharma (Dataset Preparation, Testing)

Anurag Singh Chuhan (Model Development, Analysis)

Supervisors: Dr. Vivek Sehgal, Dr. Kushal Kanwar

📜 Acknowledgments
This project is developed as part of our academic research work. We are thankful to our professors for their guidance and to the open-source community for providing tools and libraries.


