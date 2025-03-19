# women-offensive-setection-content-memes
Women Meme Classification is an AI-powered system that classifies memes based on their filenames using TF-IDF vectorization and an SVM classifier. The project automates meme categorization into topics like leadership, shopping, working, and kitchen-related content. 

# Women Meme Classification  

## 📌 Project Overview  
This project **classifies memes based on filenames** using **TF-IDF vectorization** and an **SVM classifier**. It organizes images into four categories:  
- **Women Leadership Memes**  
- **Women Shopping Memes**  
- **Working Women Memes**  
- **Women in Kitchen Memes**  

The model analyzes image filenames and predicts the correct category, ensuring proper dataset organization.  

---

## ⚙️ Features  
✔️ **Automated Classification:** Uses machine learning to categorize memes.  
✔️ **TF-IDF Vectorization:** Converts filenames into numerical data.  
✔️ **SVM Classifier:** Predicts meme categories with high accuracy.  
✔️ **Misplacement Detection:** Identifies incorrectly categorized images.  
✔️ **Performance Evaluation:** Generates accuracy reports and a confusion matrix.  

---

## 🗂 Dataset Structure  
Organized into labeled folders:  
Each folder contains image files (**.jpg, .jpeg, .png**) representing memes for that category.  

---

## 🚀 How It Works  
### **1️⃣ Training the Model**  
- Extracts **image filenames** and applies **TF-IDF vectorization**.  
- Trains an **SVM classifier** to categorize memes.  
- Saves the trained model as `filename_classifier.pkl`.  

### **2️⃣ Classifying New Images**  
- Loads the trained classifier.  
- Predicts the category of filenames in a test folder.  
- Generates a classification report and lists misplaced images.  

### **3️⃣ Model Evaluation**  
- Runs tests on a labeled dataset.  
- Computes **accuracy, precision, recall, and F1-score**.  
- Generates a **confusion matrix** and **evaluation report (`model_evaluation_report.html`)**.  

---

## 🔧 Installation & Setup  
1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/your-username/women-meme-classification.git
cd women-meme-classification
