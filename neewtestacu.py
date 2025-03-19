import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import mpld3  # For interactive plots
from jinja2 import Template  # For generating HTML reports

# Load trained model & vectorizer
model = joblib.load("/Users/jyotiradityachopra/Desktop/new major proect/filename_classifier.pkl")
vectorizer = joblib.load("/Users/jyotiradityachopra/Desktop/new major proect/filename_vectorizer.pkl")

# Define dataset paths
dataset_paths = {
    "women_leaders": "/Users/jyotiradityachopra/Desktop/new major proect/women_leadership_memes",
    "women_shopping": "/Users/jyotiradityachopra/Desktop/new major proect/women shopping memes",
    "women_working": "/Users/jyotiradityachopra/Desktop/new major proect/working women memes",
    "women_kitchen": "/Users/jyotiradityachopra/Desktop/new major proect/women_kitchen_memes"
}

# Reverse mapping of labels
label_map = {v: k for k, v in dataset_paths.items()}  

# Function to evaluate model and generate detailed report
def evaluate_model(test_folder, model, vectorizer, label_map):
    if not os.path.exists(test_folder):
        print(f"Error: Folder '{test_folder}' not found.")
        return

    all_filenames, true_labels, pred_labels = [], [], []

    for folder, label in label_map.items():
        if not os.path.exists(folder):
            continue

        filenames = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for filename in filenames:
            all_filenames.append(filename)
            true_labels.append(label)

            X_new = vectorizer.transform([filename])
            predicted_label = model.predict(X_new)[0]
            pred_labels.append(predicted_label)

    # Compute Accuracy
    accuracy = accuracy_score(true_labels, pred_labels)

    # Ensure all classes are dynamically accounted for
    unique_labels = sorted(set(true_labels) | set(pred_labels))  
    class_report = classification_report(true_labels, pred_labels, labels=unique_labels, output_dict=True)

    # Generate Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=unique_labels)

    # Create visualizations
    figures = []

    # Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    figures.append(mpld3.fig_to_html(plt.gcf()))  # Convert to interactive HTML

    # Bar Plot of Class-wise Accuracy
    plt.figure(figsize=(10, 5))
    class_accuracies = [class_report[label]["precision"] for label in unique_labels]
    sns.barplot(x=unique_labels, y=class_accuracies, palette="viridis")
    plt.xlabel("Classes")
    plt.ylabel("Precision Score")
    plt.title("Class-wise Precision Score")
    plt.xticks(rotation=45)
    figures.append(mpld3.fig_to_html(plt.gcf()))

    # Pie Chart of Predicted Class Distribution
    plt.figure(figsize=(6, 6))
    unique, counts = np.unique(pred_labels, return_counts=True)
    plt.pie(counts, labels=unique, autopct='%1.1f%%', colors=sns.color_palette("pastel"))
    plt.title("Predicted Class Distribution")
    figures.append(mpld3.fig_to_html(plt.gcf()))

    # Create HTML Report
    report_template = """
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; padding: 20px; background-color: #f4f4f4; }
            h1 { text-align: center; color: #222; }
            .container { background: white; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px #ccc; }
            .section { margin-bottom: 40px; }
            .graph { text-align: center; margin: 20px 0; }
            .explanation { text-align: justify; margin-top: 10px; }
            .graph-container { display: flex; flex-direction: column; align-items: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Evaluation Report</h1>

            <div class="section">
                <h2>Model Performance</h2>
                <p><strong>Accuracy:</strong> {{ accuracy }}%</p>
                <p>The accuracy score measures how well the model classifies memes into different categories. Higher accuracy indicates better performance.</p>
            </div>

            <div class="section">
                <h2>Confusion Matrix</h2>
                <div class="graph graph-container">{{ conf_matrix }}</div>
                <p class="explanation">
                    The confusion matrix provides insight into how well the model distinguishes between different meme categories.
                    Each row represents the actual class, while each column represents the predicted class.
                    A perfect model would have all values along the diagonal, indicating 100% correct classifications.
                </p>
            </div>

            <div class="section">
                <h2>Class-wise Precision Scores</h2>
                <div class="graph graph-container">{{ bar_plot }}</div>
                <p class="explanation">
                    This bar chart displays the precision score for each class.
                    A high precision score means the model is good at avoiding false positives for that category.
                    If a class has a low precision score, it indicates that the model often misclassifies non-relevant images as belonging to that class.
                </p>
            </div>

            <div class="section">
                <h2>Predicted Class Distribution</h2>
                <div class="graph graph-container">{{ pie_chart }}</div>
                <p class="explanation">
                    The pie chart shows the distribution of predicted classes.
                    If one class is disproportionately large, the model might be biased towards predicting that category more frequently.
                    Ideally, the distribution should be balanced to indicate fair classification.
                </p>
            </div>

            <div class="section">
                <h2>Classification Report</h2>
                <pre>{{ class_report }}</pre>
            </div>
        </div>
    </body>
    </html>
    """

    # Render HTML Report
    template = Template(report_template)
    report_html = template.render(
        accuracy=f"{accuracy * 100:.2f}",
        conf_matrix=figures[0],
        bar_plot=figures[1],
        pie_chart=figures[2],
        class_report=pd.DataFrame(class_report).transpose().to_html()
    )

    # Save Report
    report_path = os.path.join(test_folder, "model_evaluation_report.html")
    with open(report_path, "w") as f:
        f.write(report_html)

    print(f"✅ Report saved at: {report_path}")


# Example Usage
if __name__ == "__main__":
    test_folder = "/Users/jyotiradityachopra/Desktop/new major proect/test image folder"
    evaluate_model(test_folder, model, vectorizer, label_map)
