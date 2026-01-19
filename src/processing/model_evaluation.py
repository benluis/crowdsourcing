import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_performance(results_file_path, true_label_col, score_col, threshold=0.5):
    """
    Loads model predictions and evaluates performance against true labels.

    Args:
        results_file_path (str): Path to the CSV file with results.
        true_label_col (str): The name of the column with the correct answers (0 or 1).
        score_col (str): The name of the column with the model's predicted scores (0.0 to 1.0).
        threshold (float): The cutoff to convert scores to binary predictions.
    """
    try:
        df = pd.read_csv(results_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{results_file_path}' was not found.")
        print("Please make sure the script is in the same folder as your results CSV.")
        return

    # Convert the probability scores into binary predictions (0 or 1)
    df['predicted_label'] = (df[score_col] >= threshold).astype(int)

    # Get the true labels and the predicted labels from the DataFrame
    true_labels = df[true_label_col]
    predicted_labels = df['predicted_label']

    # --- Calculate and Print Performance Metrics ---
    print(f"--- Model Performance Metrics (Threshold = {threshold}) ---")
    print(f"Accuracy:  {accuracy_score(true_labels, predicted_labels):.4f}")
    print(f"Precision: {precision_score(true_labels, predicted_labels):.4f}")
    print(f"Recall:    {recall_score(true_labels, predicted_labels):.4f}")
    print(f"F1-Score:  {f1_score(true_labels, predicted_labels):.4f} (This is a key metric!)")
    print("-" * 50)

    # Print a more detailed report from scikit-learn
    print("Full Classification Report:")
    # target_names maps the labels (0, 1) to human-readable names
    print(classification_report(true_labels, predicted_labels, target_names=['Human (0)', 'AI (1)']))
    print("-" * 50)


    # --- Generate and Display a Confusion Matrix ---
    print("Generating Confusion Matrix visualization...")
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Use seaborn for a more visually appealing matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Human', 'Predicted AI'],
                yticklabels=['Actual Human', 'Actual AI'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()


if __name__ == "__main__":
    # --- CONFIGURATION ---
    # This should be the name of the CSV file you downloaded from Longleaf
    RESULTS_FILENAME = "validation_scored_results.csv"
    
    # These are the column names from your screenshot
    TRUE_LABEL_COLUMN = "LABEL_A"
    SCORE_COLUMN = "ai_score"
    # -------------------

    evaluate_performance(RESULTS_FILENAME, TRUE_LABEL_COLUMN, SCORE_COLUMN)
