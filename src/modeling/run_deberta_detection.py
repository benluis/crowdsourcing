#!/usr/bin/env python3
"""
AI Text Detection for Crowdfunding Project Descriptions

This script uses a fine-tuned DeBERTa model to predict the probability that a given
text is AI-generated. It loads a pre-trained model, processes text in batches,
and appends the resulting 'ai_score' to the input data.

Author: Analysis Team (Updated by Ben Luis)
Date: 2025
"""

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

class DeBERTaDetector:
    """
    A detector that uses a fine-tuned DeBERTa model for AI text detection.
    """
    def __init__(self, model_path="./deberta_finetuned_model"):
        """
        Initializes the detector by loading the model and tokenizer.

        Args:
            model_path (str): The local path to the saved fine-tuned model directory.
        """
        print(f"🔍 Loading fine-tuned model from: {model_path}")
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Model directory not found at '{model_path}'. "
                "Please make sure the trained model files are in this location."
            )
        
        # Set up the device (use GPU if available, otherwise CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load the tokenizer and model from the saved directory
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Move the model to the selected device and set it to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model and tokenizer loaded successfully.")

    def predict_scores(self, texts, batch_size=32):
        """
        Generates AI scores for a list of texts.

        Args:
            texts (list): A list of strings to analyze.
            batch_size (int): The number of texts to process at once.

        Returns:
            list: A list of AI probability scores (floats).
        """
        all_scores = []
        print(f"🤖 Predicting scores for {len(texts)} texts in batches of {batch_size}...")
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize the batch of texts
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)

            # Perform inference without calculating gradients for speed
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Convert logits to probabilities using the softmax function
            probabilities = torch.softmax(logits, dim=1)
            
            # The 'ai_score' is the probability of the "AI" class (label 1)
            ai_scores = probabilities[:, 1].cpu().numpy()
            
            all_scores.extend(ai_scores)
        
        print("Prediction complete.")
        return all_scores

def analyze_ai_usage(df, text_column='story_content'):
    """
    Analyzes AI usage across a DataFrame using the DeBERTa model.

    Args:
        df (pd.DataFrame): DataFrame with project data.
        text_column (str): The name of the column containing text to analyze.

    Returns:
        pd.DataFrame: The original DataFrame with the 'ai_score' column added.
    """
    print("\n--- Starting DeBERTa AI Detection Analysis ---")
    
    try:
        detector = DeBERTaDetector()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None

    # Get all texts from the specified column, ensuring they are strings
    texts_to_score = df[text_column].astype(str).fillna('').tolist()
    
    # Get all scores in one efficient call
    ai_scores = detector.predict_scores(texts_to_score)
    
    # Append the new 'ai_score' column to the original DataFrame
    result_df = df.copy()
    result_df['ai_score'] = ai_scores
    print("📝 Appended 'ai_score' column to the DataFrame.")
    
    # --- Summary Statistics ---
    print(f"\n📊 AI Detection Summary:")
    print(f"  Average AI score: {result_df['ai_score'].mean():.4f}")
    high_ai_mask = result_df['ai_score'] > 0.7
    print(f"  High AI likelihood (>0.7): {high_ai_mask.sum():,} projects ({high_ai_mask.mean()*100:.1f}%)")
    
    return result_df

def main():
    """Main execution function"""
    print("🚀 Starting AI Text Detection Script")
    
    # --- Configuration ---
    # This should point to the .pkl file you want to process
    input_file = "results/intermediate_with_text_quality.pkl"
    # -------------------

    print(f"🎯 Using input file: {input_file}")
    
    try:
        df = pd.read_pickle(input_file)
        print(f"Loaded {len(df):,} projects. Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find the input file '{input_file}'.")
        return
    
    # Find the best available text column to use for detection
    text_cols = ['story_content', 'description', 'story', 'blurb', 'Text']
    available_text_col = next((col for col in text_cols if col in df.columns), None)
    
    if not available_text_col:
        print(f"❌ Error: No suitable text column found. Looked for: {text_cols}")
        return
    
    print(f"Using text column: '{available_text_col}' for analysis.")
    
    # Run the AI detection using the DeBERTa model
    result_df = analyze_ai_usage(df, text_column=available_text_col)
    
    if result_df is None:
        print("Analysis could not be completed.")
        return

    # --- Save Results ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_pkl = f"results/final_with_deberta_ai_score_{timestamp}.pkl"
    output_csv_sample = f"results/final_with_deberta_ai_score_{timestamp}_sample.csv"
    
    print(f"\n💾 Saving results...")
    
    # Ensure the 'results' directory exists
    os.makedirs("results", exist_ok=True)
    
    # Save the full DataFrame as a pickle file
    result_df.to_pickle(output_pkl)
    print(f"  Full results saved to: {output_pkl}")
    
    # Save a sample of the results as a CSV for easy inspection
    result_df.head(1000).to_csv(output_csv_sample, index=False)
    print(f"  CSV sample saved to: {output_csv_sample}")
    
    print("\n✅ Script finished successfully!")

if __name__ == "__main__":
    main()
