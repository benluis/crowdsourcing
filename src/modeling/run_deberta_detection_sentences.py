#!/usr/bin/env python3
"""
AI Text Detection (Sentence-Level) for Crowdfunding Project Descriptions

This script uses a fine-tuned DeBERTa model to predict the probability that 
EACH SENTENCE in a given text is AI-generated.

It outputs a new column containing a list of scores (one for each sentence).

Author: Analysis Team (Updated by Assistant)
Date: 2026
"""

import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
import warnings
import os
import nltk
from tqdm import tqdm

# Ensure NLTK data is available for sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

warnings.filterwarnings('ignore')

class DeBERTaDetector:
    """
    A detector that uses a fine-tuned DeBERTa model for AI text detection.
    """
    def __init__(self, model_path="models/deberta_v3"):
        """
        Initializes the detector by loading the model and tokenizer.
        Args:
            model_path (str): The local path to the saved fine-tuned model directory.
        """
        print(f"🔍 Loading fine-tuned model from: {model_path}")
        
        # Handle potential absolute paths or fallback
        if not os.path.isdir(model_path):
             # Try absolute path if provided path fails
            if os.path.isdir(os.path.abspath(model_path)):
                model_path = os.path.abspath(model_path)
                print(f"   Found model at absolute path: {model_path}")
            # Fallback to current dir
            elif os.path.isdir("./deberta_finetuned_model"):
                model_path = "./deberta_finetuned_model"
                print(f"   ⚠️ path not found, using fallback: {model_path}")
            else:
                raise FileNotFoundError(
                    f"Model directory not found at '{model_path}'. "
                    "Please make sure the trained model files are in this location."
                )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model and tokenizer loaded successfully.")

    def predict_batch(self, texts, batch_size=32):
        """
        Generates AI scores for a list of texts efficiently.
        """
        all_scores = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            probabilities = torch.softmax(logits, dim=1)
            ai_scores = probabilities[:, 1].cpu().numpy()
            all_scores.extend(ai_scores)
            
        return all_scores

def analyze_ai_usage_sentences(df, text_column='story_content'):
    """
    Analyzes AI usage per sentence across a DataFrame.
    """
    print("\n--- Starting DeBERTa Sentence-Level Analysis ---")
    
    try:
        # Defaults to models/deberta_v3
        detector = DeBERTaDetector() 
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None

    # 1. Prepare data: Split all texts into sentences
    print("✂️  Splitting texts into sentences...")
    
    # This list will hold tuples of (original_index, sentence_text)
    all_sentences_flat = []
    
    # We iterate with index to map back later
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Tokenizing"):
        text = str(row[text_column]) if pd.notna(row[text_column]) else ""
        if not text.strip():
            continue
            
        # Split into sentences
        try:
            sentences = nltk.sent_tokenize(text)
        except LookupError:
             # Fallback if punkt_tab is still missing for some reason
            print("⚠️ NLTK punkt_tab missing, attempting download...")
            nltk.download('punkt_tab')
            sentences = nltk.sent_tokenize(text)
        
        for sent in sentences:
            if len(sent.strip()) > 5: # Skip very short snippets/garbage
                all_sentences_flat.append((idx, sent))

    print(f"   Total sentences to score: {len(all_sentences_flat):,}")

    if not all_sentences_flat:
        print("❌ No valid sentences found to score.")
        return df

    # 2. Extract just the text for batch processing
    texts_to_score = [item[1] for item in all_sentences_flat]
    
    # 3. Score all sentences in batches
    print("🤖 Predicting scores...")
    scores = detector.predict_batch(texts_to_score, batch_size=32)
    
    # 4. Reconstruct the results
    # Create mappings: index -> list of scores, index -> list of sentences
    print("🔄 Reconstructing results...")
    results_map_scores = {idx: [] for idx in df.index}
    results_map_sentences = {idx: [] for idx in df.index}
    
    for (idx, sent), score in zip(all_sentences_flat, scores):
        score_val = float(score)
        results_map_scores[idx].append(score_val)
        results_map_sentences[idx].append(sent)

    # 5. Add to DataFrame
    result_df = df.copy()
    
    # Create the new columns
    result_df['ai_scores_sentences'] = [results_map_scores.get(idx, []) for idx in result_df.index]
    result_df['ai_sentences'] = [results_map_sentences.get(idx, []) for idx in result_df.index]
    
    # Calculate aggregates
    print("📊 Calculating aggregates...")
    def calc_mean(x): return np.mean(x) if x else 0.0
    def calc_median(x): return np.median(x) if x else 0.0
    def calc_max(x): return np.max(x) if x else 0.0

    result_df['ai_score_mean'] = result_df['ai_scores_sentences'].apply(calc_mean)
    result_df['ai_score_median'] = result_df['ai_scores_sentences'].apply(calc_median)
    result_df['ai_score_max'] = result_df['ai_scores_sentences'].apply(calc_max)
    
    print("📝 Appended columns: 'ai_scores_sentences', 'ai_sentences', 'ai_score_mean', 'ai_score_median', 'ai_score_max'")
    
    return result_df

def main():
    print("🚀 Starting AI Sentence Detection Script")
    
    # --- Configuration ---
    # Using the PKL file from data/processed
    input_file = "data/processed/final_with_deberta_ai_score_20251003_151656.pkl"
    # -------------------
    
    print(f"🎯 Using input file: {input_file}")
    
    try:
        # Load PKL instead of CSV
        df = pd.read_pickle(input_file)
        print(f"Loaded {len(df):,} projects. Columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find '{input_file}'.")
        return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Determine text column
    text_cols = ['story_content', 'description', 'story', 'blurb', 'Text']
    available_text_col = next((col for col in text_cols if col in df.columns), None)
    
    if not available_text_col:
        print(f"❌ Error: No suitable text column found.")
        return
    
    print(f"Using text column: '{available_text_col}'")
    
    # Run analysis
    result_df = analyze_ai_usage_sentences(df, text_column=available_text_col)
    
    if result_df is None:
        return

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_pkl = f"results/final_with_sentence_ai_scores_{timestamp}.pkl"
    # Also save a CSV version
    output_csv = f"results/final_with_sentence_ai_scores_{timestamp}.csv"
    
    print(f"\n💾 Saving results...")
    os.makedirs("results", exist_ok=True)
    
    result_df.to_pickle(output_pkl)
    print(f"  Full results saved to: {output_pkl}")
    
    # For CSV saving, lists might look like strings, but good for inspection
    result_df.to_csv(output_csv, index=False)
    print(f"  CSV results saved to: {output_csv}")
    
    print("\n✅ Script finished successfully!")

if __name__ == "__main__":
    main()
