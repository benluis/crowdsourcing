#!/usr/bin/env python3
"""
AI Text Detection for Crowdfunding Project Descriptions

This script implements multiple methods to detect AI-generated text in project descriptions,
ranging from simple heuristics to advanced machine learning models.

Methods implemented:
1. GPTZero API (when available)
2. Linguistic feature analysis 
3. Perplexity-based detection
4. Writing pattern analysis
5. Ensemble scoring

Author: Analysis Team
Date: 2025
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import textstat
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class AITextDetector:
    """
    Multi-method AI text detection system for crowdfunding descriptions
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def calculate_linguistic_features(self, text):
        """
        Calculate linguistic features that may indicate AI generation
        
        Args:
            text (str): Text to analyze
            
        Returns:
            dict: Dictionary of linguistic features
        """
        if pd.isna(text) or not isinstance(text, str) or len(text.strip()) == 0:
            return self._empty_features()
            
        features = {}
        
        # Basic text statistics
        features['word_count'] = len(word_tokenize(text))
        features['sentence_count'] = len(sent_tokenize(text))
        features['char_count'] = len(text)
        
        # Avoid division by zero
        if features['sentence_count'] > 0:
            features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
        else:
            features['avg_sentence_length'] = 0
            
        if features['word_count'] > 0:
            features['avg_word_length'] = features['char_count'] / features['word_count']
        else:
            features['avg_word_length'] = 0
        
        # Readability scores (AI text often has consistent readability)
        features['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
        features['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
        features['gunning_fog'] = textstat.gunning_fog(text)
        features['automated_readability_index'] = textstat.automated_readability_index(text)
        
        # Lexical diversity (AI text often less diverse)
        words = word_tokenize(text.lower())
        unique_words = set(words)
        if len(words) > 0:
            features['lexical_diversity'] = len(unique_words) / len(words)
        else:
            features['lexical_diversity'] = 0
            
        # Stop word ratio (AI text patterns)
        stop_word_count = sum(1 for word in words if word in self.stop_words)
        if len(words) > 0:
            features['stop_word_ratio'] = stop_word_count / len(words)
        else:
            features['stop_word_ratio'] = 0
        
        # Punctuation patterns
        features['exclamation_ratio'] = text.count('!') / len(text) if len(text) > 0 else 0
        features['question_ratio'] = text.count('?') / len(text) if len(text) > 0 else 0
        features['comma_ratio'] = text.count(',') / len(text) if len(text) > 0 else 0
        
        # Repetition patterns (AI sometimes repeats phrases)
        features['repetition_score'] = self._calculate_repetition_score(text)
        
        # Formality score (AI text often more formal)
        features['formality_score'] = self._calculate_formality_score(text)
        
        # AI-characteristic patterns
        features['ai_phrases_count'] = self._count_ai_phrases(text)
        
        return features
    
    def _empty_features(self):
        """Return empty features dict for invalid input"""
        return {
            'word_count': 0, 'sentence_count': 0, 'char_count': 0,
            'avg_sentence_length': 0, 'avg_word_length': 0,
            'flesch_reading_ease': 0, 'flesch_kincaid_grade': 0,
            'gunning_fog': 0, 'automated_readability_index': 0,
            'lexical_diversity': 0, 'stop_word_ratio': 0,
            'exclamation_ratio': 0, 'question_ratio': 0, 'comma_ratio': 0,
            'repetition_score': 0, 'formality_score': 0, 'ai_phrases_count': 0
        }
    
    def _calculate_repetition_score(self, text):
        """Calculate how much text repeats itself"""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0
            
        repetition_count = 0
        for i, sent1 in enumerate(sentences):
            for j, sent2 in enumerate(sentences[i+1:], i+1):
                # Simple similarity check
                words1 = set(word_tokenize(sent1.lower()))
                words2 = set(word_tokenize(sent2.lower()))
                if len(words1.union(words2)) > 0:
                    similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                    if similarity > 0.5:  # High similarity threshold
                        repetition_count += 1
                        
        return repetition_count / len(sentences) if len(sentences) > 0 else 0
    
    def _calculate_formality_score(self, text):
        """Calculate formality score (AI text often more formal)"""
        formal_indicators = [
            'furthermore', 'moreover', 'therefore', 'consequently', 'additionally',
            'specifically', 'particularly', 'essentially', 'ultimately', 'overall',
            'comprehensive', 'innovative', 'cutting-edge', 'state-of-the-art'
        ]
        
        words = word_tokenize(text.lower())
        formal_count = sum(1 for word in words if word in formal_indicators)
        
        return formal_count / len(words) if len(words) > 0 else 0
    
    def _count_ai_phrases(self, text):
        """Count phrases commonly used by AI models"""
        ai_phrases = [
            'i am excited to', 'we are thrilled to', 'cutting-edge technology',
            'state-of-the-art', 'innovative solution', 'game-changing',
            'revolutionary approach', 'seamless experience', 'user-friendly interface',
            'leverage the power of', 'harness the potential', 'transform the way',
            'bridge the gap', 'take it to the next level', 'paradigm shift'
        ]
        
        text_lower = text.lower()
        count = sum(1 for phrase in ai_phrases if phrase in text_lower)
        
        return count
    
    
    def calculate_ai_score(self, text, created_date=None):
        """
        Calculate overall AI likelihood score based on linguistic patterns only
        
        Args:
            text (str): Text to analyze
            created_date: When the project was created (unused, kept for compatibility)
            
        Returns:
            dict: AI detection results
        """
        # Get linguistic features
        features = self.calculate_linguistic_features(text)
        
        # AI likelihood based on linguistic patterns only
        linguistic_score = 0
        
        # Readability consistency (AI often very consistent)
        if 30 <= features['flesch_reading_ease'] <= 70:  # Moderate readability
            linguistic_score += 0.1
            
        # Moderate lexical diversity (AI often in middle range)
        if 0.3 <= features['lexical_diversity'] <= 0.7:
            linguistic_score += 0.1
            
        # Consistent sentence length (AI often uniform)
        if 15 <= features['avg_sentence_length'] <= 25:
            linguistic_score += 0.1
            
        # High formality score
        if features['formality_score'] > 0.02:
            linguistic_score += 0.2
            
        # AI phrases present
        if features['ai_phrases_count'] > 0:
            linguistic_score += 0.3
            
        # Low repetition (AI usually avoids obvious repetition)
        if features['repetition_score'] < 0.1:
            linguistic_score += 0.1
            
        # Moderate punctuation usage
        total_punct = features['exclamation_ratio'] + features['question_ratio']
        if 0.01 <= total_punct <= 0.05:
            linguistic_score += 0.1
        
        # Use linguistic score as final AI score
        ai_score = linguistic_score
            
        # Ensure score is between 0 and 1
        ai_score = max(0, min(1, ai_score))
        
        return {
            'ai_score': ai_score,
            'linguistic_score': linguistic_score,
            'ai_phrases_count': features['ai_phrases_count'],
            'formality_score': features['formality_score'],
            'lexical_diversity': features['lexical_diversity'],
            'readability_score': features['flesch_reading_ease']
        }

def analyze_ai_usage(df, text_column='story_content', date_column='created_at_parsed'):
    """
    Analyze AI usage across the entire dataset
    
    Args:
        df: DataFrame with project data
        text_column: Column containing text to analyze
        date_column: Column containing creation dates (unused, kept for compatibility)
        
    Returns:
        DataFrame: Original data with AI detection scores added
    """
    print("🤖 AI Text Detection Analysis")
    print("=" * 50)
    
    detector = AITextDetector()
    results = []
    
    print(f"Analyzing {len(df):,} projects...")
    
    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  Processed {idx:,} projects...")
            
        text = row.get(text_column, '')
        
        ai_result = detector.calculate_ai_score(text)
        results.append(ai_result)
    
    # Add results to dataframe - always append to original structure
    ai_df = pd.DataFrame(results)
    result_df = df.copy()
    
    print("  📝 Appending AI scores to original data structure...")
    for col in ai_df.columns:
        result_df[col] = ai_df[col]
    
    # Summary statistics
    print(f"\n📊 AI Detection Summary:")
    print(f"  Average AI score: {result_df['ai_score'].mean():.3f}")
    print(f"  High AI likelihood (>0.7): {(result_df['ai_score'] > 0.7).sum():,} projects ({(result_df['ai_score'] > 0.7).mean()*100:.1f}%)")
    print(f"  Medium AI likelihood (0.4-0.7): {((result_df['ai_score'] >= 0.4) & (result_df['ai_score'] <= 0.7)).sum():,} projects")
    print(f"  Low AI likelihood (<0.4): {(result_df['ai_score'] < 0.4).sum():,} projects")
    
    return result_df

def main():
    """Main execution function"""
    print("🚀 Starting AI Text Detection Analysis")
    print("=" * 50)
    
    # Change this path to the file you want to analyze
    input_file = "results/intermediate_with_text_quality.pkl"
    
    print(f"🎯 Using input file: {input_file}")
    
    # Load data
    try:
        print("Loading data...")
        df = pd.read_pickle(input_file)
        print(f"Loaded {len(df):,} projects")
        print(f"Available columns: {list(df.columns)}")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {input_file}")
        print("Please check the file path and make sure it exists")
        return
    
    # Check for required columns - prioritize story_content
    text_cols = ['story_content', 'description', 'story', 'blurb']
    available_text_col = None
    
    for col in text_cols:
        if col in df.columns:
            available_text_col = col
            break
    
    if not available_text_col:
        print(f"❌ Error: No text column found. Looking for: {text_cols}")
        print("Available columns:", list(df.columns))
        
        # Look for any columns that might contain text
        potential_text_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['story', 'description', 'text', 'content', 'blurb', 'narrative'])]
        if potential_text_cols:
            print(f"Potential text columns found: {potential_text_cols}")
        return
    
    print(f"Using text column: {available_text_col}")
    
    # Check if the text column has meaningful data
    if available_text_col in df.columns:
        non_empty_text = df[available_text_col].notna() & (df[available_text_col].astype(str).str.strip() != '')
        print(f"Text column '{available_text_col}' has {non_empty_text.sum():,} non-empty entries out of {len(df):,} total")
        
        # Show sample of the text content
        sample_texts = df[available_text_col].dropna().head(3)
        print(f"\nSample text content from '{available_text_col}':")
        for i, text in enumerate(sample_texts):
            print(f"  Sample {i+1}: {str(text)[:100]}...")
        
        if non_empty_text.sum() == 0:
            print(f"❌ Warning: No meaningful text found in '{available_text_col}' column")
            return
    
    # Show platform distribution if available
    if 'platform' in df.columns:
        print(f"\n📊 Platform distribution:")
        platform_counts = df['platform'].value_counts()
        for platform, count in platform_counts.items():
            print(f"  {platform}: {count:,} projects")
    
    # Run AI detection
    result_df = analyze_ai_usage(df, text_column=available_text_col)
    
    # Save results - always use update_original format
    output_file = f"results/intermediate_with_text_quality_and_ai_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    print(f"\n📝 AI scores appended to original data structure")
    
    result_df.to_pickle(output_file)
    
    # Save CSV sample for inspection
    csv_file = output_file.replace('.pkl', '_sample.csv')
    
    # Save ALL columns to CSV (original + AI scores)
    print(f"  Saving CSV with {len(result_df.columns)} columns...")
    result_df.head(1000).to_csv(csv_file, index=False)
    
    # Also save a summary CSV with key columns for quick inspection
    summary_csv = output_file.replace('.pkl', '_summary.csv')
    
    # Choose key columns for summary
    key_cols = ['platform', 'created_at_parsed', available_text_col, 'ai_score', 
                'linguistic_score', 'ai_phrases_count', 'formality_score', 
                'lexical_diversity', 'readability_score']
    
    key_cols = [col for col in key_cols if col in result_df.columns]
    result_df[key_cols].head(1000).to_csv(summary_csv, index=False)
    
    print(f"\n💾 Results saved:")
    print(f"  Full results: {output_file}")
    print(f"  Complete CSV sample: {csv_file} (all {len(result_df.columns)} columns)")
    print(f"  Summary CSV: {summary_csv} (key columns only)")
    
    # Show high AI score examples
    print(f"\n🔍 Examples of High AI Score Projects:")
    high_ai = result_df[result_df['ai_score'] > 0.8].head(3)
    for idx, row in high_ai.iterrows():
        print(f"\n  Project (AI Score: {row['ai_score']:.3f}):")
        text = str(row[available_text_col])[:200] + "..."
        print(f"  {text}")

if __name__ == "__main__":
    main()
    
    print("\n" + "="*60)
    print("💡 USAGE:")
    print("  To change input file, modify 'selected_file' variable in main() function")
    print("  AI scores are automatically appended to the original data structure")
    print("  Temporal scoring removed - use DiD analysis for time-based effects")
    print("="*60)
