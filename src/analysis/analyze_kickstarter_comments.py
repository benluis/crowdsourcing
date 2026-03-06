#!/usr/bin/env python3
"""
Kickstarter Comment Sentiment Analysis (Parallelized)
---------------------------------------------------
This script performs sentiment analysis on scraped Kickstarter comments.
It uses:
1. NLTK's NaiveBayesClassifier trained on the 'subjectivity' corpus.
2. NLTK's VADER for sentiment polarity.

Optimized with multiprocessing to utilize all available CPU cores.

Usage:
    python analyze_kickstarter_comments.py <path_to_csv>
"""

import os
import sys
import pandas as pd
import numpy as np
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime
import logging
import multiprocessing
from functools import partial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Global analyzer instance for workers
analyzer_instance = None

class KickstarterSentimentAnalyzer:
    def __init__(self):
        """
        Initializes the analyzer by downloading necessary NLTK data
        and training the Naive Bayes classifier.
        """
        logging.info(f"Initializing Sentiment Analyzer on process {os.getpid()}...")
        self._download_nltk_resources()
        self._train_subjectivity_classifier()
        self.vader = SentimentIntensityAnalyzer()
        logging.info("Initialization complete.")

    def _download_nltk_resources(self):
        """Downloads required NLTK corpora."""
        resources = ['subjectivity', 'vader_lexicon', 'punkt']
        for resource in resources:
            try:
                nltk.data.find(f'corpora/{resource}')
            except LookupError:
                # Only main process should log download info to avoid spam
                if multiprocessing.current_process().name == 'MainProcess':
                    logging.info(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
            except Exception:
                try:
                    nltk.data.find(f'tokenizers/{resource}')
                except LookupError:
                    if multiprocessing.current_process().name == 'MainProcess':
                        logging.info(f"Downloading NLTK resource: {resource}")
                    nltk.download(resource, quiet=True)

    def _train_subjectivity_classifier(self):
        """
        Trains a Naive Bayes classifier on the NLTK subjectivity corpus.
        """
        # Using the entire corpus for training
        n_instances = 5000 
        
        subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
        obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]
        
        train_docs = subj_docs + obj_docs
        
        all_words = SentimentAnalyzer().all_words([doc for doc in train_docs])
        unigram_feats = SentimentAnalyzer().unigram_word_feats(all_words, top_n=1000)
        self.sentim_analyzer = SentimentAnalyzer()
        self.sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)
        
        training_set = self.sentim_analyzer.apply_features(train_docs)
        self.classifier = NaiveBayesClassifier.train(training_set)

    def analyze_text(self, text):
        """
        Analyzes a single text string.
        """
        if not isinstance(text, str) or not text.strip():
            return {
                'subjectivity_label': None,
                'subjectivity_prob': 0.0,
                'vader_neg': 0.0,
                'vader_neu': 0.0,
                'vader_pos': 0.0,
                'vader_compound': 0.0
            }

        # 1. Subjectivity Analysis
        try:
            tokens = nltk.word_tokenize(text.lower())
            features = self.sentim_analyzer.apply_features([(tokens, '')], labeled=False)[0]
            subj_label = self.classifier.classify(features)
            prob_dist = self.classifier.prob_classify(features)
            subj_prob = prob_dist.prob(subj_label)
        except Exception:
            subj_label = None
            subj_prob = 0.0

        # 2. VADER Polarity Analysis
        try:
            vader_scores = self.vader.polarity_scores(text)
        except Exception:
            vader_scores = {'neg': 0.0, 'neu': 0.0, 'pos': 0.0, 'compound': 0.0}

        return {
            'subjectivity_label': subj_label,
            'subjectivity_prob': subj_prob,
            'vader_neg': vader_scores['neg'],
            'vader_neu': vader_scores['neu'],
            'vader_pos': vader_scores['pos'],
            'vader_compound': vader_scores['compound']
        }

def init_worker():
    """
    Initializer for worker processes. 
    Creates a global analyzer instance once per process.
    """
    global analyzer_instance
    analyzer_instance = KickstarterSentimentAnalyzer()

def process_row(text):
    """
    Worker function to process a single row of text.
    """
    global analyzer_instance
    return analyzer_instance.analyze_text(text)

def process_file(input_path):
    """
    Loads a CSV, processes comments in parallel, and saves the result.
    """
    if not os.path.exists(input_path):
        logging.error(f"Input file not found: {input_path}")
        return

    logging.info(f"Reading input file: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logging.error(f"Failed to read CSV: {e}")
        return

    # Identify the comment text column
    text_col = None
    candidates = ['body', 'comment', 'text', 'content']
    for col in candidates:
        if col in df.columns:
            text_col = col
            break
    
    if not text_col:
        # Fallback: check if it's an update file which might have different column names
        if 'update' in input_path.lower() or 'updates' in input_path.lower():
             candidates_update = ['body', 'update_body', 'post_body']
             for col in candidates_update:
                if col in df.columns:
                    text_col = col
                    break
    
    if not text_col:
        logging.error(f"Could not find a text column. Checked: {candidates}")
        return

    logging.info(f"Using column '{text_col}' for analysis. Total rows: {len(df)}")

    # Determine number of CPUs
    num_cpus = multiprocessing.cpu_count()
    # Reserve one core for system/overhead if possible, but on SLURM we use what we requested
    # If SLURM_CPUS_PER_TASK is set, use that
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    if slurm_cpus:
        num_cpus = int(slurm_cpus)
    
    logging.info(f"Starting parallel processing with {num_cpus} processes...")

    # Prepare data
    texts = df[text_col].fillna('').astype(str).tolist()

    # Run Analysis in Parallel
    with multiprocessing.Pool(processes=num_cpus, initializer=init_worker) as pool:
        # chunksize helps reduce IPC overhead
        chunksize = max(1, len(texts) // (num_cpus * 4))
        results = pool.map(process_row, texts, chunksize=chunksize)

    # Append results to DataFrame
    logging.info("Processing complete. Combining results...")
    results_df = pd.DataFrame(results)
    final_df = pd.concat([df, results_df], axis=1)

    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.basename(input_path)
    name, ext = os.path.splitext(base_name)
    output_path = os.path.join(os.path.dirname(input_path), f"{name}_sentiment_{timestamp}{ext}")

    logging.info(f"Saving results to: {output_path}")
    final_df.to_csv(output_path, index=False)
    logging.info("Done.")

import glob

def main():
    # Default to data/scraped if no argument provided
    if len(sys.argv) < 2:
        input_path = "data/scraped"
        logging.info(f"No input path provided. Defaulting to: {input_path}")
    else:
        input_path = sys.argv[1]

    if os.path.isdir(input_path):
        logging.info(f"Processing all COMMENT CSV files in directory: {input_path}")
        # Only process files containing 'comments' in the name
        csv_files = glob.glob(os.path.join(input_path, "*comments*.csv"))
        
        # Filter out files that look like they are already sentiment analyzed
        csv_files = [f for f in csv_files if "_sentiment_" not in f]
        
        if not csv_files:
            logging.warning(f"No CSV files found in {input_path}")
            return

        logging.info(f"Found {len(csv_files)} CSV files to process.")
        for csv_file in csv_files:
            logging.info(f"--- Processing {csv_file} ---")
            process_file(csv_file)
            
    elif os.path.isfile(input_path):
        process_file(input_path)
    else:
        logging.error(f"Input path not found: {input_path}")
        sys.exit(1)

if __name__ == "__main__":
    # NLTK downloader fix for multiprocessing
    multiprocessing.freeze_support()
    main()
