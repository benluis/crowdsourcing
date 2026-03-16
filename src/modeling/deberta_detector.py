import torch
import os
import warnings
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Suppress warnings
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
                # If model is not found, we can try to download from HuggingFace if it's a standard model,
                # but for a fine-tuned model we expect it to be present.
                # For robustness in this pipeline, we'll try to use the base model if fine-tuned is missing
                # BUT warn heavily.
                print(f"   ⚠️ Model directory not found at '{model_path}'.")
                print("   ⚠️ Attempting to use base 'microsoft/deberta-v3-base' (WARNING: NOT FINE-TUNED)")
                model_path = "microsoft/deberta-v3-base"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

    def predict_score(self, text):
        """
        Generates a single AI probability score for a given text.
        """
        if not text or not isinstance(text, str) or not text.strip():
            return 0.0

        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        
        probabilities = torch.softmax(logits, dim=1)
        # Assuming index 1 is the "AI" class
        ai_score = probabilities[0, 1].item()
        
        return ai_score

    def predict_batch(self, texts, batch_size=32):
        """
        Generates AI scores for a list of texts efficiently.
        """
        all_scores = []
        
        # Filter out empty texts but keep track of indices to return correct length
        valid_indices = [i for i, t in enumerate(texts) if t and isinstance(t, str) and t.strip()]
        valid_texts = [texts[i] for i in valid_indices]
        
        if not valid_texts:
            return [0.0] * len(texts)

        # Process in batches
        for i in range(0, len(valid_texts), batch_size):
            batch_texts = valid_texts[i:i + batch_size]
            
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
            ai_scores = probabilities[:, 1].cpu().numpy().tolist()
            all_scores.extend(ai_scores)
            
        # Reconstruct full list with 0.0 for empty texts
        final_scores = [0.0] * len(texts)
        for idx, score in zip(valid_indices, all_scores):
            final_scores[idx] = score
            
        return final_scores
