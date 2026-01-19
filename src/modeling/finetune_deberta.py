import pandas as pd
import torch
from torch import nn, optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --- Configuration ---
MODEL_PATH = "./deberta_finetuned_model"
CALIBRATION_DATA_PATH = "validation.csv"
DATA_TO_SCORE_PATH = "test.csv"
TEXT_COLUMN = "Text"
CALIBRATION_LABEL_COLUMN = "LABEL_A"
OUTPUT_FILE_PATH = "final_scored_results.csv"
# -------------------

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs).logits
        return self.scale_temperature(logits)

    def scale_temperature(self, logits):
        return logits / self.temperature

def calibrate_model(model, tokenizer, calibration_df):
    print("--- Calibrating model with Temperature Scaling ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    scaled_model = ModelWithTemperature(model)
    scaled_model.to(device)

    cal_texts = calibration_df[TEXT_COLUMN].astype(str).tolist()
    cal_labels = torch.LongTensor(calibration_df[CALIBRATION_LABEL_COLUMN].values).to(device)

    cal_encodings = tokenizer(cal_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = cal_encodings['input_ids'].to(device)
    attention_mask = cal_encodings['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask).logits

    optimizer = optim.LBFGS([scaled_model.temperature], lr=0.01, max_iter=50)
    nll_criterion = nn.CrossEntropyLoss()

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(scaled_model.scale_temperature(logits), cal_labels)
        loss.backward()
        return loss

    optimizer.step(eval)

    optimal_temp = scaled_model.temperature.item()
    print(f"Optimal temperature found: {optimal_temp:.3f}")
    return optimal_temp

def predict(texts, model, tokenizer, temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    batch_size = 32
    all_ai_scores = []

    print(f"Predicting scores for {len(texts)} texts in batches of {batch_size}...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        encodings = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask=attention_mask).logits
            scaled_logits = logits / temperature
            probabilities = torch.softmax(scaled_logits, dim=-1)
            ai_scores = probabilities[:, 1].cpu().numpy()
            all_ai_scores.extend(ai_scores)

    return all_ai_scores

def main():
    print("--- Loading fine-tuned model and tokenizer ---")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"--- Loading labeled validation data for calibration from '{CALIBRATION_DATA_PATH}' ---")
    cal_df = pd.read_csv(CALIBRATION_DATA_PATH)

    temperature = calibrate_model(model, tokenizer, cal_df)

    print(f"--- Loading unlabeled test data to score from '{DATA_TO_SCORE_PATH}' ---")
    df_to_score = pd.read_csv(DATA_TO_SCORE_PATH)
    texts_to_score = df_to_score[TEXT_COLUMN].astype(str).tolist()

    print("--- Generating AI scores for the test data ---")
    ai_scores = predict(texts_to_score, model, tokenizer, temperature)

    df_to_score['ai_score'] = ai_scores

    print(f"--- Saving final scored results to '{OUTPUT_FILE_PATH}' ---")
    df_to_score.to_csv(OUTPUT_FILE_PATH, index=False)

    print("--- Prediction complete! ---")

if __name__ == "__main__":
    main()
