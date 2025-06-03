from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

MODEL_NAME = "mrm8488/bert-tiny-finetuned-fake-news"
SAVE_DIR = "./model/bert_fake_news"

def download_model():
    print(f"🔄 Downloading model '{MODEL_NAME}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    os.makedirs(SAVE_DIR, exist_ok=True)
    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(SAVE_DIR)
    print(f"✅ Model and tokenizer saved to '{SAVE_DIR}'")

if __name__ == "__main__":
    download_model()
