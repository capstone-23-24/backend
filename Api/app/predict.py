from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
import torch

# Load configuration
config_path = "../config/config.json"  # Adjust the path accordingly
config = RobertaConfig.from_pretrained(config_path)

# Load the RoBERTa model
roberta_model = RobertaForSequenceClassification.from_pretrained("../models/trained_model/")

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained(config.model_name_or_path)

def predict(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")

    # Make prediction
    with torch.no_grad():
        logits = roberta_model(**inputs).logitsx

    # Get predicted label
    predicted_label = torch.argmax(logits, dim=1).item()

    return predicted_label
