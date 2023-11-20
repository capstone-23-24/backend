from fastapi import FastAPI
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig
from pydantic import BaseModel
import torch
import os

app = FastAPI()

class InputText(BaseModel):
    text: str

# Load configuration
config_path = os.path.abspath('../config/config.json')  # Adjust the path accordingly
config = RobertaConfig.from_pretrained(config_path)

# Load the RoBERTa model
roberta_model = RobertaForSequenceClassification.from_pretrained("../models/trained_model/")

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained(config.model_name_or_path)

@app.post("/predict")
async def predict(input_text: InputText):
    # Tokenize input text
    inputs = tokenizer(input_text.text, return_tensors="pt")

    # Make prediction
    with torch.no_grad():
        logits = roberta_model(**inputs).logits

    # Get predicted label
    predicted_label = torch.argmax(logits, dim=1).item()

    return {"predicted_label": predicted_label}
