from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError, validator
from typing import Optional
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class TextInput(BaseModel):
    text: str
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('text cannot be empty')
        return v.strip()

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    timestamp: str
    text_length: int
    processing_time: float

# Load model and tokenizer
try:
    MODEL_PATH = "models/hate_speech_model"
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)
    model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    logger.info(f"Model loaded successfully. Using device: {device}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.route('/predict', methods=['POST'])
def predict():
    start_time = datetime.now()
    
    try:
        # Parse and validate input using Pydantic
        input_data = request.json
        try:
            input_model = TextInput(**input_data)
        except ValidationError as e:
            return jsonify({"error": e.errors()}), 400
        
        # Tokenize input
        encoding = tokenizer.encode_plus(
            input_model.text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        # Make prediction
        with torch.no_grad():
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
            confidence = probabilities[0][prediction].item()
        
        # Convert prediction to label
        label = "hate" if prediction.item() == 1 else "noHate"
        
        # Calculate processing time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        response = {
            "prediction": label,
            "confidence": confidence,
            "timestamp": end_time.isoformat(),
            "text_length": len(input_model.text),
            "processing_time": processing_time
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
