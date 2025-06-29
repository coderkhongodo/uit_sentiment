# -*- coding: utf-8 -*-
"""
PhoBERT Model for Sentiment Analysis
Updated to use RobertaForSequenceClassification like reference notebook
"""

import torch
import torch.nn as nn
import json
import os
from transformers import (
    RobertaForSequenceClassification, 
    RobertaConfig, 
    AutoTokenizer,
    AutoModel
)

class PhoBERTSentimentClassifier:
    """
    PhoBERT-based sentiment classifier using RobertaForSequenceClassification
    Based on reference notebook implementation that achieved 93.9% accuracy
    """
    
    def __init__(self, num_labels=3, model_name="vinai/phobert-base"):
        self.num_labels = num_labels
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self, model_path=None):
        """Load PhoBERT model for sequence classification"""
        print(f"Loading PhoBERT model...")
        
        try:
            if model_path and os.path.exists(model_path):
                # Load from saved model
                print(f"Loading from saved model: {model_path}")
                config = RobertaConfig.from_pretrained(
                    model_path,
                    from_tf=False,
                    num_labels=self.num_labels,
                    output_hidden_states=False
                )
                self.model = RobertaForSequenceClassification.from_pretrained(
                    model_path,
                    config=config
                )
                self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
            else:
                # Load pretrained model
                print(f"Loading pretrained model: {self.model_name}")
                config = RobertaConfig.from_pretrained(
                    self.model_name,
                    from_tf=False,
                    num_labels=self.num_labels,
                    output_hidden_states=False
                )
                self.model = RobertaForSequenceClassification.from_pretrained(
                    self.model_name,
                    config=config
                )
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
            
            # Move to device
            self.model.to(self.device)
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(f"Model loaded successfully!")
            print(f"Device: {self.device}")
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_model(self, save_path):
        """Save the model and tokenizer"""
        if self.model is None:
            raise ValueError("No model to save. Please load a model first.")
        
        print(f"Saving model to {save_path}...")
        os.makedirs(save_path, exist_ok=True)
        
        # Save model and config
        self.model.save_pretrained(save_path)
        
        # Save model info
        model_info = {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'device': str(self.device),
            'architecture': 'RobertaForSequenceClassification'
        }
        
        with open(os.path.join(save_path, 'model_info.json'), 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model saved successfully to {save_path}")
    
    def predict(self, text, return_probabilities=False):
        """
        Predict sentiment for a single text
        Based on reference notebook prediction function
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            if return_probabilities:
                probabilities = torch.softmax(logits, dim=-1)
                return probabilities.cpu().numpy()[0]
            else:
                predicted_class_id = logits.argmax().item()
                return predicted_class_id
    
    def predict_batch(self, texts, batch_size=16):
        """Predict sentiment for batch of texts"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Please load a model first.")
        
        self.model.eval()
        predictions = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=256,
                truncation=True,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_predictions = logits.argmax(dim=-1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        return predictions
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            return "No model loaded"
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            'architecture': 'RobertaForSequenceClassification',
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }
        
        return summary

def create_model():
    """Create and return a new PhoBERT sentiment classifier"""
    return PhoBERTSentimentClassifier(
        num_labels=3,
        model_name="vinai/phobert-base"
    )

def load_trained_model(model_path):
    """Load a trained model"""
    classifier = PhoBERTSentimentClassifier()
    success = classifier.load_model(model_path)
    
    if success:
        return classifier
    else:
        return None

def test_model():
    """Test model functionality"""
    print("=== Testing PhoBERT Model ===")
    
    # Create model
    model = create_model()
    
    # Load model
    success = model.load_model()
    if not success:
        print("Failed to load model")
        return
    
    # Test prediction
    test_texts = [
        "Giáo viên giảng dở, chưa nhiệt tình, còn nhiều thiếu xót",
        "Giáo viên giảng hay, khá nhiệt tình, cho điểm tương đối cao",
        "ổn"
    ]
    
    print("\nTesting predictions:")
    for text in test_texts:
        prediction = model.predict(text)
        probabilities = model.predict(text, return_probabilities=True)
        print(f"Text: {text}")
        print(f"Prediction: {prediction}")
        print(f"Probabilities: {probabilities}")
        print()
    
    # Test batch prediction
    batch_predictions = model.predict_batch(test_texts)
    print(f"Batch predictions: {batch_predictions}")
    
    # Print model summary
    summary = model.get_model_summary()
    print("\nModel Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    test_model() 