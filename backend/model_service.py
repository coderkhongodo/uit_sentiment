"""
Model Service for PhoBERT Sentiment Analysis
Wraps the existing PhoBERT model for API integration
"""

import asyncio
import logging
import time
import sys
import os
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import torch

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import PhoBERTSentimentClassifier

logger = logging.getLogger(__name__)

class SentimentModelService:
    """Service wrapper for PhoBERT sentiment analysis model"""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), '..', 'saved_results', 'final_model'
        )
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.label_mapping = {
            0: "negative",
            1: "neutral", 
            2: "positive"
        }
        self._is_loaded = False
        
    async def initialize(self):
        """Initialize the model asynchronously"""
        try:
            logger.info("Initializing PhoBERT sentiment model...")
            
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.executor, self._load_model)
            
            self._is_loaded = True
            logger.info("Model initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def _load_model(self):
        """Load the PhoBERT model (blocking operation)"""
        try:
            self.model = PhoBERTSentimentClassifier()
            success = self.model.load_model(self.model_path)
            
            if not success:
                raise Exception("Failed to load model from path")
                
            # Test model with a simple prediction
            test_result = self.model.predict("Test text", return_probabilities=True)
            logger.info(f"Model test successful: {test_result}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        return self._is_loaded and self.model is not None
    
    async def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict sentiment for a single text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment, confidence, and probabilities
        """
        if not self.is_loaded():
            raise Exception("Model not loaded. Call initialize() first.")
        
        try:
            # Run prediction in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor, 
                self._predict_sync, 
                text
            )
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise
    
    def _predict_sync(self, text: str) -> Dict[str, Any]:
        """Synchronous prediction method"""
        try:
            # Get prediction with probabilities
            prediction = self.model.predict(text)
            probabilities = self.model.predict(text, return_probabilities=True)
            
            # Convert to our format
            sentiment_label = self.label_mapping[prediction]
            confidence = max(probabilities)
            
            # Create probability dictionary
            prob_dict = {
                "negative": float(probabilities[0]),
                "neutral": float(probabilities[1]),
                "positive": float(probabilities[2])
            }
            
            return {
                "sentiment": sentiment_label,
                "confidence": float(confidence),
                "probabilities": prob_dict
            }
            
        except Exception as e:
            logger.error(f"Sync prediction error: {str(e)}")
            raise
    
    async def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of prediction dictionaries
        """
        if not self.is_loaded():
            raise Exception("Model not loaded. Call initialize() first.")
        
        try:
            # Run batch prediction in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                self.executor,
                self._predict_batch_sync,
                texts
            )
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            raise
    
    def _predict_batch_sync(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Synchronous batch prediction method"""
        try:
            results = []
            
            # Use model's batch prediction if available
            if hasattr(self.model, 'predict_batch'):
                predictions = self.model.predict_batch(texts)
                
                for i, text in enumerate(texts):
                    prediction = predictions[i]
                    probabilities = self.model.predict(text, return_probabilities=True)
                    
                    sentiment_label = self.label_mapping[prediction]
                    confidence = max(probabilities)
                    
                    prob_dict = {
                        "negative": float(probabilities[0]),
                        "neutral": float(probabilities[1]),
                        "positive": float(probabilities[2])
                    }
                    
                    results.append({
                        "sentiment": sentiment_label,
                        "confidence": float(confidence),
                        "probabilities": prob_dict
                    })
            else:
                # Fallback to individual predictions
                for text in texts:
                    result = self._predict_sync(text)
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Sync batch prediction error: {str(e)}")
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_loaded():
            return {"status": "not_loaded"}
        
        try:
            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(
                self.executor,
                self._get_model_info_sync
            )
            return info
            
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return {"error": str(e)}
    
    def _get_model_info_sync(self) -> Dict[str, Any]:
        """Get model information synchronously"""
        try:
            if hasattr(self.model, 'get_model_summary'):
                summary = self.model.get_model_summary()
            else:
                summary = {
                    "model_name": "PhoBERT Sentiment Classifier",
                    "num_labels": 3,
                    "labels": list(self.label_mapping.values())
                }
            
            # Add device info
            device = "cuda" if torch.cuda.is_available() else "cpu"
            summary["device"] = device
            summary["status"] = "loaded"
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting model info sync: {str(e)}")
            return {"error": str(e)}
    
    def __del__(self):
        """Cleanup when service is destroyed"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)