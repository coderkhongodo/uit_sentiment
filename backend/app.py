"""
FastAPI Backend for Vietnamese Sentiment Analysis
Integrates with PhoBERT model for real-time sentiment analysis
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import asyncio
from datetime import datetime
import logging
import sys
import os

# Add src directory to path to import our model
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_service import SentimentModelService
from database import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Vietnamese Sentiment Analysis API",
    description="Real-time sentiment analysis for Vietnamese text using PhoBERT",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
model_service = SentimentModelService()
db_manager = DatabaseManager()

# Pydantic models for request/response
class TextInput(BaseModel):
    text: str
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class BatchTextInput(BaseModel):
    texts: List[str]
    user_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: Dict[str, float]
    timestamp: datetime
    analysis_id: str

class BatchSentimentResponse(BaseModel):
    results: List[SentimentResponse]
    total_count: int
    processing_time: float

class AnalyticsResponse(BaseModel):
    total_analyses: int
    sentiment_distribution: Dict[str, int]
    average_confidence: float
    recent_analyses: List[SentimentResponse]

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove broken connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting Vietnamese Sentiment Analysis API...")
    
    # Initialize model service
    await model_service.initialize()
    
    # Initialize database
    await db_manager.initialize()
    
    logger.info("API startup completed successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API...")
    await db_manager.close()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "model_loaded": model_service.is_loaded(),
        "database_connected": await db_manager.is_connected()
    }

# Single text analysis
@app.post("/analyze", response_model=SentimentResponse)
async def analyze_sentiment(input_data: TextInput):
    """Analyze sentiment of a single text"""
    try:
        if not input_data.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Perform sentiment analysis
        result = await model_service.predict(input_data.text)
        
        # Save to database
        analysis_id = await db_manager.save_analysis(
            text=input_data.text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            user_id=input_data.user_id,
            metadata=input_data.metadata
        )
        
        response = SentimentResponse(
            text=input_data.text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            timestamp=datetime.now(),
            analysis_id=analysis_id
        )
        
        # Broadcast to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "new_analysis",
            "data": response.dict()
        }, default=str))
        
        return response
        
    except Exception as e:
        logger.error(f"Error in sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

# Batch text analysis
@app.post("/analyze/batch", response_model=BatchSentimentResponse)
async def analyze_batch_sentiment(input_data: BatchTextInput):
    """Analyze sentiment of multiple texts"""
    try:
        if not input_data.texts or len(input_data.texts) == 0:
            raise HTTPException(status_code=400, detail="Texts list cannot be empty")
        
        if len(input_data.texts) > 100:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size cannot exceed 100 texts")
        
        start_time = datetime.now()
        
        # Perform batch analysis
        results = await model_service.predict_batch(input_data.texts)
        
        # Save all results to database
        response_results = []
        for i, (text, result) in enumerate(zip(input_data.texts, results)):
            analysis_id = await db_manager.save_analysis(
                text=text,
                sentiment=result["sentiment"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                user_id=input_data.user_id,
                metadata=input_data.metadata
            )
            
            response_results.append(SentimentResponse(
                text=text,
                sentiment=result["sentiment"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                timestamp=datetime.now(),
                analysis_id=analysis_id
            ))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = BatchSentimentResponse(
            results=response_results,
            total_count=len(response_results),
            processing_time=processing_time
        )
        
        # Broadcast batch results
        await manager.broadcast(json.dumps({
            "type": "batch_analysis",
            "data": {
                "count": len(response_results),
                "processing_time": processing_time
            }
        }, default=str))
        
        return response
        
    except Exception as e:
        logger.error(f"Error in batch sentiment analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

# Analytics endpoint
@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get analytics and statistics"""
    try:
        analytics = await db_manager.get_analytics()
        return AnalyticsResponse(**analytics)
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analytics failed: {str(e)}")

# Get analysis history
@app.get("/history")
async def get_analysis_history(
    limit: int = 50,
    offset: int = 0,
    sentiment: Optional[str] = None,
    user_id: Optional[str] = None
):
    """Get analysis history with optional filtering"""
    try:
        history = await db_manager.get_analysis_history(
            limit=limit,
            offset=offset,
            sentiment=sentiment,
            user_id=user_id
        )
        return history
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")

# Admin suggestions endpoint
@app.get("/admin/suggestions")
async def get_admin_suggestions():
    """Get suggestions for admin based on recent analysis"""
    try:
        suggestions = await db_manager.get_admin_suggestions()
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error(f"Error getting admin suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Suggestions failed: {str(e)}")

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time sentiment analysis updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "analyze":
                # Real-time analysis via WebSocket
                text = message.get("text", "")
                if text.strip():
                    result = await model_service.predict(text)
                    await websocket.send_text(json.dumps({
                        "type": "analysis_result",
                        "data": {
                            "text": text,
                            "sentiment": result["sentiment"],
                            "confidence": result["confidence"],
                            "probabilities": result["probabilities"],
                            "timestamp": datetime.now().isoformat()
                        }
                    }))
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)