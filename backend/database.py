"""
Database Manager for Sentiment Analysis Application
Handles data persistence and analytics
"""

import asyncio
import logging
import sqlite3
import aiosqlite
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages database operations for sentiment analysis data"""
    
    def __init__(self, db_path: str = "sentiment_analysis.db"):
        self.db_path = db_path
        self.connection = None
        
    async def initialize(self):
        """Initialize database and create tables"""
        try:
            logger.info("Initializing database...")
            
            # Create database and tables
            await self._create_tables()
            
            logger.info("Database initialized successfully!")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    async def _create_tables(self):
        """Create necessary database tables"""
        async with aiosqlite.connect(self.db_path) as db:
            # Analysis results table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    sentiment TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    probabilities TEXT NOT NULL,
                    user_id TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # User sessions table (for tracking)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT,
                    session_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    session_end TIMESTAMP,
                    total_analyses INTEGER DEFAULT 0
                )
            """)
            
            # Admin feedback table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS admin_feedback (
                    id TEXT PRIMARY KEY,
                    analysis_id TEXT,
                    feedback_type TEXT,
                    feedback_text TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (analysis_id) REFERENCES analyses (id)
                )
            """)
            
            # Create indexes for better performance
            await db.execute("CREATE INDEX IF NOT EXISTS idx_analyses_sentiment ON analyses(sentiment)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_analyses_created_at ON analyses(created_at)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses(user_id)")
            
            await db.commit()
    
    async def save_analysis(
        self,
        text: str,
        sentiment: str,
        confidence: float,
        probabilities: Dict[str, float],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save analysis result to database"""
        try:
            analysis_id = str(uuid.uuid4())
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO analyses (id, text, sentiment, confidence, probabilities, user_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis_id,
                    text,
                    sentiment,
                    confidence,
                    json.dumps(probabilities),
                    user_id,
                    json.dumps(metadata) if metadata else None
                ))
                await db.commit()
            
            return analysis_id
            
        except Exception as e:
            logger.error(f"Error saving analysis: {str(e)}")
            raise
    
    async def get_analysis_history(
        self,
        limit: int = 50,
        offset: int = 0,
        sentiment: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get analysis history with optional filtering"""
        try:
            query = "SELECT * FROM analyses WHERE 1=1"
            params = []
            
            if sentiment:
                query += " AND sentiment = ?"
                params.append(sentiment)
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    
                    results = []
                    for row in rows:
                        result = dict(row)
                        # Map database fields to expected API fields
                        result['analysis_id'] = result.pop('id')
                        result['timestamp'] = result.pop('created_at')
                        result['probabilities'] = json.loads(result['probabilities'])
                        if result['metadata']:
                            result['metadata'] = json.loads(result['metadata'])
                        else:
                            result['metadata'] = None
                        results.append(result)
                    
                    return results
            
        except Exception as e:
            logger.error(f"Error getting analysis history: {str(e)}")
            raise
    
    async def get_analytics(self) -> Dict[str, Any]:
        """Get analytics and statistics"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Total analyses
                async with db.execute("SELECT COUNT(*) as total FROM analyses") as cursor:
                    total_row = await cursor.fetchone()
                    total_analyses = total_row['total']
                
                # Sentiment distribution
                async with db.execute("""
                    SELECT sentiment, COUNT(*) as count 
                    FROM analyses 
                    GROUP BY sentiment
                """) as cursor:
                    sentiment_rows = await cursor.fetchall()
                    sentiment_distribution = {row['sentiment']: row['count'] for row in sentiment_rows}
                
                # Average confidence
                async with db.execute("SELECT AVG(confidence) as avg_confidence FROM analyses") as cursor:
                    avg_row = await cursor.fetchone()
                    average_confidence = avg_row['avg_confidence'] or 0.0
                
                # Recent analyses (last 10)
                async with db.execute("""
                    SELECT * FROM analyses 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """) as cursor:
                    recent_rows = await cursor.fetchall()
                    recent_analyses = []
                    for row in recent_rows:
                        analysis = dict(row)
                        # Map database fields to expected API fields
                        analysis['analysis_id'] = analysis.pop('id')
                        analysis['timestamp'] = analysis.pop('created_at')
                        analysis['probabilities'] = json.loads(analysis['probabilities'])
                        if analysis['metadata']:
                            analysis['metadata'] = json.loads(analysis['metadata'])
                        else:
                            analysis['metadata'] = None
                        recent_analyses.append(analysis)
                
                return {
                    "total_analyses": total_analyses,
                    "sentiment_distribution": sentiment_distribution,
                    "average_confidence": float(average_confidence),
                    "recent_analyses": recent_analyses
                }
            
        except Exception as e:
            logger.error(f"Error getting analytics: {str(e)}")
            raise
    
    async def get_admin_suggestions(self) -> List[Dict[str, Any]]:
        """Generate suggestions for admin based on recent analysis"""
        try:
            suggestions = []
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                # Check for low confidence predictions
                async with db.execute("""
                    SELECT COUNT(*) as low_confidence_count
                    FROM analyses 
                    WHERE confidence < 0.7 
                    AND created_at > datetime('now', '-24 hours')
                """) as cursor:
                    row = await cursor.fetchone()
                    if row['low_confidence_count'] > 5:
                        suggestions.append({
                            "type": "warning",
                            "title": "Low Confidence Predictions",
                            "message": f"Found {row['low_confidence_count']} predictions with low confidence in the last 24 hours. Consider reviewing the model or data quality.",
                            "action": "review_low_confidence"
                        })
                
                # Check sentiment distribution imbalance
                async with db.execute("""
                    SELECT sentiment, COUNT(*) as count
                    FROM analyses 
                    WHERE created_at > datetime('now', '-7 days')
                    GROUP BY sentiment
                """) as cursor:
                    sentiment_rows = await cursor.fetchall()
                    sentiment_counts = {row['sentiment']: row['count'] for row in sentiment_rows}
                    
                    total = sum(sentiment_counts.values())
                    if total > 0:
                        negative_ratio = sentiment_counts.get('negative', 0) / total
                        if negative_ratio > 0.7:
                            suggestions.append({
                                "type": "alert",
                                "title": "High Negative Sentiment",
                                "message": f"70%+ of recent feedback is negative. Consider immediate attention to customer concerns.",
                                "action": "review_negative_feedback"
                            })
                        elif negative_ratio < 0.1:
                            suggestions.append({
                                "type": "info",
                                "title": "Positive Trend",
                                "message": "Recent feedback shows very positive sentiment. Great job!",
                                "action": "maintain_quality"
                            })
                
                # Check for unusual activity patterns
                async with db.execute("""
                    SELECT COUNT(*) as recent_count
                    FROM analyses 
                    WHERE created_at > datetime('now', '-1 hour')
                """) as cursor:
                    row = await cursor.fetchone()
                    if row['recent_count'] > 100:
                        suggestions.append({
                            "type": "info",
                            "title": "High Activity",
                            "message": f"Unusual high activity detected: {row['recent_count']} analyses in the last hour.",
                            "action": "monitor_activity"
                        })
                
                # Default suggestion if no specific issues
                if not suggestions:
                    suggestions.append({
                        "type": "success",
                        "title": "System Running Smoothly",
                        "message": "No issues detected. System is performing well.",
                        "action": "continue_monitoring"
                    })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting admin suggestions: {str(e)}")
            return [{
                "type": "error",
                "title": "Error Getting Suggestions",
                "message": f"Unable to generate suggestions: {str(e)}",
                "action": "check_system"
            }]
    
    async def save_admin_feedback(
        self,
        analysis_id: str,
        feedback_type: str,
        feedback_text: str
    ) -> str:
        """Save admin feedback for an analysis"""
        try:
            feedback_id = str(uuid.uuid4())
            
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    INSERT INTO admin_feedback (id, analysis_id, feedback_type, feedback_text)
                    VALUES (?, ?, ?, ?)
                """, (feedback_id, analysis_id, feedback_type, feedback_text))
                await db.commit()
            
            return feedback_id
            
        except Exception as e:
            logger.error(f"Error saving admin feedback: {str(e)}")
            raise
    
    async def is_connected(self) -> bool:
        """Check if database is accessible"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("SELECT 1")
                return True
        except:
            return False
    
    async def close(self):
        """Close database connections"""
        # aiosqlite handles connection closing automatically
        logger.info("Database connections closed")
    
    async def get_daily_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get daily statistics for the last N days"""
        try:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                
                async with db.execute("""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as total_analyses,
                        AVG(confidence) as avg_confidence,
                        SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive_count,
                        SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative_count,
                        SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral_count
                    FROM analyses 
                    WHERE created_at > datetime('now', '-{} days')
                    GROUP BY DATE(created_at)
                    ORDER BY date DESC
                """.format(days)) as cursor:
                    rows = await cursor.fetchall()
                    
                    return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting daily stats: {str(e)}")
            raise