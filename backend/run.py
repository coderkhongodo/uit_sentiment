"""
Development server runner for FastAPI backend
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Add current directory to Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Run the FastAPI application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )