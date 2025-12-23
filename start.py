#!/usr/bin/env python
"""
Production start script for FastAPI backend on Render.
This script starts the uvicorn server without opening a browser.
"""
import uvicorn
import os

if __name__ == "__main__":
    # Get host and port from environment variables (Render sets PORT)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"🚀 Starting FastAPI server on {host}:{port}")
    
    # Run the server without browser auto-open
    uvicorn.run(
        "src.explainable_dbms.app:app",
        host=host,
        port=port,
        log_level="info"
    )
