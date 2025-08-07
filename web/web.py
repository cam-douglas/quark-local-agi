#!/usr/bin/env python3
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.response_generation_agent import ResponseGenerationAgent

class Query(BaseModel):
    question: str

app = FastAPI(title="Quark AI Assistant")

# Initialize response agent
response_agent = ResponseGenerationAgent()

@app.on_event("startup")
def load_models():
    """Initialize the response agent on startup"""
    print("🤖 Quark AI Assistant starting...")
    print("✅ Ready for questions!")

@app.post("/ask")
async def ask(q: Query):
    """POST {"question": "..."} → JSON response"""
    try:
        result = response_agent.generate(q.question)
        if isinstance(result, dict) and "response" in result:
            return {"answer": result["response"]}
        else:
            return {"answer": str(result)}
    except Exception as e:
        return {"answer": f"Sorry, I encountered an error: {str(e)}"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Quark AI Assistant",
        "status": "ready",
        "endpoints": {
            "ask": "POST /ask with {'question': 'your question'}"
        }
    }

def main():
    uvicorn.run(
        "web.web:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )

if __name__ == "__main__":
    main()

