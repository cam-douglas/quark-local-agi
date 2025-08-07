#!/usr/bin/env python3
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from quark.orchestrator import Orchestrator

class Query(BaseModel):
    question: str

app = FastAPI(title="Meta Model AI Assistant")

# instantiate once
orch = Orchestrator()

@app.on_event("startup")
def load_models():
    # this blocks until all models are ready
    orch.wait_ready()

@app.post("/ask")
async def ask(q: Query):
    """POST {"question": "..."} → JSON response"""
    return orch.handle(q.question)

def main():
    uvicorn.run(
        "quark.web:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

if __name__ == "__main__":
    main()

