"""
FASTAPI APPLICATION
==================

FastAPI web application for Meta-Model AI Assistant.
Provides REST endpoints, WebSocket support, and API documentation.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.orchestrator import Orchestrator
from core.streaming_manager import get_streaming_manager, StreamChunk
from core.websocket_manager import get_websocket_manager
from core.safety_enforcement import get_safety_enforcement

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Meta-Model AI Assistant API",
    description="Real-time AI assistant with streaming capabilities",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
orchestrator = Orchestrator()
streaming_manager = get_streaming_manager()
websocket_manager = get_websocket_manager()
safety_enforcement = get_safety_enforcement()


# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    stream: bool = Field(False, description="Enable streaming response")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    category: str = Field(..., description="Response category")
    stream_id: Optional[str] = Field(None, description="Stream ID if streaming")
    timestamp: str = Field(..., description="Response timestamp")
    safety_validated: bool = Field(..., description="Safety validation status")


class StreamRequest(BaseModel):
    stream_id: str = Field(..., description="Stream identifier")


class StatusResponse(BaseModel):
    status: str = Field(..., description="Service status")
    active_connections: int = Field(..., description="Active WebSocket connections")
    streaming_stats: Dict[str, Any] = Field(..., description="Streaming statistics")
    safety_status: Dict[str, Any] = Field(..., description="Safety system status")


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Meta-Model AI Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "websocket": "/ws"
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "orchestrator": "active",
            "streaming": "active",
            "websocket": "active",
            "safety": "active"
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Chat endpoint for AI interactions.
    
    Args:
        request: Chat request with message and options
        
    Returns:
        AI response with metadata
    """
    try:
        # Safety validation
        safety_result = safety_enforcement.validate_action("process_chat_request", {
            "message": request.message,
            "stream": request.stream
        })
        
        if not safety_result["safe"]:
            raise HTTPException(status_code=400, detail=f"Request blocked: {safety_result['reason']}")
        
        # Process with orchestrator
        result = orchestrator.handle(request.message)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Extract response
        response_text = ""
        if "results" in result and "Reasoning" in result["results"]:
            response_text = result["results"]["Reasoning"]
        
        # Create stream if requested
        stream_id = None
        if request.stream:
            stream_id = streaming_manager.create_stream(
                f"api_stream_{int(datetime.now().timestamp())}",
                {"message": request.message}
            )
        
        return ChatResponse(
            response=response_text,
            category=result.get("category", "Unknown"),
            stream_id=stream_id,
            timestamp=datetime.now().isoformat(),
            safety_validated=result.get("safety_validated", True)
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses.
    
    Args:
        request: Chat request with message
        
    Returns:
        Streaming response
    """
    try:
        # Safety validation
        safety_result = safety_enforcement.validate_action("process_streaming_chat", {
            "message": request.message
        })
        
        if not safety_result["safe"]:
            raise HTTPException(status_code=400, detail=f"Request blocked: {safety_result['reason']}")
        
        # Process with orchestrator
        result = orchestrator.handle(request.message)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Extract response
        response_text = ""
        if "results" in result and "Reasoning" in result["results"]:
            response_text = result["results"]["Reasoning"]
        
        # Create streaming response
        async def generate_stream():
            async for chunk in streaming_manager.stream_response(request.message, response_text):
                yield f"data: {json.dumps({
                    'type': chunk.chunk_type,
                    'content': chunk.content,
                    'metadata': chunk.metadata,
                    'timestamp': datetime.now().isoformat()
                })}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/stream/{stream_id}")
async def get_stream_status(stream_id: str):
    """Get status of a specific stream."""
    status = streaming_manager.get_stream_status(stream_id)
    
    if not status:
        raise HTTPException(status_code=404, detail=f"Stream not found: {stream_id}")
    
    return status


@app.get("/streams")
async def get_all_streams():
    """Get all active streams."""
    return {
        "active_streams": streaming_manager.get_active_streams(),
        "streaming_stats": streaming_manager.get_streaming_stats()
    }


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get comprehensive system status."""
    return StatusResponse(
        status="active",
        active_connections=len(websocket_manager.active_connections),
        streaming_stats=streaming_manager.get_streaming_stats(),
        safety_status=safety_enforcement.get_safety_report()
    )


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": "Connected to Meta-Model AI Assistant",
            "timestamp": datetime.now().isoformat()
        }))
        
        # Handle incoming messages
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)
            
            # Route to appropriate handler
            if data.get("type") == "chat":
                await handle_websocket_chat(websocket, data)
            elif data.get("type") == "status_request":
                await handle_websocket_status(websocket, data)
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Unknown message type: {data.get('type')}",
                    "timestamp": datetime.now().isoformat()
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Internal error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }))
        except:
            pass


async def handle_websocket_chat(websocket: WebSocket, data: Dict[str, Any]):
    """Handle chat message via WebSocket."""
    message = data.get("message", "")
    
    if not message:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": "Empty message",
            "timestamp": datetime.now().isoformat()
        }))
        return
    
    try:
        # Process with orchestrator
        result = orchestrator.handle(message)
        
        if "error" in result:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": result["error"],
                "timestamp": datetime.now().isoformat()
            }))
            return
        
        # Extract response
        response_text = ""
        if "results" in result and "Reasoning" in result["results"]:
            response_text = result["results"]["Reasoning"]
        
        # Stream response
        await websocket.send_text(json.dumps({
            "type": "stream_start",
            "timestamp": datetime.now().isoformat()
        }))
        
        async for chunk in streaming_manager.stream_response(message, response_text):
            await websocket.send_text(json.dumps({
                "type": "stream_chunk",
                "chunk_type": chunk.chunk_type,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "timestamp": datetime.now().isoformat()
            }))
        
        await websocket.send_text(json.dumps({
            "type": "stream_end",
            "timestamp": datetime.now().isoformat()
        }))
        
    except Exception as e:
        logger.error(f"Error handling WebSocket chat: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Internal error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }))


async def handle_websocket_status(websocket: WebSocket, data: Dict[str, Any]):
    """Handle status request via WebSocket."""
    status = {
        "active_connections": len(websocket_manager.active_connections),
        "streaming_stats": streaming_manager.get_streaming_stats(),
        "server_time": datetime.now().isoformat()
    }
    
    await websocket.send_text(json.dumps({
        "type": "status_response",
        "status": status,
        "timestamp": datetime.now().isoformat()
    }))


# Background tasks
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    logger.info("Starting Meta-Model AI Assistant API")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Meta-Model AI Assistant API")


if __name__ == "__main__":
    uvicorn.run(
        "web.fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 