"""
FASTAPI APPLICATION
==================

FastAPI web application for Quark AI Assistant.
Provides REST endpoints, WebSocket support, and API documentation.
Enhanced for Pillar 9: Streaming & Real-Time I/O
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
from agents.streaming_agent import StreamingAgent, StreamEvent
from dataclasses import asdict

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Quark AI Assistant API",
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
streaming_agent = StreamingAgent(model_name="streaming_agent")

# Pydantic models for API requests/responses
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    stream: bool = Field(False, description="Enable streaming response")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    user_id: Optional[str] = Field(None, description="User identifier")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI response")
    category: str = Field(..., description="Response category")
    stream_id: Optional[str] = Field(None, description="Stream ID if streaming")
    timestamp: str = Field(..., description="Response timestamp")
    safety_validated: bool = Field(..., description="Safety validation status")


class StreamRequest(BaseModel):
    stream_id: str = Field(..., description="Stream identifier")


class StreamSessionRequest(BaseModel):
    user_id: Optional[str] = Field(None, description="User identifier")
    context: Optional[Dict[str, Any]] = Field(None, description="Session context")


class StatusResponse(BaseModel):
    status: str = Field(..., description="Service status")
    active_connections: int = Field(..., description="Active WebSocket connections")
    streaming_stats: Dict[str, Any] = Field(..., description="Streaming statistics")
    safety_status: Dict[str, Any] = Field(..., description="Safety system status")
    session_stats: Dict[str, Any] = Field(..., description="Session statistics")


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Quark AI Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "websocket": "/ws",
        "streaming": "/stream",
        "pillar": "Pillar 9: Streaming & Real-Time I/O"
    }


@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "orchestrator": "active",
            "streaming_manager": "active",
            "websocket_manager": "active",
            "safety_enforcement": "active",
            "streaming_agent": "active"
        }
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with optional streaming."""
    try:
        # Safety validation
        safety_result = safety_enforcement.validate_input(request.message)
        if not safety_result["safe"]:
            raise HTTPException(status_code=400, detail=safety_result["reason"])
        
        # Process through orchestrator
        result = orchestrator.handle(request.message)
        
        # Safety validation of response
        response_safety = safety_enforcement.validate_response(result["response"])
        if not response_safety["safe"]:
            result["response"] = "I apologize, but I cannot provide that response."
        
        return ChatResponse(
            response=result["response"],
            category=result["category"],
            stream_id=None,
            timestamp=datetime.now().isoformat(),
            safety_validated=response_safety["safe"]
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint with real-time output."""
    try:
        # Safety validation
        safety_result = safety_enforcement.validate_input(request.message)
        if not safety_result["safe"]:
            raise HTTPException(status_code=400, detail=safety_result["reason"])
        
        # Create streaming session
        session_id = streaming_agent.create_session(
            user_id=request.user_id,
            context=request.context
        )
        
        async def generate_stream():
            """Generate streaming response."""
            try:
                # Process through orchestrator
                result = orchestrator.handle(request.message)
                
                # Safety validation
                response_safety = safety_enforcement.validate_response(result["response"])
                if not response_safety["safe"]:
                    result["response"] = "I apologize, but I cannot provide that response."
                
                # Stream the response
                async def response_generator(prompt: str):
                    return result["response"]
                
                async for event in streaming_agent.stream_response(
                    request.message, 
                    response_generator, 
                    session_id, 
                    request.user_id
                ):
                    yield f"data: {json.dumps(asdict(event))}\n\n"
                
                # Send completion
                yield f"data: {json.dumps({'type': 'complete', 'session_id': session_id})}\n\n"
                
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                error_event = {
                    "type": "error",
                    "content": str(e),
                    "session_id": session_id
                }
                yield f"data: {json.dumps(error_event)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Session-ID": session_id
            }
        )
        
    except Exception as e:
        logger.error(f"Error in streaming endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stream/session", response_model=Dict[str, Any])
async def create_stream_session(request: StreamSessionRequest):
    """Create a new streaming session."""
    try:
        session_id = streaming_agent.create_session(
            user_id=request.user_id,
            context=request.context
        )
        
        return {
            "session_id": session_id,
            "status": "created",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error creating stream session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stream/{stream_id}")
async def get_stream_status(stream_id: str):
    """Get status of a streaming session."""
    try:
        status = streaming_agent.get_session_status(stream_id)
        if not status:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        return {
            "stream_id": stream_id,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stream status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/stream/{stream_id}")
async def close_stream_session(stream_id: str):
    """Close a streaming session."""
    try:
        success = await streaming_agent.close_session(stream_id)
        if not success:
            raise HTTPException(status_code=404, detail="Stream not found")
        
        return {
            "stream_id": stream_id,
            "status": "closed",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing stream session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/streams")
async def get_all_streams():
    """Get all active streaming sessions."""
    try:
        stats = streaming_agent.get_streaming_stats()
        return {
            "active_sessions": stats["active_sessions"],
            "total_sessions": stats["total_sessions"],
            "streaming_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting streams: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status", response_model=StatusResponse)
async def get_status():
    """Get comprehensive system status."""
    try:
        # Get streaming statistics
        streaming_stats = streaming_agent.get_streaming_stats()
        
        # Get WebSocket statistics
        websocket_stats = websocket_manager.get_connection_stats()
        
        # Get safety status
        safety_status = safety_enforcement.get_status()
        
        return StatusResponse(
            status="operational",
            active_connections=websocket_stats.get("active_connections", 0),
            streaming_stats=streaming_stats,
            safety_status=safety_status,
            session_stats={
                "total_sessions": streaming_stats["total_sessions"],
                "active_sessions": streaming_stats["active_sessions"],
                "average_stream_time": streaming_stats["average_stream_time"]
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication."""
    await websocket.accept()
    
    try:
        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            "message": "Connected to Quark AI Assistant",
            "timestamp": datetime.now().isoformat(),
            "features": ["streaming", "real_time", "websocket"]
        }))
        
        # Handle incoming messages
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "chat":
                    await handle_websocket_chat(websocket, message)
                elif message.get("type") == "stream_request":
                    await handle_websocket_stream(websocket, message)
                elif message.get("type") == "status_request":
                    await handle_websocket_status(websocket, message)
                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Unknown message type",
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                }))
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")


async def handle_websocket_chat(websocket: WebSocket, data: Dict[str, Any]):
    """Handle chat messages via WebSocket."""
    try:
        message = data.get("message", "")
        user_id = data.get("user_id")
        
        # Safety validation
        safety_result = safety_enforcement.validate_input(message)
        if not safety_result["safe"]:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": safety_result["reason"],
                "timestamp": datetime.now().isoformat()
            }))
            return
        
        # Process through orchestrator
        result = orchestrator.handle(message)
        
        # Safety validation of response
        response_safety = safety_enforcement.validate_response(result["response"])
        if not response_safety["safe"]:
            result["response"] = "I apologize, but I cannot provide that response."
        
        # Send response
        await websocket.send_text(json.dumps({
            "type": "chat_response",
            "response": result["response"],
            "category": result["category"],
            "timestamp": datetime.now().isoformat(),
            "safety_validated": response_safety["safe"]
        }))
        
    except Exception as e:
        logger.error(f"Error handling WebSocket chat: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }))


async def handle_websocket_stream(websocket: WebSocket, data: Dict[str, Any]):
    """Handle streaming requests via WebSocket."""
    try:
        message = data.get("message", "")
        user_id = data.get("user_id")
        
        # Create streaming session
        session_id = streaming_agent.create_session(user_id)
        
        # Send session created message
        await websocket.send_text(json.dumps({
            "type": "stream_start",
            "session_id": session_id,
            "timestamp": datetime.now().isoformat()
        }))
        
        # Process and stream response
        result = orchestrator.handle(message)
        
        # Safety validation
        response_safety = safety_enforcement.validate_response(result["response"])
        if not response_safety["safe"]:
            result["response"] = "I apologize, but I cannot provide that response."
        
        # Stream response tokens
        tokens = result["response"].split()
        for i, token in enumerate(tokens):
            await websocket.send_text(json.dumps({
                "type": "stream_token",
                "content": token + " ",
                "token_index": i,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }))
            await asyncio.sleep(0.02)  # Small delay for realistic streaming
        
        # Send completion
        await websocket.send_text(json.dumps({
            "type": "stream_complete",
            "session_id": session_id,
            "total_tokens": len(tokens),
            "timestamp": datetime.now().isoformat()
        }))
        
    except Exception as e:
        logger.error(f"Error handling WebSocket stream: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }))


async def handle_websocket_status(websocket: WebSocket, data: Dict[str, Any]):
    """Handle status requests via WebSocket."""
    try:
        # Get comprehensive status
        streaming_stats = streaming_agent.get_streaming_stats()
        websocket_stats = websocket_manager.get_connection_stats()
        safety_status = safety_enforcement.get_status()
        
        await websocket.send_text(json.dumps({
            "type": "status_response",
            "streaming_stats": streaming_stats,
            "websocket_stats": websocket_stats,
            "safety_status": safety_status,
            "timestamp": datetime.now().isoformat()
        }))
        
    except Exception as e:
        logger.error(f"Error handling WebSocket status: {e}")
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }))


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Quark AI Assistant API starting up...")
    logger.info("Pillar 9: Streaming & Real-Time I/O enabled")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Quark AI Assistant API shutting down...")
    streaming_agent.shutdown()


if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 