"""
WEBSOCKET MANAGER
================

Handles WebSocket communication for real-time AI assistant interactions.
Provides live streaming, interactive communication, and WebSocket endpoints.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import websockets
from websockets.server import WebSocketServerProtocol

from .streaming_manager import get_streaming_manager, StreamChunk
from .safety_enforcement import get_safety_enforcement

logger = logging.getLogger(__name__)


class WebSocketManager:
    """
    Manages WebSocket connections and real-time communication.
    """
    
    def __init__(self):
        """Initialize WebSocket manager."""
        self.active_connections = {}
        self.connection_handlers = {}
        self.streaming_manager = get_streaming_manager()
        self.safety_enforcement = get_safety_enforcement()
        
        self.websocket_config = {
            "max_connections": 100,
            "connection_timeout": 300,  # 5 minutes
            "heartbeat_interval": 30,   # 30 seconds
            "enable_compression": True,
            "max_message_size": 1024 * 1024  # 1MB
        }
    
    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle a new WebSocket connection.
        
        Args:
            websocket: WebSocket connection object
            path: Connection path
        """
        connection_id = f"ws_{int(datetime.now().timestamp())}"
        
        try:
            # Register connection
            self.active_connections[connection_id] = {
                "websocket": websocket,
                "path": path,
                "connected_at": datetime.now(),
                "last_heartbeat": datetime.now(),
                "status": "connected"
            }
            
            logger.info(f"WebSocket connection established: {connection_id}")
            
            # Send welcome message
            await self._send_message(websocket, {
                "type": "connection_established",
                "connection_id": connection_id,
                "timestamp": datetime.now().isoformat(),
                "message": "Connected to Meta-Model AI Assistant"
            })
            
            # Handle incoming messages
            async for message in websocket:
                await self._handle_message(connection_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"WebSocket error for {connection_id}: {e}")
        finally:
            # Clean up connection
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
    
    async def _handle_message(self, connection_id: str, message: str):
        """
        Handle incoming WebSocket message.
        
        Args:
            connection_id: Connection identifier
            message: Incoming message
        """
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")
            
            if message_type == "chat":
                await self._handle_chat_message(connection_id, data)
            elif message_type == "stream_request":
                await self._handle_stream_request(connection_id, data)
            elif message_type == "heartbeat":
                await self._handle_heartbeat(connection_id, data)
            elif message_type == "status_request":
                await self._handle_status_request(connection_id, data)
            else:
                await self._send_error(connection_id, f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError:
            await self._send_error(connection_id, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self._send_error(connection_id, f"Internal error: {str(e)}")
    
    async def _handle_chat_message(self, connection_id: str, data: Dict[str, Any]):
        """Handle chat message from client."""
        prompt = data.get("message", "")
        stream_id = data.get("stream_id")
        
        if not prompt:
            await self._send_error(connection_id, "Empty message")
            return
        
        # Safety validation
        safety_result = self.safety_enforcement.validate_action("process_chat_message", {
            "prompt": prompt,
            "connection_id": connection_id
        })
        
        if not safety_result["safe"]:
            await self._send_error(connection_id, f"Message blocked: {safety_result['reason']}")
            return
        
        # Process with AI (this would integrate with the orchestrator)
        try:
            # Simulate AI response for now
            ai_response = f"I received your message: '{prompt}'. This is a simulated response."
            
            # Stream the response
            await self._stream_response(connection_id, prompt, ai_response, stream_id)
            
        except Exception as e:
            logger.error(f"Error processing chat message: {e}")
            await self._send_error(connection_id, f"Error processing message: {str(e)}")
    
    async def _handle_stream_request(self, connection_id: str, data: Dict[str, Any]):
        """Handle streaming request from client."""
        stream_id = data.get("stream_id")
        
        if not stream_id:
            await self._send_error(connection_id, "No stream ID provided")
            return
        
        # Get stream status
        stream_status = self.streaming_manager.get_stream_status(stream_id)
        
        if not stream_status:
            await self._send_error(connection_id, f"Stream not found: {stream_id}")
            return
        
        await self._send_message(self.active_connections[connection_id]["websocket"], {
            "type": "stream_status",
            "stream_id": stream_id,
            "status": stream_status,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_heartbeat(self, connection_id: str, data: Dict[str, Any]):
        """Handle heartbeat message from client."""
        if connection_id in self.active_connections:
            self.active_connections[connection_id]["last_heartbeat"] = datetime.now()
        
        await self._send_message(self.active_connections[connection_id]["websocket"], {
            "type": "heartbeat_ack",
            "timestamp": datetime.now().isoformat()
        })
    
    async def _handle_status_request(self, connection_id: str, data: Dict[str, Any]):
        """Handle status request from client."""
        status = {
            "active_connections": len(self.active_connections),
            "streaming_stats": self.streaming_manager.get_streaming_stats(),
            "server_time": datetime.now().isoformat()
        }
        
        await self._send_message(self.active_connections[connection_id]["websocket"], {
            "type": "status_response",
            "status": status,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _stream_response(self, connection_id: str, prompt: str, response: str, 
                             stream_id: str = None):
        """
        Stream AI response to WebSocket client.
        
        Args:
            connection_id: WebSocket connection ID
            prompt: User prompt
            response: AI response to stream
            stream_id: Optional stream ID
        """
        if connection_id not in self.active_connections:
            return
        
        websocket = self.active_connections[connection_id]["websocket"]
        
        # Send streaming start notification
        await self._send_message(websocket, {
            "type": "stream_start",
            "stream_id": stream_id,
            "timestamp": datetime.now().isoformat()
        })
        
        # Stream response chunks
        async for chunk in self.streaming_manager.stream_response(prompt, response, stream_id):
            await self._send_message(websocket, {
                "type": "stream_chunk",
                "stream_id": stream_id,
                "chunk_type": chunk.chunk_type,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "timestamp": datetime.now().isoformat()
            })
        
        # Send streaming end notification
        await self._send_message(websocket, {
            "type": "stream_end",
            "stream_id": stream_id,
            "timestamp": datetime.now().isoformat()
        })
    
    async def _send_message(self, websocket: WebSocketServerProtocol, message: Dict[str, Any]):
        """Send message to WebSocket client."""
        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
    
    async def _send_error(self, connection_id: str, error_message: str):
        """Send error message to WebSocket client."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]["websocket"]
            await self._send_message(websocket, {
                "type": "error",
                "message": error_message,
                "timestamp": datetime.now().isoformat()
            })
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        active_connections = len([conn for conn in self.active_connections.values() 
                               if conn["status"] == "connected"])
        
        return {
            "active_connections": active_connections,
            "total_connections": len(self.active_connections),
            "max_connections": self.websocket_config["max_connections"],
            "config": self.websocket_config
        }
    
    async def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients."""
        disconnected = []
        
        for connection_id, connection in self.active_connections.items():
            try:
                await self._send_message(connection["websocket"], message)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                disconnected.append(connection_id)
        
        # Clean up disconnected connections
        for connection_id in disconnected:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
    
    async def cleanup_expired_connections(self):
        """Clean up expired connections."""
        current_time = datetime.now()
        expired = []
        
        for connection_id, connection in self.active_connections.items():
            time_diff = (current_time - connection["last_heartbeat"]).total_seconds()
            if time_diff > self.websocket_config["connection_timeout"]:
                expired.append(connection_id)
        
        for connection_id in expired:
            try:
                await self.active_connections[connection_id]["websocket"].close()
            except Exception as e:
                logger.error(f"Error closing expired connection {connection_id}: {e}")
            finally:
                del self.active_connections[connection_id]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired WebSocket connections")


# Global WebSocket manager instance
WEBSOCKET_MANAGER = WebSocketManager()


def get_websocket_manager() -> WebSocketManager:
    """Get the global WebSocket manager instance."""
    return WEBSOCKET_MANAGER


async def start_websocket_server(host: str = "localhost", port: int = 8765):
    """Start WebSocket server."""
    server = await websockets.serve(
        WEBSOCKET_MANAGER.handle_connection,
        host,
        port
    )
    
    logger.info(f"WebSocket server started on ws://{host}:{port}")
    
    # Start cleanup task
    asyncio.create_task(_cleanup_task())
    
    return server


async def _cleanup_task():
    """Background task for cleaning up expired connections."""
    while True:
        await asyncio.sleep(60)  # Run every minute
        await WEBSOCKET_MANAGER.cleanup_expired_connections()
        WEBSOCKET_MANAGER.streaming_manager.cleanup_expired_streams() 