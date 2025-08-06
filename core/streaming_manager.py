"""
STREAMING MANAGER
================

Handles real-time token generation and streaming responses for the AI assistant.
Provides progressive output, WebSocket support, and streaming capabilities.
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

import logging
logger = logging.getLogger(__name__)


@dataclass
class StreamChunk:
    """Represents a chunk of streaming data."""
    content: str
    chunk_type: str  # 'token', 'metadata', 'error', 'complete'
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class StreamingManager:
    """
    Manages real-time streaming of AI responses.
    """
    
    def __init__(self):
        """Initialize streaming manager."""
        self.active_streams = {}
        self.stream_config = {
            "chunk_size": 10,  # tokens per chunk
            "stream_delay": 0.05,  # seconds between chunks
            "max_stream_time": 300,  # max 5 minutes per stream
            "enable_progressive": True,
            "enable_metadata": True
        }
    
    def create_stream(self, stream_id: str, initial_context: Dict[str, Any] = None) -> str:
        """
        Create a new streaming session.
        
        Args:
            stream_id: Unique identifier for the stream
            initial_context: Initial context for the stream
            
        Returns:
            Stream ID
        """
        self.active_streams[stream_id] = {
            "created_at": time.time(),
            "context": initial_context or {},
            "chunks": [],
            "status": "active",
            "total_tokens": 0,
            "last_chunk_time": time.time()
        }
        
        logger.info(f"Created stream: {stream_id}")
        return stream_id
    
    def add_chunk(self, stream_id: str, content: str, chunk_type: str = "token", 
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a chunk to an active stream.
        
        Args:
            stream_id: Stream identifier
            content: Content of the chunk
            chunk_type: Type of chunk ('token', 'metadata', 'error', 'complete')
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        if stream_id not in self.active_streams:
            logger.error(f"Stream not found: {stream_id}")
            return False
        
        stream = self.active_streams[stream_id]
        
        chunk = StreamChunk(
            content=content,
            chunk_type=chunk_type,
            timestamp=time.time(),
            metadata=metadata
        )
        
        stream["chunks"].append(chunk)
        stream["total_tokens"] += len(content.split())
        stream["last_chunk_time"] = time.time()
        
        logger.debug(f"Added chunk to stream {stream_id}: {chunk_type} - {content[:50]}...")
        return True
    
    def get_stream_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a stream."""
        if stream_id not in self.active_streams:
            return None
        
        stream = self.active_streams[stream_id]
        return {
            "stream_id": stream_id,
            "status": stream["status"],
            "total_chunks": len(stream["chunks"]),
            "total_tokens": stream["total_tokens"],
            "created_at": stream["created_at"],
            "last_chunk_time": stream["last_chunk_time"],
            "duration": time.time() - stream["created_at"]
        }
    
    def close_stream(self, stream_id: str) -> bool:
        """Close an active stream."""
        if stream_id not in self.active_streams:
            return False
        
        self.active_streams[stream_id]["status"] = "closed"
        logger.info(f"Closed stream: {stream_id}")
        return True
    
    def cleanup_expired_streams(self) -> int:
        """Clean up expired streams."""
        current_time = time.time()
        expired_streams = []
        
        for stream_id, stream in self.active_streams.items():
            if current_time - stream["created_at"] > self.stream_config["max_stream_time"]:
                expired_streams.append(stream_id)
        
        for stream_id in expired_streams:
            del self.active_streams[stream_id]
        
        if expired_streams:
            logger.info(f"Cleaned up {len(expired_streams)} expired streams")
        
        return len(expired_streams)
    
    async def stream_response(self, prompt: str, model_response: str, 
                           stream_id: str = None) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream a response in real-time.
        
        Args:
            prompt: User prompt
            model_response: Full model response to stream
            stream_id: Optional stream ID
            
        Yields:
            StreamChunk objects
        """
        if not stream_id:
            stream_id = f"stream_{int(time.time())}"
        
        self.create_stream(stream_id, {"prompt": prompt})
        
        # Split response into tokens/words for streaming
        tokens = model_response.split()
        
        # Send initial metadata
        if self.stream_config["enable_metadata"]:
            yield StreamChunk(
                content="",
                chunk_type="metadata",
                timestamp=time.time(),
                metadata={
                    "total_tokens": len(tokens),
                    "estimated_time": len(tokens) * self.stream_config["stream_delay"],
                    "stream_id": stream_id
                }
            )
        
        # Stream tokens progressively
        for i, token in enumerate(tokens):
            # Add delay for realistic streaming effect
            await asyncio.sleep(self.stream_config["stream_delay"])
            
            chunk = StreamChunk(
                content=token + " ",
                chunk_type="token",
                timestamp=time.time(),
                metadata={"token_index": i, "total_tokens": len(tokens)}
            )
            
            self.add_chunk(stream_id, chunk.content, chunk.chunk_type, chunk.metadata)
            yield chunk
        
        # Send completion signal
        yield StreamChunk(
            content="",
            chunk_type="complete",
            timestamp=time.time(),
            metadata={"stream_id": stream_id, "final_token_count": len(tokens)}
        )
        
        self.close_stream(stream_id)
    
    def get_stream_history(self, stream_id: str) -> List[StreamChunk]:
        """Get complete history of a stream."""
        if stream_id not in self.active_streams:
            return []
        
        return self.active_streams[stream_id]["chunks"]
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs."""
        return [stream_id for stream_id, stream in self.active_streams.items() 
                if stream["status"] == "active"]
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming statistics."""
        active_count = len(self.get_active_streams())
        total_chunks = sum(len(stream["chunks"]) for stream in self.active_streams.values())
        total_tokens = sum(stream["total_tokens"] for stream in self.active_streams.values())
        
        return {
            "active_streams": active_count,
            "total_streams": len(self.active_streams),
            "total_chunks": total_chunks,
            "total_tokens": total_tokens,
            "config": self.stream_config
        }


class ProgressiveOutput:
    """
    Handles progressive output generation for real-time responses.
    """
    
    def __init__(self, streaming_manager: StreamingManager):
        """Initialize progressive output handler."""
        self.streaming_manager = streaming_manager
        self.output_buffer = {}
        self.progressive_config = {
            "buffer_size": 1000,
            "flush_interval": 0.1,
            "enable_typing_indicators": True
        }
    
    async def generate_progressive_response(self, prompt: str, 
                                         response_generator: callable) -> AsyncGenerator[str, None]:
        """
        Generate progressive response with real-time output.
        
        Args:
            prompt: User prompt
            response_generator: Function that generates response
            
        Yields:
            Progressive response chunks
        """
        stream_id = f"progressive_{int(time.time())}"
        
        # Start typing indicator
        if self.progressive_config["enable_typing_indicators"]:
            yield "🤖 Thinking..."
            await asyncio.sleep(0.5)
        
        # Generate response progressively
        try:
            async for chunk in response_generator(prompt):
                yield chunk
                await asyncio.sleep(0.05)  # Small delay for realistic typing
                
        except Exception as e:
            yield f"❌ Error: {str(e)}"
            logger.error(f"Error in progressive response: {e}")
    
    def add_to_buffer(self, stream_id: str, content: str):
        """Add content to output buffer."""
        if stream_id not in self.output_buffer:
            self.output_buffer[stream_id] = ""
        
        self.output_buffer[stream_id] += content
        
        # Flush if buffer is full
        if len(self.output_buffer[stream_id]) > self.progressive_config["buffer_size"]:
            self.flush_buffer(stream_id)
    
    def flush_buffer(self, stream_id: str) -> str:
        """Flush buffer and return content."""
        if stream_id not in self.output_buffer:
            return ""
        
        content = self.output_buffer[stream_id]
        self.output_buffer[stream_id] = ""
        return content


# Global streaming manager instance
STREAMING_MANAGER = StreamingManager()


def get_streaming_manager() -> StreamingManager:
    """Get the global streaming manager instance."""
    return STREAMING_MANAGER


async def stream_ai_response(prompt: str, response: str, stream_id: str = None) -> AsyncGenerator[StreamChunk, None]:
    """Stream an AI response in real-time."""
    async for chunk in STREAMING_MANAGER.stream_response(prompt, response, stream_id):
        yield chunk


def create_stream_session(initial_context: Dict[str, Any] = None) -> str:
    """Create a new streaming session."""
    return STREAMING_MANAGER.create_stream(f"session_{int(time.time())}", initial_context)


def get_streaming_stats() -> Dict[str, Any]:
    """Get streaming statistics."""
    return STREAMING_MANAGER.get_streaming_stats() 