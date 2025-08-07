#!/usr/bin/env python3
"""
Streaming Agent for Quark AI Assistant
Handles real-time I/O, progressive output, and streaming capabilities for Pillar 9
"""

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from .base import Agent

logger = logging.getLogger(__name__)

@dataclass
class StreamEvent:
    """A streaming event with metadata."""
    event_id: str
    event_type: str  # 'token', 'metadata', 'error', 'complete', 'heartbeat'
    content: str
    timestamp: float
    stream_id: str
    metadata: Dict[str, Any] = None

@dataclass
class StreamSession:
    """A streaming session with state management."""
    session_id: str
    user_id: Optional[str]
    created_at: float
    last_activity: float
    status: str  # 'active', 'paused', 'completed', 'error'
    total_tokens: int
    events: List[StreamEvent]
    context: Dict[str, Any]

class StreamingAgent(Agent):
    """
    Streaming Agent for real-time I/O and progressive output.
    
    Features:
    - Real-time token streaming
    - Progressive output generation
    - WebSocket communication
    - Stream session management
    - Event-driven architecture
    - Performance monitoring
    """
    
    def __init__(self, model_name: str = "streaming_agent"):
        super().__init__(model_name)
        # Initialize without model since this is a streaming agent
        self.model = None
        
        # Streaming configuration
        self.stream_config = {
            'chunk_size': 5,  # tokens per chunk
            'stream_delay': 0.02,  # seconds between chunks
            'max_stream_time': 600,  # 10 minutes max
            'enable_progressive': True,
            'enable_metadata': True,
            'compression_enabled': True,
            'heartbeat_interval': 30  # seconds
        }
        
        # Active sessions
        self.active_sessions = {}
        self.session_handlers = {}
        
        # Performance tracking
        self.streaming_stats = {
            'total_sessions': 0,
            'active_sessions': 0,
            'total_tokens_streamed': 0,
            'average_stream_time': 0.0,
            'error_count': 0
        }
        
        # Event handlers
        self.event_handlers = {
            'token': self._handle_token_event,
            'metadata': self._handle_metadata_event,
            'error': self._handle_error_event,
            'complete': self._handle_complete_event,
            'heartbeat': self._handle_heartbeat_event
        }
        
        # Background tasks
        self.cleanup_task = None
        self.heartbeat_task = None
        
        # Start background tasks
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks for session management."""
        try:
            # Check if event loop is running
            loop = asyncio.get_running_loop()
            
            # Start cleanup task
            asyncio.create_task(self._cleanup_expired_sessions())
            
            # Start heartbeat task
            asyncio.create_task(self._send_heartbeats())
            
            logger.info("Background tasks started for streaming agent")
        except RuntimeError:
            # No running event loop, skip async task creation
            logger.info("No running event loop, skipping background tasks")
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def _cleanup_expired_sessions(self):
        """Clean up expired streaming sessions."""
        while True:
            try:
                current_time = time.time()
                expired_sessions = []
                
                for session_id, session in self.active_sessions.items():
                    # Check if session has expired
                    if (current_time - session.last_activity) > self.stream_config['max_stream_time']:
                        expired_sessions.append(session_id)
                    elif session.status == 'completed':
                        expired_sessions.append(session_id)
                
                # Remove expired sessions
                for session_id in expired_sessions:
                    await self.close_session(session_id)
                    logger.info(f"Cleaned up expired session: {session_id}")
                
                # Update stats
                self.streaming_stats['active_sessions'] = len(self.active_sessions)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    async def _send_heartbeats(self):
        """Send heartbeat messages to active sessions."""
        while True:
            try:
                current_time = time.time()
                heartbeat_interval = self.stream_config['heartbeat_interval']
                
                for session_id, session in self.active_sessions.items():
                    if session.status == 'active':
                        # Send heartbeat event
                        heartbeat_event = StreamEvent(
                            event_id=str(uuid.uuid4()),
                            event_type='heartbeat',
                            content='',
                            timestamp=current_time,
                            stream_id=session_id,
                            metadata={'session_id': session_id}
                        )
                        
                        # Add to session events
                        session.events.append(heartbeat_event)
                        session.last_activity = current_time
                        
                        # Notify handlers
                        await self._notify_handlers(session_id, heartbeat_event)
                
                await asyncio.sleep(heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in heartbeat task: {e}")
                await asyncio.sleep(30)
    
    def create_session(self, user_id: Optional[str] = None, context: Dict[str, Any] = None) -> str:
        """Create a new streaming session."""
        session_id = f"stream_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        current_time = time.time()
        
        session = StreamSession(
            session_id=session_id,
            user_id=user_id,
            created_at=current_time,
            last_activity=current_time,
            status='active',
            total_tokens=0,
            events=[],
            context=context or {}
        )
        
        self.active_sessions[session_id] = session
        self.streaming_stats['total_sessions'] += 1
        self.streaming_stats['active_sessions'] += 1
        
        logger.info(f"Created streaming session: {session_id}")
        return session_id
    
    async def stream_response(self, prompt: str, response_generator: Callable, 
                           session_id: str = None, user_id: Optional[str] = None) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream a response using progressive output.
        
        Args:
            prompt: Input prompt
            response_generator: Function that generates response tokens
            session_id: Optional session ID
            user_id: Optional user ID
            
        Yields:
            StreamEvent objects
        """
        if not session_id:
            session_id = self.create_session(user_id)
        
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session not found: {session_id}")
        
        try:
            session.status = 'active'
            session.last_activity = time.time()
            
            # Send start event
            start_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type='metadata',
                content='',
                timestamp=time.time(),
                stream_id=session_id,
                metadata={'action': 'stream_start', 'prompt': prompt}
            )
            
            session.events.append(start_event)
            yield start_event
            
            # Stream response tokens
            async for token in self._generate_tokens(response_generator, prompt):
                # Create token event
                token_event = StreamEvent(
                    event_id=str(uuid.uuid4()),
                    event_type='token',
                    content=token,
                    timestamp=time.time(),
                    stream_id=session_id,
                    metadata={'token_index': session.total_tokens}
                )
                
                session.events.append(token_event)
                session.total_tokens += 1
                session.last_activity = time.time()
                self.streaming_stats['total_tokens_streamed'] += 1
                
                yield token_event
                
                # Add delay for realistic streaming
                await asyncio.sleep(self.stream_config['stream_delay'])
            
            # Send complete event
            complete_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type='complete',
                content='',
                timestamp=time.time(),
                stream_id=session_id,
                metadata={'total_tokens': session.total_tokens}
            )
            
            session.events.append(complete_event)
            session.status = 'completed'
            yield complete_event
            
        except Exception as e:
            logger.error(f"Error streaming response for session {session_id}: {e}")
            
            # Send error event
            error_event = StreamEvent(
                event_id=str(uuid.uuid4()),
                event_type='error',
                content=str(e),
                timestamp=time.time(),
                stream_id=session_id,
                metadata={'error_type': type(e).__name__}
            )
            
            session.events.append(error_event)
            session.status = 'error'
            self.streaming_stats['error_count'] += 1
            yield error_event
    
    async def _generate_tokens(self, response_generator: Callable, prompt: str) -> AsyncGenerator[str, None]:
        """Generate tokens using the response generator."""
        try:
            # Get full response first (simplified for now)
            response = await response_generator(prompt)
            
            # Split into tokens and yield progressively
            tokens = response.split()
            chunk_size = self.stream_config['chunk_size']
            
            for i in range(0, len(tokens), chunk_size):
                chunk = tokens[i:i + chunk_size]
                yield ' '.join(chunk) + ' '
                
        except Exception as e:
            logger.error(f"Error generating tokens: {e}")
            yield f"Error: {str(e)}"
    
    async def _notify_handlers(self, session_id: str, event: StreamEvent):
        """Notify registered handlers of an event."""
        if session_id in self.session_handlers:
            for handler in self.session_handlers[session_id]:
                try:
                    await handler(event)
                except Exception as e:
                    logger.error(f"Error in session handler: {e}")
    
    def register_handler(self, session_id: str, handler: Callable):
        """Register an event handler for a session."""
        if session_id not in self.session_handlers:
            self.session_handlers[session_id] = []
        self.session_handlers[session_id].append(handler)
    
    def unregister_handler(self, session_id: str, handler: Callable):
        """Unregister an event handler."""
        if session_id in self.session_handlers and handler in self.session_handlers[session_id]:
            self.session_handlers[session_id].remove(handler)
    
    def close_session_sync(self, session_id: str) -> bool:
        """Close a streaming session synchronously."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.status = 'completed'
        session.last_activity = time.time()
        
        # Send close event
        close_event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type='metadata',
            content='',
            timestamp=time.time(),
            stream_id=session_id,
            metadata={'action': 'session_closed'}
        )
        
        session.events.append(close_event)
        
        # Clean up handlers
        if session_id in self.session_handlers:
            del self.session_handlers[session_id]
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        self.streaming_stats['active_sessions'] -= 1
        
        logger.info(f"Closed streaming session: {session_id}")
        return True

    async def close_session(self, session_id: str) -> bool:
        """Close a streaming session."""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        session.status = 'completed'
        session.last_activity = time.time()
        
        # Send close event
        close_event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type='metadata',
            content='',
            timestamp=time.time(),
            stream_id=session_id,
            metadata={'action': 'session_closed'}
        )
        
        session.events.append(close_event)
        await self._notify_handlers(session_id, close_event)
        
        # Clean up handlers
        if session_id in self.session_handlers:
            del self.session_handlers[session_id]
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        self.streaming_stats['active_sessions'] -= 1
        
        logger.info(f"Closed streaming session: {session_id}")
        return True
    
    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a streaming session."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'status': session.status,
            'created_at': session.created_at,
            'last_activity': session.last_activity,
            'total_tokens': session.total_tokens,
            'event_count': len(session.events),
            'context': session.context
        }
    
    def get_session_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all events for a session."""
        if session_id not in self.active_sessions:
            return []
        
        session = self.active_sessions[session_id]
        return [asdict(event) for event in session.events]
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get comprehensive streaming statistics."""
        current_time = time.time()
        
        # Calculate average stream time
        completed_sessions = [s for s in self.active_sessions.values() if s.status == 'completed']
        if completed_sessions:
            avg_time = sum(current_time - s.created_at for s in completed_sessions) / len(completed_sessions)
        else:
            avg_time = 0.0
        
        return {
            **self.streaming_stats,
            'average_stream_time': avg_time,
            'config': self.stream_config,
            'session_count': len(self.active_sessions)
        }
    
    async def _handle_token_event(self, event: StreamEvent):
        """Handle token events."""
        logger.debug(f"Token event: {event.content[:50]}...")
    
    async def _handle_metadata_event(self, event: StreamEvent):
        """Handle metadata events."""
        logger.debug(f"Metadata event: {event.metadata}")
    
    async def _handle_error_event(self, event: StreamEvent):
        """Handle error events."""
        logger.error(f"Error event: {event.content}")
    
    async def _handle_complete_event(self, event: StreamEvent):
        """Handle complete events."""
        logger.info(f"Complete event for session: {event.stream_id}")
    
    async def _handle_heartbeat_event(self, event: StreamEvent):
        """Handle heartbeat events."""
        logger.debug(f"Heartbeat for session: {event.stream_id}")
    
    def _ensure_model(self):
        """Ensure the streaming system is initialized."""
        return True
    
    def load_model(self):
        """Load and return the model/pipeline."""
        # Streaming agent doesn't need a model, return True for initialization
        return True
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate streaming response based on prompt."""
        operation = kwargs.get('operation', 'stream')
        
        if operation == 'create_session':
            user_id = kwargs.get('user_id')
            context = kwargs.get('context')
            session_id = self.create_session(user_id, context)
            
            return {
                'operation': 'create_session',
                'session_id': session_id,
                'success': True
            }
        
        elif operation == 'get_status':
            session_id = kwargs.get('session_id')
            if session_id:
                status = self.get_session_status(session_id)
                return {
                    'operation': 'get_status',
                    'session_id': session_id,
                    'status': status,
                    'success': status is not None
                }
            else:
                return {
                    'operation': 'get_status',
                    'stats': self.get_streaming_stats(),
                    'success': True
                }
        
        elif operation == 'close_session':
            session_id = kwargs.get('session_id')
            if session_id:
                success = self.close_session_sync(session_id)
                return {
                    'operation': 'close_session',
                    'session_id': session_id,
                    'success': success
                }
            else:
                return {
                    'operation': 'close_session',
                    'error': 'session_id required',
                    'success': False
                }
        
        else:
            return {
                'operation': 'unknown',
                'error': f'Unknown operation: {operation}'
            }
    
    def shutdown(self):
        """Shutdown the streaming agent."""
        # Cancel background tasks
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        # Close all active sessions
        for session_id in list(self.active_sessions.keys()):
            asyncio.run(self.close_session(session_id))
        
        logger.info("Streaming agent shutdown complete") 