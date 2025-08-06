#!/usr/bin/env python3
"""
Context Window Manager for Meta-Model AI Assistant
Handles short-term context management and conversation history
"""

import time
import json
from typing import List, Dict, Any, Optional
from collections import deque
from datetime import datetime

class ContextWindowManager:
    def __init__(self, max_tokens: int = 4000, max_messages: int = 20):
        self.max_tokens = max_tokens
        self.max_messages = max_messages
        self.context_window = deque(maxlen=max_messages)
        self.current_tokens = 0
        self.session_id = None
        self.session_start = None
        
    def start_session(self, session_id: str = None):
        """Start a new conversation session."""
        if session_id is None:
            session_id = f"session_{int(time.time())}"
        
        self.session_id = session_id
        self.session_start = datetime.now()
        self.context_window.clear()
        self.current_tokens = 0
        
        # Add session start marker
        self.add_message("system", f"Session started: {session_id}")
        
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the context window."""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'tokens': self._estimate_tokens(content)
        }
        
        if metadata:
            message['metadata'] = metadata
            
        # Add to context window
        self.context_window.append(message)
        
        # Update token count
        self.current_tokens += message['tokens']
        
        # Trim if necessary
        self._trim_context()
        
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4 + 1
        
    def _trim_context(self):
        """Trim context window if it exceeds limits."""
        while (self.current_tokens > self.max_tokens or 
               len(self.context_window) > self.max_messages):
            
            if len(self.context_window) == 0:
                break
                
            # Remove oldest message
            removed_message = self.context_window.popleft()
            self.current_tokens -= removed_message['tokens']
            
    def get_context(self, include_system: bool = True) -> List[Dict[str, Any]]:
        """Get current context window."""
        context = []
        
        for message in self.context_window:
            # Skip system messages if not requested
            if not include_system and message['role'] == 'system':
                continue
            context.append(message)
            
        return context
        
    def get_context_string(self, include_system: bool = True) -> str:
        """Get context as a formatted string."""
        context = self.get_context(include_system)
        
        if not context:
            return ""
            
        formatted_context = []
        for message in context:
            role = message['role'].upper()
            content = message['content']
            formatted_context.append(f"{role}: {content}")
            
        return "\n".join(formatted_context)
        
    def get_recent_context(self, n_messages: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent n messages from context."""
        return list(self.context_window)[-n_messages:]
        
    def clear_context(self):
        """Clear the context window."""
        self.context_window.clear()
        self.current_tokens = 0
        
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about the current context."""
        return {
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat() if self.session_start else None,
            'message_count': len(self.context_window),
            'current_tokens': self.current_tokens,
            'max_tokens': self.max_tokens,
            'max_messages': self.max_messages,
            'token_usage_percent': (self.current_tokens / self.max_tokens) * 100 if self.max_tokens > 0 else 0
        }
        
    def is_context_full(self) -> bool:
        """Check if context window is at capacity."""
        return (self.current_tokens >= self.max_tokens or 
                len(self.context_window) >= self.max_messages)
                
    def get_context_summary(self) -> str:
        """Get a summary of the current context."""
        if not self.context_window:
            return "No context available"
            
        stats = self.get_context_stats()
        recent_messages = self.get_recent_context(3)
        
        summary = f"Session: {stats['session_id']}\n"
        summary += f"Messages: {stats['message_count']}\n"
        summary += f"Tokens: {stats['current_tokens']}/{stats['max_tokens']} ({stats['token_usage_percent']:.1f}%)\n"
        summary += f"Recent messages:\n"
        
        for message in recent_messages:
            role = message['role'].upper()
            content = message['content'][:100] + "..." if len(message['content']) > 100 else message['content']
            summary += f"  {role}: {content}\n"
            
        return summary

