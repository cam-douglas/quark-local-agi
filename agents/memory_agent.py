#!/usr/bin/env python3
# File: meta_model/agents/memory_agent.py

import os
import json
import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from .base import Agent

class MemoryAgent(Agent):
    def __init__(self, db_path: str = None):
        # Initialize without model_name since this is a memory agent
        self.model_name = None
        self.model = None
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), '..', 'memory_db')
        self.client = None
        self.collection = None
        self.embedding_function = None
        self.context_window_size = 10  # Number of recent memories to keep in context
        self.max_memories = 1000  # Maximum number of memories to store
        self.memory_ttl = 7 * 24 * 60 * 60  # 7 days in seconds
        
    def load_model(self):
        """Initialize ChromaDB and embedding function."""
        try:
            # Create memory directory if it doesn't exist
            os.makedirs(self.db_path, exist_ok=True)
            
            # Initialize ChromaDB with new client
            self.client = Client()
            
            # Initialize embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
            
            # Create or get collection
            self.collection = self.client.get_or_create_collection(
                name="meta_model_memory",
                embedding_function=self.embedding_function,
                metadata={"description": "Meta-Model AI Assistant Memory Store"}
            )
            
            return True
        except Exception as e:
            print(f"Error initializing memory agent: {e}")
            return False

    def _ensure_model(self):
        """Ensure the memory system is initialized."""
        if self.collection is None:
            self.load_model()
        return self.collection is not None

    def store_memory(self, content: str, memory_type: str = "conversation", 
                    metadata: Dict[str, Any] = None) -> str:
        """Store a new memory."""
        if not self._ensure_model():
            return None
            
        try:
            # Generate unique ID
            memory_id = str(uuid.uuid4())
            
            # Prepare metadata
            memory_metadata = {
                "type": memory_type,
                "timestamp": datetime.now().isoformat(),
                "created_at": time.time(),
                "content_length": len(content)
            }
            
            # Add custom metadata
            if metadata:
                memory_metadata.update(metadata)
            
            # Store in ChromaDB
            self.collection.add(
                documents=[content],
                metadatas=[memory_metadata],
                ids=[memory_id]
            )
            
            # Cleanup old memories if needed
            self._cleanup_old_memories()
            
            return memory_id
        except Exception as e:
            print(f"Error storing memory: {e}")
            return None

    def retrieve_memories(self, query: str, n_results: int = 5, 
                         memory_type: str = None) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query."""
        if not self._ensure_model():
            return []
            
        try:
            # Build where clause for memory type filter
            where_clause = {}
            if memory_type:
                where_clause["type"] = memory_type
            
            # Query ChromaDB
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Format results
            memories = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    memory = {
                        'id': results['ids'][0][i],
                        'content': doc,
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    }
                    memories.append(memory)
            
            return memories
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []

    def get_recent_memories(self, n_results: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent memories."""
        if not self._ensure_model():
            return []
            
        try:
            # Get all memories and sort by timestamp
            results = self.collection.get()
            
            if not results['documents']:
                return []
            
            # Create list of memories with timestamps
            memories = []
            for i, doc in enumerate(results['documents']):
                memory = {
                    'id': results['ids'][i],
                    'content': doc,
                    'metadata': results['metadatas'][i],
                    'timestamp': results['metadatas'][i].get('timestamp', '')
                }
                memories.append(memory)
            
            # Sort by timestamp (newest first)
            memories.sort(key=lambda x: x['timestamp'], reverse=True)
            
            return memories[:n_results]
        except Exception as e:
            print(f"Error getting recent memories: {e}")
            return []

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        if not self._ensure_model():
            return False
            
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False

    def clear_memories(self, memory_type: str = None) -> bool:
        """Clear all memories or memories of a specific type."""
        if not self._ensure_model():
            return False
            
        try:
            if memory_type:
                # Delete memories of specific type
                self.collection.delete(where={"type": memory_type})
            else:
                # Delete all memories
                self.collection.delete()
            return True
        except Exception as e:
            print(f"Error clearing memories: {e}")
            return False

    def _cleanup_old_memories(self):
        """Remove old memories based on TTL."""
        try:
            current_time = time.time()
            cutoff_time = current_time - self.memory_ttl
            
            # Get all memories
            results = self.collection.get()
            
            if not results['documents']:
                return
            
            # Find old memories
            old_memory_ids = []
            for i, metadata in enumerate(results['metadatas']):
                created_at = metadata.get('created_at', 0)
                if created_at < cutoff_time:
                    old_memory_ids.append(results['ids'][i])
            
            # Delete old memories
            if old_memory_ids:
                self.collection.delete(ids=old_memory_ids)
                print(f"Cleaned up {len(old_memory_ids)} old memories")
        except Exception as e:
            print(f"Error during memory cleanup: {e}")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories."""
        if not self._ensure_model():
            return {}
            
        try:
            results = self.collection.get()
            
            if not results['documents']:
                return {
                    'total_memories': 0,
                    'memory_types': {},
                    'oldest_memory': None,
                    'newest_memory': None
                }
            
            # Count by type
            memory_types = {}
            timestamps = []
            
            for metadata in results['metadatas']:
                memory_type = metadata.get('type', 'unknown')
                memory_types[memory_type] = memory_types.get(memory_type, 0) + 1
                
                timestamp = metadata.get('timestamp', '')
                if timestamp:
                    timestamps.append(timestamp)
            
            return {
                'total_memories': len(results['documents']),
                'memory_types': memory_types,
                'oldest_memory': min(timestamps) if timestamps else None,
                'newest_memory': max(timestamps) if timestamps else None
            }
        except Exception as e:
            print(f"Error getting memory stats: {e}")
            return {}

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Main memory agent interface - store and retrieve memories."""
        if not self._ensure_model():
            return {"error": "Memory system not initialized"}
        
        try:
            # Determine operation based on prompt or kwargs
            operation = kwargs.get('operation', 'retrieve')
            
            if operation == 'store':
                # Store new memory
                content = kwargs.get('content', prompt)
                memory_type = kwargs.get('memory_type', 'conversation')
                metadata = kwargs.get('metadata', {})
                
                memory_id = self.store_memory(content, memory_type, metadata)
                
                return {
                    'operation': 'store',
                    'memory_id': memory_id,
                    'success': memory_id is not None
                }
            
            elif operation == 'retrieve':
                # Retrieve relevant memories
                n_results = kwargs.get('n_results', 5)
                memory_type = kwargs.get('memory_type')
                
                memories = self.retrieve_memories(prompt, n_results, memory_type)
                
                return {
                    'operation': 'retrieve',
                    'memories': memories,
                    'count': len(memories)
                }
            
            elif operation == 'recent':
                # Get recent memories
                n_results = kwargs.get('n_results', 10)
                memories = self.get_recent_memories(n_results)
                
                return {
                    'operation': 'recent',
                    'memories': memories,
                    'count': len(memories)
                }
            
            elif operation == 'stats':
                # Get memory statistics
                stats = self.get_memory_stats()
                
                return {
                    'operation': 'stats',
                    'stats': stats
                }
            
            else:
                # Default: retrieve relevant memories
                memories = self.retrieve_memories(prompt, 5)
                
                return {
                    'operation': 'retrieve',
                    'memories': memories,
                    'count': len(memories)
                }
                
        except Exception as e:
            return {
                'error': f"Memory operation failed: {str(e)}",
                'operation': kwargs.get('operation', 'unknown')
            }

