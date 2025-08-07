#!/usr/bin/env python3
"""
Long-Term Memory System for Quark AI Assistant
==================================================

Implements persistent memory systems with episodic and semantic memory,
memory consolidation, and intelligent retrieval mechanisms.

Part of Pillar 17: Long-Term Memory & Knowledge Graphs
"""

import os
import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict

from agents.base import Agent


class MemoryType(Enum):
    """Types of memory in the system."""
    EPISODIC = "episodic"      # Event-based memories with temporal context
    SEMANTIC = "semantic"       # Factual knowledge and concepts
    PROCEDURAL = "procedural"   # Skills and procedures
    WORKING = "working"         # Short-term working memory


class MemoryPriority(Enum):
    """Priority levels for memory storage."""
    CRITICAL = "critical"       # Essential information
    HIGH = "high"              # Important information
    MEDIUM = "medium"          # Standard information
    LOW = "low"                # Less important information


@dataclass
class MemoryItem:
    """Individual memory item with metadata."""
    id: str
    content: str
    memory_type: MemoryType
    priority: MemoryPriority
    timestamp: float
    context: Dict[str, Any]
    associations: List[str]  # IDs of related memories
    access_count: int
    last_accessed: float
    confidence: float  # Confidence in the memory's accuracy
    source: str  # Source of the memory
    tags: List[str]  # Categorization tags


@dataclass
class MemoryQuery:
    """Query for memory retrieval."""
    query: str
    memory_types: List[MemoryType]
    max_results: int
    min_confidence: float
    time_range: Optional[Tuple[float, float]]
    tags: List[str]


class LongTermMemory:
    """Advanced long-term memory system with multiple memory types."""
    
    def __init__(self, memory_dir: str = None):
        self.memory_dir = memory_dir or os.path.join(os.path.dirname(__file__), '..', 'memory_db')
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Memory storage
        self.episodic_memories = {}  # Event-based memories
        self.semantic_memories = {}  # Factual knowledge
        self.procedural_memories = {}  # Skills and procedures
        self.working_memories = {}  # Short-term working memory
        
        # Memory indexing
        self.content_index = defaultdict(list)  # Content-based indexing
        self.tag_index = defaultdict(list)  # Tag-based indexing
        self.temporal_index = defaultdict(list)  # Time-based indexing
        
        # Memory consolidation settings
        self.consolidation_threshold = 0.7
        self.retention_threshold = 0.5
        self.max_working_memories = 100
        
        # Load existing memories
        self._load_memories()
    
    def initialize(self):
        """Initialize the long-term memory system"""
        # Load existing memories
        self._load_memories()
    
    def _load_memories(self):
        """Load existing memories from storage."""
        try:
            for memory_type in MemoryType:
                filename = f"{memory_type.value}_memories.json"
                filepath = os.path.join(self.memory_dir, filename)
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        memories_data = json.load(f)
                        
                    for memory_id, memory_data in memories_data.items():
                        memory_item = MemoryItem(
                            id=memory_id,
                            content=memory_data['content'],
                            memory_type=MemoryType(memory_data['memory_type']),
                            priority=MemoryPriority(memory_data['priority']),
                            timestamp=memory_data['timestamp'],
                            context=memory_data['context'],
                            associations=memory_data['associations'],
                            access_count=memory_data['access_count'],
                            last_accessed=memory_data['last_accessed'],
                            confidence=memory_data['confidence'],
                            source=memory_data['source'],
                            tags=memory_data['tags']
                        )
                        
                        self._store_memory(memory_item)
                        
        except Exception as e:
            print(f"Error loading memories: {e}")
    
    def _store_memory(self, memory_item: MemoryItem):
        """Store memory item in appropriate collection and indexes."""
        # Store in appropriate collection
        if memory_item.memory_type == MemoryType.EPISODIC:
            self.episodic_memories[memory_item.id] = memory_item
        elif memory_item.memory_type == MemoryType.SEMANTIC:
            self.semantic_memories[memory_item.id] = memory_item
        elif memory_item.memory_type == MemoryType.PROCEDURAL:
            self.procedural_memories[memory_item.id] = memory_item
        elif memory_item.memory_type == MemoryType.WORKING:
            self.working_memories[memory_item.id] = memory_item
            
            # Limit working memory size
            if len(self.working_memories) > self.max_working_memories:
                self._evict_oldest_working_memory()
        
        # Update indexes
        self._update_indexes(memory_item)
    
    def _update_indexes(self, memory_item: MemoryItem):
        """Update memory indexes for efficient retrieval."""
        # Content index
        words = memory_item.content.lower().split()
        for word in words:
            if len(word) > 2:  # Only index words longer than 2 characters
                self.content_index[word].append(memory_item.id)
        
        # Tag index
        for tag in memory_item.tags:
            self.tag_index[tag].append(memory_item.id)
        
        # Temporal index
        date_key = datetime.fromtimestamp(memory_item.timestamp).strftime("%Y-%m-%d")
        self.temporal_index[date_key].append(memory_item.id)
    
    def _evict_oldest_working_memory(self):
        """Evict the oldest working memory when limit is reached."""
        if not self.working_memories:
            return
        
        oldest_id = min(self.working_memories.keys(), 
                       key=lambda x: self.working_memories[x].last_accessed)
        oldest_memory = self.working_memories[oldest_id]
        
        # Move to episodic memory if it's important enough
        if oldest_memory.access_count > 2 or oldest_memory.priority in [MemoryPriority.HIGH, MemoryPriority.CRITICAL]:
            oldest_memory.memory_type = MemoryType.EPISODIC
            self.episodic_memories[oldest_id] = oldest_memory
        
        # Remove from working memory
        del self.working_memories[oldest_id]
    
    def store_memory(self, content: str, memory_type: MemoryType, 
                    priority: MemoryPriority = MemoryPriority.MEDIUM,
                    context: Dict[str, Any] = None, 
                    associations: List[str] = None,
                    confidence: float = 1.0,
                    source: str = "user_input",
                    tags: List[str] = None) -> str:
        """Store a new memory item."""
        memory_id = self._generate_memory_id(content, memory_type)
        
        memory_item = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            priority=priority,
            timestamp=time.time(),
            context=context or {},
            associations=associations or [],
            access_count=0,
            last_accessed=time.time(),
            confidence=confidence,
            source=source,
            tags=tags or []
        )
        
        self._store_memory(memory_item)
        return memory_id
    
    def _generate_memory_id(self, content: str, memory_type: MemoryType) -> str:
        """Generate unique memory ID."""
        timestamp = str(time.time())
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{memory_type.value}_{content_hash}_{timestamp}"
    
    def retrieve_memories(self, query: str, memory_types: List[MemoryType] = None,
                         max_results: int = 10, min_confidence: float = 0.0,
                         time_range: Optional[Tuple[float, float]] = None,
                         tags: List[str] = None) -> List[MemoryItem]:
        """Retrieve memories based on query and filters."""
        if memory_types is None:
            memory_types = list(MemoryType)
        
        # Get all memories of specified types
        all_memories = []
        for memory_type in memory_types:
            if memory_type == MemoryType.EPISODIC:
                all_memories.extend(self.episodic_memories.values())
            elif memory_type == MemoryType.SEMANTIC:
                all_memories.extend(self.semantic_memories.values())
            elif memory_type == MemoryType.PROCEDURAL:
                all_memories.extend(self.procedural_memories.values())
            elif memory_type == MemoryType.WORKING:
                all_memories.extend(self.working_memories.values())
        
        # Filter by confidence and time range
        filtered_memories = []
        for memory in all_memories:
            if memory.confidence < min_confidence:
                continue
            
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= memory.timestamp <= end_time):
                    continue
            
            if tags:
                if not any(tag in memory.tags for tag in tags):
                    continue
            
            filtered_memories.append(memory)
        
        # Score memories based on relevance
        scored_memories = []
        query_words = query.lower().split()
        
        for memory in filtered_memories:
            score = self._calculate_relevance_score(memory, query_words)
            scored_memories.append((memory, score))
        
        # Sort by relevance score and return top results
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Update access statistics
        for memory, _ in scored_memories[:max_results]:
            memory.access_count += 1
            memory.last_accessed = time.time()
        
        return [memory for memory, _ in scored_memories[:max_results]]
    
    def get_recent_memories(self, limit: int = 10) -> List[MemoryItem]:
        """Get the most recent memories across all types."""
        all_memories = []
        
        # Collect all memories
        all_memories.extend(self.episodic_memories.values())
        all_memories.extend(self.semantic_memories.values())
        all_memories.extend(self.procedural_memories.values())
        all_memories.extend(self.working_memories.values())
        
        # Sort by timestamp (most recent first)
        all_memories.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Return the most recent ones
        return all_memories[:limit]
    
    def _calculate_relevance_score(self, memory: MemoryItem, query_words: List[str]) -> float:
        """Calculate relevance score for memory retrieval."""
        score = 0.0
        
        # Content relevance
        memory_words = memory.content.lower().split()
        word_matches = sum(1 for word in query_words if word in memory_words)
        content_score = word_matches / len(query_words) if query_words else 0
        
        # Recency score (more recent memories get higher scores)
        time_diff = time.time() - memory.timestamp
        recency_score = 1.0 / (1.0 + time_diff / 86400)  # Decay over days
        
        # Access frequency score
        frequency_score = min(memory.access_count / 10.0, 1.0)
        
        # Priority score
        priority_scores = {
            MemoryPriority.CRITICAL: 1.0,
            MemoryPriority.HIGH: 0.8,
            MemoryPriority.MEDIUM: 0.6,
            MemoryPriority.LOW: 0.4
        }
        priority_score = priority_scores.get(memory.priority, 0.5)
        
        # Combined score
        score = (content_score * 0.4 + 
                recency_score * 0.2 + 
                frequency_score * 0.2 + 
                priority_score * 0.2)
        
        return score
    
    def consolidate_memories(self):
        """Consolidate and optimize memory storage."""
        # Consolidate similar memories
        self._consolidate_similar_memories()
        
        # Move important working memories to episodic
        self._promote_important_working_memories()
        
        # Clean up old, rarely accessed memories
        self._cleanup_old_memories()
        
        # Save memories to disk
        self._save_memories()
    
    def _consolidate_similar_memories(self):
        """Consolidate memories with similar content."""
        # Group memories by type for consolidation
        for memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC]:
            memories = (self.episodic_memories if memory_type == MemoryType.EPISODIC 
                      else self.semantic_memories)
            
            memory_list = list(memories.values())
            
            for i, memory1 in enumerate(memory_list):
                for j, memory2 in enumerate(memory_list[i+1:], i+1):
                    similarity = self._calculate_memory_similarity(memory1, memory2)
                    
                    if similarity > self.consolidation_threshold:
                        # Consolidate memories
                        consolidated = self._consolidate_memory_pair(memory1, memory2)
                        
                        # Remove original memories
                        del memories[memory1.id]
                        del memories[memory2.id]
                        
                        # Add consolidated memory
                        memories[consolidated.id] = consolidated
    
    def _calculate_memory_similarity(self, memory1: MemoryItem, memory2: MemoryItem) -> float:
        """Calculate similarity between two memories."""
        # Simple word overlap similarity
        words1 = set(memory1.content.lower().split())
        words2 = set(memory2.content.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _consolidate_memory_pair(self, memory1: MemoryItem, memory2: MemoryItem) -> MemoryItem:
        """Consolidate two similar memories into one."""
        # Combine content
        combined_content = f"{memory1.content} {memory2.content}"
        
        # Use higher priority and confidence
        priority = memory1.priority if memory1.priority.value > memory2.priority.value else memory2.priority
        confidence = max(memory1.confidence, memory2.confidence)
        
        # Combine tags
        combined_tags = list(set(memory1.tags + memory2.tags))
        
        # Use earlier timestamp
        timestamp = min(memory1.timestamp, memory2.timestamp)
        
        return MemoryItem(
            id=self._generate_memory_id(combined_content, memory1.memory_type),
            content=combined_content,
            memory_type=memory1.memory_type,
            priority=priority,
            timestamp=timestamp,
            context={**memory1.context, **memory2.context},
            associations=list(set(memory1.associations + memory2.associations)),
            access_count=memory1.access_count + memory2.access_count,
            last_accessed=max(memory1.last_accessed, memory2.last_accessed),
            confidence=confidence,
            source=f"consolidated_{memory1.source}_{memory2.source}",
            tags=combined_tags
        )
    
    def _promote_important_working_memories(self):
        """Promote important working memories to episodic memory."""
        to_promote = []
        
        for memory in self.working_memories.values():
            if (memory.access_count > 3 or 
                memory.priority in [MemoryPriority.HIGH, MemoryPriority.CRITICAL] or
                time.time() - memory.timestamp > 3600):  # Older than 1 hour
                to_promote.append(memory)
        
        for memory in to_promote:
            memory.memory_type = MemoryType.EPISODIC
            self.episodic_memories[memory.id] = memory
            del self.working_memories[memory.id]
    
    def _cleanup_old_memories(self):
        """Remove old, rarely accessed memories."""
        current_time = time.time()
        
        for memory_type in [MemoryType.EPISODIC, MemoryType.SEMANTIC]:
            memories = (self.episodic_memories if memory_type == MemoryType.EPISODIC 
                      else self.semantic_memories)
            
            to_remove = []
            
            for memory in memories.values():
                # Remove if old, low priority, and rarely accessed
                age_days = (current_time - memory.timestamp) / 86400
                
                if (age_days > 30 and  # Older than 30 days
                    memory.priority == MemoryPriority.LOW and
                    memory.access_count < 2):
                    to_remove.append(memory.id)
            
            for memory_id in to_remove:
                del memories[memory_id]
    
    def _save_memories(self):
        """Save memories to disk."""
        try:
            for memory_type in MemoryType:
                memories = {}
                
                if memory_type == MemoryType.EPISODIC:
                    memories = self.episodic_memories
                elif memory_type == MemoryType.SEMANTIC:
                    memories = self.semantic_memories
                elif memory_type == MemoryType.PROCEDURAL:
                    memories = self.procedural_memories
                elif memory_type == MemoryType.WORKING:
                    memories = self.working_memories
                
                filename = f"{memory_type.value}_memories.json"
                filepath = os.path.join(self.memory_dir, filename)
                
                # Convert to serializable format
                memories_data = {}
                for memory_id, memory in memories.items():
                    memories_data[memory_id] = asdict(memory)
                    memories_data[memory_id]['memory_type'] = memory.memory_type.value
                    memories_data[memory_id]['priority'] = memory.priority.value
                
                with open(filepath, 'w') as f:
                    json.dump(memories_data, f, indent=2)
                    
        except Exception as e:
            print(f"Error saving memories: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'episodic_memories': len(self.episodic_memories),
            'semantic_memories': len(self.semantic_memories),
            'procedural_memories': len(self.procedural_memories),
            'working_memories': len(self.working_memories),
            'total_memories': (len(self.episodic_memories) + 
                             len(self.semantic_memories) + 
                             len(self.procedural_memories) + 
                             len(self.working_memories)),
            'content_index_size': len(self.content_index),
            'tag_index_size': len(self.tag_index),
            'temporal_index_size': len(self.temporal_index)
        }
    
    def export_memory_data(self, filename: str = None) -> str:
        """Export memory data to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"memory_export_{timestamp}.json"
        
        filepath = os.path.join(self.memory_dir, filename)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'memory_stats': self.get_memory_stats(),
            'episodic_memories': {k: asdict(v) for k, v in self.episodic_memories.items()},
            'semantic_memories': {k: asdict(v) for k, v in self.semantic_memories.items()},
            'procedural_memories': {k: asdict(v) for k, v in self.procedural_memories.items()},
            'working_memories': {k: asdict(v) for k, v in self.working_memories.items()}
        }
        
        # Convert enum values to strings
        for memory_type in ['episodic_memories', 'semantic_memories', 'procedural_memories', 'working_memories']:
            for memory_id, memory_data in export_data[memory_type].items():
                memory_data['memory_type'] = memory_data['memory_type'].value
                memory_data['priority'] = memory_data['priority'].value
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath 