#!/usr/bin/env python3
"""
Memory Agent for Quark AI Assistant
=======================================

Manages long-term memory operations including storage, retrieval,
consolidation, and memory-based reasoning.

Part of Pillar 17: Long-Term Memory & Knowledge Graphs
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from agents.base import Agent
from memory.long_term_memory import LongTermMemory, MemoryType, MemoryPriority, MemoryItem


class MemoryAgent(Agent):
    """Advanced memory management agent for long-term memory operations."""
    
    def __init__(self, model_name: str = "memory_agent", memory_dir: str = None):
        super().__init__(model_name)
        self.memory_dir = memory_dir or os.path.join(os.path.dirname(__file__), '..', 'memory_db')
        os.makedirs(self.memory_dir, exist_ok=True)
        
        # Initialize long-term memory system
        self.long_term_memory = LongTermMemory(self.memory_dir)
        
        # Memory operation settings
        self.auto_consolidation = True
        self.consolidation_interval = 3600  # 1 hour
        self.last_consolidation = time.time()
        
        # Memory tracking
        self.memory_operations = []
        self.retrieval_stats = {
            'total_queries': 0,
            'successful_retrievals': 0,
            'average_results_per_query': 0.0
        }
    
    def load_model(self):
        """Load memory models and components."""
        try:
            # Initialize long-term memory system
            self.long_term_memory = LongTermMemory(self.memory_dir)
            
            # Perform initial consolidation
            self.long_term_memory.consolidate_memories()
            
            return True
        except Exception as e:
            print(f"Error loading memory models: {e}")
            return False
    
    def generate(self, input_data: str, operation: str = "store_memory", **kwargs) -> Dict[str, Any]:
        """
        Generate memory operations or perform memory management.
        
        Args:
            input_data: Memory content or query
            operation: Memory operation to perform
            **kwargs: Additional parameters
            
        Returns:
            Memory operation result
        """
        try:
            if operation == "store_memory":
                return self._store_memory(input_data, **kwargs)
            elif operation == "retrieve_memories":
                return self._retrieve_memories(input_data, **kwargs)
            elif operation == "consolidate_memories":
                return self._consolidate_memories(**kwargs)
            elif operation == "get_memory_stats":
                return self._get_memory_stats()
            elif operation == "export_memory_data":
                return self._export_memory_data(**kwargs)
            elif operation == "search_memories":
                return self._search_memories(input_data, **kwargs)
            elif operation == "associate_memories":
                return self._associate_memories(input_data, **kwargs)
            elif operation == "forget_memory":
                return self._forget_memory(input_data, **kwargs)
            else:
                return {"error": f"Unknown memory operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Memory operation failed: {str(e)}"}
    
    def _store_memory(self, content: str, memory_type: str = "episodic",
                     priority: str = "medium", context: Dict[str, Any] = None,
                     associations: List[str] = None, confidence: float = 1.0,
                     source: str = "user_input", tags: List[str] = None) -> Dict[str, Any]:
        """Store a new memory item."""
        try:
            # Convert string parameters to enums
            memory_type_enum = MemoryType(memory_type)
            priority_enum = MemoryPriority(priority)
            
            # Store memory
            memory_id = self.long_term_memory.store_memory(
                content=content,
                memory_type=memory_type_enum,
                priority=priority_enum,
                context=context or {},
                associations=associations or [],
                confidence=confidence,
                source=source,
                tags=tags or []
            )
            
            # Track operation
            self.memory_operations.append({
                'operation': 'store_memory',
                'memory_id': memory_id,
                'memory_type': memory_type,
                'priority': priority,
                'timestamp': time.time(),
                'content_length': len(content)
            })
            
            return {
                "status": "success",
                "memory_id": memory_id,
                "memory_type": memory_type,
                "priority": priority,
                "timestamp": datetime.now().isoformat(),
                "message": f"Memory stored successfully with ID: {memory_id}"
            }
            
        except Exception as e:
            return {"error": f"Failed to store memory: {str(e)}"}
    
    def _retrieve_memories(self, query: str, memory_types: List[str] = None,
                          max_results: int = 10, min_confidence: float = 0.0,
                          time_range: Optional[tuple] = None,
                          tags: List[str] = None) -> Dict[str, Any]:
        """Retrieve memories based on query and filters."""
        try:
            # Convert memory types to enums
            memory_type_enums = None
            if memory_types:
                memory_type_enums = [MemoryType(mt) for mt in memory_types]
            
            # Retrieve memories
            memories = self.long_term_memory.retrieve_memories(
                query=query,
                memory_types=memory_type_enums,
                max_results=max_results,
                min_confidence=min_confidence,
                time_range=time_range,
                tags=tags
            )
            
            # Convert memories to serializable format
            memory_results = []
            for memory in memories:
                memory_dict = asdict(memory)
                memory_dict['memory_type'] = memory.memory_type.value
                memory_dict['priority'] = memory.priority.value
                memory_results.append(memory_dict)
            
            # Update retrieval stats
            self.retrieval_stats['total_queries'] += 1
            self.retrieval_stats['successful_retrievals'] += 1
            self.retrieval_stats['average_results_per_query'] = (
                (self.retrieval_stats['average_results_per_query'] * 
                 (self.retrieval_stats['total_queries'] - 1) + len(memories)) /
                self.retrieval_stats['total_queries']
            )
            
            return {
                "status": "success",
                "query": query,
                "results_count": len(memories),
                "memories": memory_results,
                "retrieval_stats": self.retrieval_stats
            }
            
        except Exception as e:
            return {"error": f"Failed to retrieve memories: {str(e)}"}
    
    def _consolidate_memories(self, force: bool = False) -> Dict[str, Any]:
        """Consolidate and optimize memory storage."""
        try:
            current_time = time.time()
            
            # Check if consolidation is needed
            if not force and current_time - self.last_consolidation < self.consolidation_interval:
                return {
                    "status": "skipped",
                    "message": "Consolidation not needed yet",
                    "next_consolidation": self.last_consolidation + self.consolidation_interval
                }
            
            # Perform consolidation
            self.long_term_memory.consolidate_memories()
            self.last_consolidation = current_time
            
            # Get updated stats
            stats = self.long_term_memory.get_memory_stats()
            
            return {
                "status": "success",
                "consolidation_timestamp": datetime.now().isoformat(),
                "memory_stats": stats,
                "message": "Memory consolidation completed successfully"
            }
            
        except Exception as e:
            return {"error": f"Failed to consolidate memories: {str(e)}"}
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        try:
            # Get basic stats
            basic_stats = self.long_term_memory.get_memory_stats()
            
            # Get operation stats
            operation_stats = {
                'total_operations': len(self.memory_operations),
                'operations_by_type': {},
                'recent_operations': []
            }
            
            # Count operations by type
            for op in self.memory_operations:
                op_type = op['operation']
                operation_stats['operations_by_type'][op_type] = operation_stats['operations_by_type'].get(op_type, 0) + 1
            
            # Get recent operations (last 10)
            recent_ops = sorted(self.memory_operations, key=lambda x: x['timestamp'], reverse=True)[:10]
            operation_stats['recent_operations'] = recent_ops
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "basic_stats": basic_stats,
                "operation_stats": operation_stats,
                "retrieval_stats": self.retrieval_stats,
                "consolidation_settings": {
                    "auto_consolidation": self.auto_consolidation,
                    "consolidation_interval": self.consolidation_interval,
                    "last_consolidation": self.last_consolidation
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to get memory stats: {str(e)}"}
    
    def _export_memory_data(self, filename: str = None) -> Dict[str, Any]:
        """Export memory data to JSON file."""
        try:
            export_file = self.long_term_memory.export_memory_data(filename)
            
            return {
                "status": "success",
                "export_file": export_file,
                "export_timestamp": datetime.now().isoformat(),
                "message": f"Memory data exported to: {export_file}"
            }
            
        except Exception as e:
            return {"error": f"Failed to export memory data: {str(e)}"}
    
    def _search_memories(self, search_query: str, search_type: str = "content",
                        memory_types: List[str] = None, max_results: int = 20) -> Dict[str, Any]:
        """Advanced memory search with different search types."""
        try:
            # Convert memory types
            memory_type_enums = None
            if memory_types:
                memory_type_enums = [MemoryType(mt) for mt in memory_types]
            
            # Perform search based on type
            if search_type == "content":
                # Standard content-based search
                memories = self.long_term_memory.retrieve_memories(
                    query=search_query,
                    memory_types=memory_type_enums,
                    max_results=max_results
                )
            elif search_type == "tags":
                # Tag-based search
                memories = self.long_term_memory.retrieve_memories(
                    query="",  # Empty query for tag search
                    memory_types=memory_type_enums,
                    max_results=max_results,
                    tags=[search_query]
                )
            elif search_type == "temporal":
                # Time-based search (search for recent memories)
                current_time = time.time()
                time_range = (current_time - 86400, current_time)  # Last 24 hours
                memories = self.long_term_memory.retrieve_memories(
                    query=search_query,
                    memory_types=memory_type_enums,
                    max_results=max_results,
                    time_range=time_range
                )
            else:
                return {"error": f"Unknown search type: {search_type}"}
            
            # Convert to serializable format
            memory_results = []
            for memory in memories:
                memory_dict = asdict(memory)
                memory_dict['memory_type'] = memory.memory_type.value
                memory_dict['priority'] = memory.priority.value
                memory_results.append(memory_dict)
            
            return {
                "status": "success",
                "search_query": search_query,
                "search_type": search_type,
                "results_count": len(memories),
                "memories": memory_results
            }
            
        except Exception as e:
            return {"error": f"Failed to search memories: {str(e)}"}
    
    def _associate_memories(self, memory_id: str, associated_ids: List[str]) -> Dict[str, Any]:
        """Associate memories with each other."""
        try:
            # Find the target memory
            target_memory = None
            for memory_type in MemoryType:
                memories = getattr(self.long_term_memory, f"{memory_type.value}_memories")
                if memory_id in memories:
                    target_memory = memories[memory_id]
                    break
            
            if not target_memory:
                return {"error": f"Memory with ID {memory_id} not found"}
            
            # Add associations
            for associated_id in associated_ids:
                if associated_id not in target_memory.associations:
                    target_memory.associations.append(associated_id)
            
            # Track operation
            self.memory_operations.append({
                'operation': 'associate_memories',
                'memory_id': memory_id,
                'associated_ids': associated_ids,
                'timestamp': time.time()
            })
            
            return {
                "status": "success",
                "memory_id": memory_id,
                "associations_added": len(associated_ids),
                "total_associations": len(target_memory.associations),
                "message": f"Added {len(associated_ids)} associations to memory {memory_id}"
            }
            
        except Exception as e:
            return {"error": f"Failed to associate memories: {str(e)}"}
    
    def _forget_memory(self, memory_id: str, permanent: bool = False) -> Dict[str, Any]:
        """Remove a memory from the system."""
        try:
            # Find and remove memory
            removed = False
            for memory_type in MemoryType:
                memories = getattr(self.long_term_memory, f"{memory_type.value}_memories")
                if memory_id in memories:
                    del memories[memory_id]
                    removed = True
                    break
            
            if not removed:
                return {"error": f"Memory with ID {memory_id} not found"}
            
            # Track operation
            self.memory_operations.append({
                'operation': 'forget_memory',
                'memory_id': memory_id,
                'permanent': permanent,
                'timestamp': time.time()
            })
            
            return {
                "status": "success",
                "memory_id": memory_id,
                "permanent": permanent,
                "message": f"Memory {memory_id} removed successfully"
            }
            
        except Exception as e:
            return {"error": f"Failed to forget memory: {str(e)}"}
    
    def get_memory_recommendations(self, context: str = None) -> Dict[str, Any]:
        """Get memory-related recommendations."""
        try:
            recommendations = []
            
            # Check if consolidation is needed
            current_time = time.time()
            if current_time - self.last_consolidation > self.consolidation_interval:
                recommendations.append({
                    'type': 'consolidation',
                    'priority': 'medium',
                    'message': 'Memory consolidation is recommended',
                    'action': 'consolidate_memories'
                })
            
            # Check memory distribution
            stats = self.long_term_memory.get_memory_stats()
            total_memories = stats['total_memories']
            
            if total_memories > 1000:
                recommendations.append({
                    'type': 'cleanup',
                    'priority': 'high',
                    'message': 'Large number of memories detected, consider cleanup',
                    'action': 'consolidate_memories'
                })
            
            # Check working memory usage
            working_count = stats['working_memories']
            if working_count > 80:
                recommendations.append({
                    'type': 'working_memory',
                    'priority': 'medium',
                    'message': f'Working memory is {working_count}/100 full',
                    'action': 'consolidate_memories'
                })
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "recommendations_count": len(recommendations)
            }
            
        except Exception as e:
            return {"error": f"Failed to get recommendations: {str(e)}"}

