#!/usr/bin/env python3
"""
Memory Eviction Manager for Quark AI Assistant
Handles long-term memory management and cleanup policies
"""

import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from agents.memory_agent import MemoryAgent

class MemoryEvictionManager:
    def __init__(self, memory_agent: MemoryAgent):
        self.memory_agent = memory_agent
        self.eviction_policies = {
            'time_based': {
                'enabled': True,
                'ttl_days': 7,
                'check_interval_hours': 24
            },
            'size_based': {
                'enabled': True,
                'max_memories': 1000,
                'eviction_threshold': 0.9  # Evict when 90% full
            },
            'relevance_based': {
                'enabled': True,
                'min_relevance_score': 0.3,
                'max_age_days': 30
            }
        }
        self.last_cleanup = time.time()
        
    def should_run_cleanup(self) -> bool:
        """Check if cleanup should be run based on time interval."""
        hours_since_last = (time.time() - self.last_cleanup) / 3600
        return hours_since_last >= self.eviction_policies['time_based']['check_interval_hours']
        
    def run_cleanup(self) -> Dict[str, Any]:
        """Run memory cleanup based on configured policies."""
        if not self.should_run_cleanup():
            return {'status': 'skipped', 'reason': 'Too soon since last cleanup'}
            
        cleanup_results = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'policies_run': [],
            'memories_removed': 0,
            'errors': []
        }
        
        try:
            # Time-based cleanup
            if self.eviction_policies['time_based']['enabled']:
                removed = self._time_based_cleanup()
                cleanup_results['policies_run'].append('time_based')
                cleanup_results['memories_removed'] += removed
                
            # Size-based cleanup
            if self.eviction_policies['size_based']['enabled']:
                removed = self._size_based_cleanup()
                cleanup_results['policies_run'].append('size_based')
                cleanup_results['memories_removed'] += removed
                
            # Relevance-based cleanup
            if self.eviction_policies['relevance_based']['enabled']:
                removed = self._relevance_based_cleanup()
                cleanup_results['policies_run'].append('relevance_based')
                cleanup_results['memories_removed'] += removed
                
            self.last_cleanup = time.time()
            
        except Exception as e:
            cleanup_results['status'] = 'error'
            cleanup_results['errors'].append(str(e))
            
        return cleanup_results
        
    def _time_based_cleanup(self) -> int:
        """Remove memories older than TTL."""
        ttl_days = self.eviction_policies['time_based']['ttl_days']
        cutoff_time = time.time() - (ttl_days * 24 * 60 * 60)
        
        # Get all memories
        stats = self.memory_agent.get_memory_stats()
        if stats['total_memories'] == 0:
            return 0
            
        # Get recent memories to check timestamps
        recent_memories = self.memory_agent.get_recent_memories(1000)
        
        old_memory_ids = []
        for memory in recent_memories:
            created_at = memory['metadata'].get('created_at', 0)
            if created_at < cutoff_time:
                old_memory_ids.append(memory['id'])
                
        # Delete old memories
        removed_count = 0
        for memory_id in old_memory_ids:
            if self.memory_agent.delete_memory(memory_id):
                removed_count += 1
                
        return removed_count
        
    def _size_based_cleanup(self) -> int:
        """Remove memories when storage is too full."""
        stats = self.memory_agent.get_memory_stats()
        max_memories = self.eviction_policies['size_based']['max_memories']
        threshold = self.eviction_policies['size_based']['eviction_threshold']
        
        if stats['total_memories'] < max_memories * threshold:
            return 0
            
        # Get all memories sorted by age (oldest first)
        all_memories = self.memory_agent.get_recent_memories(10000)
        all_memories.sort(key=lambda x: x['metadata'].get('created_at', 0))
        
        # Calculate how many to remove
        memories_to_remove = int(stats['total_memories'] * 0.2)  # Remove 20%
        
        removed_count = 0
        for memory in all_memories[:memories_to_remove]:
            if self.memory_agent.delete_memory(memory['id']):
                removed_count += 1
                
        return removed_count
        
    def _relevance_based_cleanup(self) -> int:
        """Remove low-relevance memories."""
        min_score = self.eviction_policies['relevance_based']['min_relevance_score']
        max_age_days = self.eviction_policies['relevance_based']['max_age_days']
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        # Get recent memories
        recent_memories = self.memory_agent.get_recent_memories(1000)
        
        low_relevance_ids = []
        for memory in recent_memories:
            created_at = memory['metadata'].get('created_at', 0)
            
            # Check if memory is old enough to consider for relevance-based cleanup
            if created_at < cutoff_time:
                # For now, we'll use a simple heuristic based on content length
                # In a real implementation, you'd use actual relevance scores
                content_length = memory['metadata'].get('content_length', 0)
                if content_length < 50:  # Very short memories might be less relevant
                    low_relevance_ids.append(memory['id'])
                    
        # Delete low-relevance memories
        removed_count = 0
        for memory_id in low_relevance_ids:
            if self.memory_agent.delete_memory(memory_id):
                removed_count += 1
                
        return removed_count
        
    def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get statistics about cleanup operations."""
        stats = self.memory_agent.get_memory_stats()
        
        return {
            'last_cleanup': datetime.fromtimestamp(self.last_cleanup).isoformat(),
            'hours_since_last_cleanup': (time.time() - self.last_cleanup) / 3600,
            'should_run_cleanup': self.should_run_cleanup(),
            'eviction_policies': self.eviction_policies,
            'memory_stats': stats
        }
        
    def update_eviction_policy(self, policy_name: str, settings: Dict[str, Any]):
        """Update eviction policy settings."""
        if policy_name in self.eviction_policies:
            self.eviction_policies[policy_name].update(settings)
            
    def get_memory_health_report(self) -> Dict[str, Any]:
        """Generate a comprehensive memory health report."""
        stats = self.memory_agent.get_memory_stats()
        cleanup_stats = self.get_cleanup_stats()
        
        # Calculate health metrics
        total_memories = stats.get('total_memories', 0)
        max_memories = self.eviction_policies['size_based']['max_memories']
        memory_usage_percent = (total_memories / max_memories) * 100 if max_memories > 0 else 0
        
        # Determine health status
        if memory_usage_percent > 90:
            health_status = 'critical'
        elif memory_usage_percent > 70:
            health_status = 'warning'
        else:
            health_status = 'healthy'
            
        return {
            'health_status': health_status,
            'memory_usage_percent': memory_usage_percent,
            'total_memories': total_memories,
            'max_memories': max_memories,
            'memory_types': stats.get('memory_types', {}),
            'cleanup_stats': cleanup_stats,
            'recommendations': self._generate_recommendations(stats, cleanup_stats)
        }
        
    def _generate_recommendations(self, stats: Dict[str, Any], cleanup_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on memory health."""
        recommendations = []
        
        total_memories = stats.get('total_memories', 0)
        max_memories = self.eviction_policies['size_based']['max_memories']
        memory_usage_percent = (total_memories / max_memories) * 100 if max_memories > 0 else 0
        
        if memory_usage_percent > 90:
            recommendations.append("Memory usage is critical. Consider running immediate cleanup.")
        elif memory_usage_percent > 70:
            recommendations.append("Memory usage is high. Consider running cleanup soon.")
            
        if cleanup_stats['hours_since_last_cleanup'] > 48:
            recommendations.append("Cleanup hasn't run recently. Consider running cleanup.")
            
        if total_memories == 0:
            recommendations.append("No memories stored. Consider storing some test memories.")
            
        return recommendations

