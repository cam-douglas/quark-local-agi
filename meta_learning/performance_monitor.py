#!/usr/bin/env python3
"""
Performance Monitor for Quark AI Assistant
Monitors agent performance and system health

Part of Pillar 16: Meta-Learning & Self-Reflection
"""

import time
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

@dataclass
class PerformanceSnapshot:
    """A snapshot of system performance at a point in time."""
    timestamp: float
    agent_metrics: Dict[str, Dict[str, float]]
    system_health: Dict[str, Any]
    alerts: List[Dict[str, Any]]

class PerformanceMonitor:
    def __init__(self, monitor_dir: str = None):
        self.monitor_dir = monitor_dir or os.path.join(os.path.dirname(__file__), '..', 'meta_learning')
        os.makedirs(self.monitor_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_history = []
        self.alert_history = []
        self.health_thresholds = {
            'response_time': 5.0,  # seconds
            'accuracy': 0.8,
            'throughput': 10.0,  # requests per minute
            'error_rate': 0.05
        }
        
        # Monitoring settings
        self.monitoring_enabled = True
        self.alert_enabled = True
        self.auto_recovery = True
        
    def record_agent_performance(self, agent_name: str, metrics: Dict[str, float]) -> str:
        """Record performance metrics for an agent."""
        if not self.monitoring_enabled:
            return None
        
        snapshot_id = f"snapshot_{int(time.time() * 1000)}"
        
        # Create performance snapshot
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            agent_metrics={agent_name: metrics},
            system_health=self._calculate_system_health(),
            alerts=self._check_alerts(agent_name, metrics)
        )
        
        self.performance_history.append(snapshot)
        self._save_performance_data()
        
        return snapshot_id
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health."""
        if not self.performance_history:
            return {'status': 'unknown', 'score': 0.0}
        
        # Get recent snapshots (last hour)
        recent_snapshots = [s for s in self.performance_history 
                          if time.time() - s.timestamp < 3600]
        
        if not recent_snapshots:
            return {'status': 'no_data', 'score': 0.0}
        
        # Calculate health metrics
        total_agents = 0
        healthy_agents = 0
        avg_response_time = 0.0
        avg_accuracy = 0.0
        
        for snapshot in recent_snapshots:
            for agent_name, metrics in snapshot.agent_metrics.items():
                total_agents += 1
                
                # Check if agent is healthy
                if self._is_agent_healthy(metrics):
                    healthy_agents += 1
                
                # Accumulate metrics
                avg_response_time += metrics.get('response_time', 0.0)
                avg_accuracy += metrics.get('accuracy', 0.0)
        
        if total_agents > 0:
            avg_response_time /= total_agents
            avg_accuracy /= total_agents
        
        health_score = healthy_agents / total_agents if total_agents > 0 else 0.0
        
        return {
            'status': 'healthy' if health_score > 0.8 else 'warning' if health_score > 0.6 else 'critical',
            'score': health_score,
            'total_agents': total_agents,
            'healthy_agents': healthy_agents,
            'avg_response_time': avg_response_time,
            'avg_accuracy': avg_accuracy
        }
    
    def _is_agent_healthy(self, metrics: Dict[str, float]) -> bool:
        """Check if an agent is performing healthily."""
        response_time = metrics.get('response_time', 0.0)
        accuracy = metrics.get('accuracy', 0.0)
        error_rate = metrics.get('error_rate', 0.0)
        
        return (response_time <= self.health_thresholds['response_time'] and
                accuracy >= self.health_thresholds['accuracy'] and
                error_rate <= self.health_thresholds['error_rate'])
    
    def _check_alerts(self, agent_name: str, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for performance alerts."""
        alerts = []
        
        # Check response time
        if metrics.get('response_time', 0.0) > self.health_thresholds['response_time']:
            alerts.append({
                'type': 'high_response_time',
                'agent': agent_name,
                'value': metrics['response_time'],
                'threshold': self.health_thresholds['response_time'],
                'severity': 'warning'
            })
        
        # Check accuracy
        if metrics.get('accuracy', 1.0) < self.health_thresholds['accuracy']:
            alerts.append({
                'type': 'low_accuracy',
                'agent': agent_name,
                'value': metrics['accuracy'],
                'threshold': self.health_thresholds['accuracy'],
                'severity': 'critical'
            })
        
        # Check error rate
        if metrics.get('error_rate', 0.0) > self.health_thresholds['error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'agent': agent_name,
                'value': metrics['error_rate'],
                'threshold': self.health_thresholds['error_rate'],
                'severity': 'critical'
            })
        
        return alerts
    
    def get_performance_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get a summary of recent performance."""
        cutoff_time = time.time() - time_window
        recent_snapshots = [s for s in self.performance_history if s.timestamp >= cutoff_time]
        
        if not recent_snapshots:
            return {'status': 'no_data', 'message': 'No recent performance data'}
        
        # Aggregate metrics
        agent_performance = defaultdict(lambda: defaultdict(list))
        total_alerts = 0
        
        for snapshot in recent_snapshots:
            for agent_name, metrics in snapshot.agent_metrics.items():
                for metric_name, value in metrics.items():
                    agent_performance[agent_name][metric_name].append(value)
            
            total_alerts += len(snapshot.alerts)
        
        # Calculate averages
        performance_summary = {}
        for agent_name, metrics in agent_performance.items():
            agent_summary = {}
            for metric_name, values in metrics.items():
                agent_summary[metric_name] = {
                    'average': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'samples': len(values)
                }
            performance_summary[agent_name] = agent_summary
        
        return {
            'status': 'success',
            'time_window': time_window,
            'total_snapshots': len(recent_snapshots),
            'total_alerts': total_alerts,
            'performance_summary': performance_summary,
            'system_health': self._calculate_system_health()
        }
    
    def _save_performance_data(self):
        """Save performance data to disk."""
        try:
            data_file = os.path.join(self.monitor_dir, 'performance_data.json')
            with open(data_file, 'w') as f:
                json.dump([asdict(s) for s in self.performance_history], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save performance data: {e}")
    
    def get_monitor_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics."""
        return {
            'total_snapshots': len(self.performance_history),
            'total_alerts': sum(len(s.alerts) for s in self.performance_history),
            'monitoring_enabled': self.monitoring_enabled,
            'alert_enabled': self.alert_enabled,
            'auto_recovery': self.auto_recovery,
            'health_thresholds': self.health_thresholds,
            'recent_health': self._calculate_system_health()
        } 