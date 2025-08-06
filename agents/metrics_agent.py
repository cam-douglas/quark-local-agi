#!/usr/bin/env python3
"""
Metrics Agent for Meta-Model AI Assistant
Handles performance monitoring, error tracking, and evaluation metrics
"""

import time
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import os

@dataclass
class MetricPoint:
    """A single metric measurement point."""
    timestamp: float
    value: float
    metadata: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    input_length: int = 0
    output_length: int = 0
    tokens_used: int = 0
    memory_usage: float = 0.0

class MetricsAgent:
    def __init__(self, metrics_dir: str = None):
        self.metrics_dir = metrics_dir or os.path.join(os.path.dirname(__file__), '..', 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.error_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=1000)
        
        # Real-time metrics
        self.current_metrics = {
            'requests_per_minute': 0,
            'average_latency': 0.0,
            'error_rate': 0.0,
            'success_rate': 0.0,
            'total_requests': 0,
            'total_errors': 0
        }
        
        # Metric collection settings
        self.metrics_enabled = True
        self.retention_days = 30
        
        # Initialize without model since this is a metrics agent
        self.model_name = None
        self.model = None
        
    def start_operation(self, operation: str, input_data: str = "") -> str:
        """Start timing an operation and return operation ID."""
        if not self.metrics_enabled:
            return "disabled"
            
        operation_id = f"{operation}_{int(time.time() * 1000)}"
        
        metrics = PerformanceMetrics(
            operation=operation,
            start_time=time.time(),
            end_time=0.0,
            duration=0.0,
            success=False,
            input_length=len(input_data)
        )
        
        self.performance_history[operation_id].append(metrics)
        return operation_id
        
    def end_operation(self, operation_id: str, success: bool = True, 
                     output_data: str = "", error_message: str = None,
                     tokens_used: int = 0, memory_usage: float = 0.0):
        """End timing an operation and record results."""
        if not self.metrics_enabled or operation_id == "disabled":
            return
            
        if operation_id in self.performance_history:
            metrics = self.performance_history[operation_id][-1]
            metrics.end_time = time.time()
            metrics.duration = metrics.end_time - metrics.start_time
            metrics.success = success
            metrics.output_length = len(output_data)
            metrics.tokens_used = tokens_used
            metrics.memory_usage = memory_usage
            
            if error_message:
                metrics.error_message = error_message
                self.error_history.append({
                    'timestamp': time.time(),
                    'operation': metrics.operation,
                    'error': error_message,
                    'duration': metrics.duration
                })
            
            # Update real-time metrics
            self._update_real_time_metrics(metrics)
            
    def _update_real_time_metrics(self, metrics: PerformanceMetrics):
        """Update real-time metrics based on new performance data."""
        self.current_metrics['total_requests'] += 1
        
        if metrics.success:
            self.current_metrics['success_rate'] = (
                (self.current_metrics['total_requests'] - self.current_metrics['total_errors']) /
                self.current_metrics['total_requests']
            )
        else:
            self.current_metrics['total_errors'] += 1
            self.current_metrics['error_rate'] = (
                self.current_metrics['total_errors'] / self.current_metrics['total_requests']
            )
        
        # Update latency history
        self.latency_history.append(metrics.duration)
        
        # Calculate average latency
        if self.latency_history:
            self.current_metrics['average_latency'] = statistics.mean(self.latency_history)
            
    def get_performance_summary(self, operation: str = None, 
                              time_window: int = 3600) -> Dict[str, Any]:
        """Get performance summary for operations."""
        cutoff_time = time.time() - time_window
        
        if operation:
            # Filter by specific operation
            relevant_metrics = [
                m for m in self.performance_history.get(operation, [])
                if m.start_time >= cutoff_time
            ]
        else:
            # Get all metrics in time window
            relevant_metrics = []
            for metrics_list in self.performance_history.values():
                relevant_metrics.extend([
                    m for m in metrics_list
                    if m.start_time >= cutoff_time
                ])
        
        if not relevant_metrics:
            return {
                'total_operations': 0,
                'success_rate': 0.0,
                'average_latency': 0.0,
                'error_rate': 0.0,
                'total_errors': 0
            }
        
        successful_ops = [m for m in relevant_metrics if m.success]
        failed_ops = [m for m in relevant_metrics if not m.success]
        
        return {
            'total_operations': len(relevant_metrics),
            'successful_operations': len(successful_ops),
            'failed_operations': len(failed_ops),
            'success_rate': len(successful_ops) / len(relevant_metrics) if relevant_metrics else 0.0,
            'error_rate': len(failed_ops) / len(relevant_metrics) if relevant_metrics else 0.0,
            'average_latency': statistics.mean([m.duration for m in relevant_metrics]) if relevant_metrics else 0.0,
            'min_latency': min([m.duration for m in relevant_metrics]) if relevant_metrics else 0.0,
            'max_latency': max([m.duration for m in relevant_metrics]) if relevant_metrics else 0.0,
            'total_errors': len(failed_ops),
            'average_tokens_used': statistics.mean([m.tokens_used for m in relevant_metrics if m.tokens_used > 0]) if [m.tokens_used for m in relevant_metrics if m.tokens_used > 0] else 0.0,
            'average_memory_usage': statistics.mean([m.memory_usage for m in relevant_metrics if m.memory_usage > 0]) if [m.memory_usage for m in relevant_metrics if m.memory_usage > 0] else 0.0
        }
        
    def get_error_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get summary of errors in the specified time window."""
        cutoff_time = time.time() - time_window
        
        recent_errors = [
            error for error in self.error_history
            if error['timestamp'] >= cutoff_time
        ]
        
        if not recent_errors:
            return {
                'total_errors': 0,
                'error_types': {},
                'most_common_errors': []
            }
        
        # Group errors by type
        error_types = defaultdict(int)
        for error in recent_errors:
            error_type = error['error'].split(':')[0] if ':' in error['error'] else 'Unknown'
            error_types[error_type] += 1
        
        return {
            'total_errors': len(recent_errors),
            'error_types': dict(error_types),
            'most_common_errors': sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5],
            'recent_errors': recent_errors[-10:]  # Last 10 errors
        }
        
    def get_latency_analysis(self, operation: str = None, 
                           time_window: int = 3600) -> Dict[str, Any]:
        """Get detailed latency analysis."""
        cutoff_time = time.time() - time_window
        
        if operation:
            relevant_metrics = [
                m for m in self.performance_history.get(operation, [])
                if m.start_time >= cutoff_time
            ]
        else:
            relevant_metrics = []
            for metrics_list in self.performance_history.values():
                relevant_metrics.extend([
                    m for m in metrics_list
                    if m.start_time >= cutoff_time
                ])
        
        if not relevant_metrics:
            return {
                'percentiles': {},
                'distribution': {},
                'trends': []
            }
        
        latencies = [m.duration for m in relevant_metrics]
        
        return {
            'percentiles': {
                'p50': statistics.quantiles(latencies, n=2)[0],
                'p90': statistics.quantiles(latencies, n=10)[8],
                'p95': statistics.quantiles(latencies, n=20)[18],
                'p99': statistics.quantiles(latencies, n=100)[98]
            },
            'distribution': {
                'mean': statistics.mean(latencies),
                'median': statistics.median(latencies),
                'std_dev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
                'min': min(latencies),
                'max': max(latencies)
            },
            'total_operations': len(latencies)
        }
        
    def get_accuracy_metrics(self, predictions: List[Dict], 
                           ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate accuracy metrics for predictions vs ground truth."""
        if len(predictions) != len(ground_truth):
            return {'error': 'Mismatched prediction and ground truth lengths'}
        
        correct = 0
        total = len(predictions)
        
        for pred, truth in zip(predictions, ground_truth):
            # Simple exact match for now
            if pred.get('prediction') == truth.get('expected'):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': total,
            'error_rate': 1.0 - accuracy
        }
        
    def get_user_satisfaction_metrics(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """Calculate user satisfaction metrics from feedback data."""
        if not feedback_data:
            return {
                'average_rating': 0.0,
                'total_feedback': 0,
                'rating_distribution': {},
                'satisfaction_score': 0.0
            }
        
        ratings = [item.get('rating', 0) for item in feedback_data]
        comments = [item.get('comment', '') for item in feedback_data]
        
        # Calculate satisfaction score (weighted average)
        satisfaction_score = statistics.mean(ratings) if ratings else 0.0
        
        # Rating distribution
        rating_distribution = defaultdict(int)
        for rating in ratings:
            rating_distribution[rating] += 1
        
        return {
            'average_rating': statistics.mean(ratings) if ratings else 0.0,
            'total_feedback': len(feedback_data),
            'rating_distribution': dict(rating_distribution),
            'satisfaction_score': satisfaction_score,
            'positive_feedback_rate': len([r for r in ratings if r >= 4]) / len(ratings) if ratings else 0.0
        }
        
    def export_metrics(self, filename: str = None) -> str:
        """Export all metrics to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.json"
        
        filepath = os.path.join(self.metrics_dir, filename)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'current_metrics': self.current_metrics,
            'performance_history': {
                op_id: [asdict(m) for m in metrics]
                for op_id, metrics in self.performance_history.items()
            },
            'error_history': list(self.error_history),
            'latency_history': list(self.latency_history)
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath
        
    def _ensure_model(self):
        """Ensure the metrics system is initialized."""
        return True
        
    def clear_old_metrics(self, days: int = None):
        """Clear metrics older than specified days."""
        if days is None:
            days = self.retention_days
            
        cutoff_time = time.time() - (days * 24 * 3600)
        
        # Clear old performance history
        for op_id in list(self.performance_history.keys()):
            self.performance_history[op_id] = [
                m for m in self.performance_history[op_id]
                if m.start_time >= cutoff_time
            ]
            if not self.performance_history[op_id]:
                del self.performance_history[op_id]
        
        # Clear old error history
        self.error_history = deque([
            error for error in self.error_history
            if error['timestamp'] >= cutoff_time
        ], maxlen=1000)
        
        # Clear old latency history
        self.latency_history = deque([
            latency for latency in self.latency_history
            if latency >= cutoff_time
        ], maxlen=1000)
        
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Main metrics agent interface."""
        operation = kwargs.get('operation', 'summary')
        
        if operation == 'performance_summary':
            operation_name = kwargs.get('operation_name')
            time_window = kwargs.get('time_window', 3600)
            return self.get_performance_summary(operation_name, time_window)
            
        elif operation == 'error_summary':
            time_window = kwargs.get('time_window', 3600)
            return self.get_error_summary(time_window)
            
        elif operation == 'latency_analysis':
            operation_name = kwargs.get('operation_name')
            time_window = kwargs.get('time_window', 3600)
            return self.get_latency_analysis(operation_name, time_window)
            
        elif operation == 'accuracy_metrics':
            predictions = kwargs.get('predictions', [])
            ground_truth = kwargs.get('ground_truth', [])
            return self.get_accuracy_metrics(predictions, ground_truth)
            
        elif operation == 'satisfaction_metrics':
            feedback_data = kwargs.get('feedback_data', [])
            return self.get_user_satisfaction_metrics(feedback_data)
            
        elif operation == 'export':
            filename = kwargs.get('filename')
            filepath = self.export_metrics(filename)
            return {'export_path': filepath}
            
        elif operation == 'clear_old':
            days = kwargs.get('days')
            self.clear_old_metrics(days)
            return {'status': 'cleared_old_metrics'}
            
        else:
            # Default: return current metrics summary
            return {
                'current_metrics': self.current_metrics,
                'performance_summary': self.get_performance_summary(),
                'error_summary': self.get_error_summary(),
                'latency_analysis': self.get_latency_analysis()
            }

