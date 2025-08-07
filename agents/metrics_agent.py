#!/usr/bin/env python3
"""
Metrics Agent for Quark AI Assistant
Handles performance monitoring, error tracking, and evaluation metrics
"""

import time
import json
import statistics
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import os
import logging

logger = logging.getLogger(__name__)

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
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0

@dataclass
class ModelPerformance:
    """Model-specific performance metrics."""
    model_name: str
    total_requests: int
    successful_requests: int
    average_latency: float
    average_tokens: int
    error_rate: float
    last_used: float

from .base import Agent

class MetricsAgent(Agent):
    def __init__(self, model_name: str = "metrics_agent", metrics_dir: str = None):
        super().__init__(model_name)
        self.metrics_dir = metrics_dir or os.path.join(os.path.dirname(__file__), '..', 'metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Performance tracking
        self.performance_history = defaultdict(list)
        self.error_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=1000)
        self.model_performance = defaultdict(lambda: ModelPerformance(
            model_name="", total_requests=0, successful_requests=0,
            average_latency=0.0, average_tokens=0, error_rate=0.0, last_used=0.0
        ))
        
        # Real-time metrics
        self.current_metrics = {
            'requests_per_minute': 0,
            'average_latency': 0.0,
            'error_rate': 0.0,
            'success_rate': 0.0,
            'total_requests': 0,
            'total_errors': 0,
            'memory_usage_mb': 0.0,
            'cpu_usage_percent': 0.0,
            'gpu_usage_percent': 0.0
        }
        
        # Metric collection settings
        self.metrics_enabled = True
        self.retention_days = 30
        self.real_time_update_interval = 5  # seconds
        
        # System monitoring
        self.system_monitor = SystemMonitor()
        
        # Real-time metrics thread
        self.metrics_thread = None
        self.metrics_running = False
        
        # Initialize without model since this is a metrics agent
        self.model_name = None
        self.model = None
        
        # Start real-time monitoring
        self._start_real_time_monitoring()
        
    def _start_real_time_monitoring(self):
        """Start real-time metrics monitoring thread."""
        if self.metrics_thread is None or not self.metrics_thread.is_alive():
            self.metrics_running = True
            self.metrics_thread = threading.Thread(target=self._monitor_real_time_metrics, daemon=True)
            self.metrics_thread.start()
    
    def _monitor_real_time_metrics(self):
        """Monitor real-time system metrics."""
        while self.metrics_running:
            try:
                # Update system metrics
                self.current_metrics['memory_usage_mb'] = self.system_monitor.get_memory_usage()
                self.current_metrics['cpu_usage_percent'] = self.system_monitor.get_cpu_usage()
                self.current_metrics['gpu_usage_percent'] = self.system_monitor.get_gpu_usage()
                
                # Calculate request rate
                recent_requests = [m for m in self.performance_history.values() 
                                 if m and time.time() - m[-1].end_time < 60]
                self.current_metrics['requests_per_minute'] = len(recent_requests)
                
                # Calculate error rate
                if self.current_metrics['total_requests'] > 0:
                    self.current_metrics['error_rate'] = (
                        self.current_metrics['total_errors'] / self.current_metrics['total_requests']
                    )
                    self.current_metrics['success_rate'] = 1 - self.current_metrics['error_rate']
                
                time.sleep(self.real_time_update_interval)
                
            except Exception as e:
                logger.error(f"Error in real-time metrics monitoring: {e}")
                time.sleep(self.real_time_update_interval)
    
    def start_operation(self, operation: str, input_data: str = "", model_name: str = None) -> str:
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
            input_length=len(input_data),
            memory_usage=self.system_monitor.get_memory_usage(),
            cpu_usage=self.system_monitor.get_cpu_usage(),
            gpu_usage=self.system_monitor.get_gpu_usage()
        )
        
        self.performance_history[operation_id].append(metrics)
        
        # Track model usage
        if model_name:
            model_perf = self.model_performance[model_name]
            model_perf.model_name = model_name
            model_perf.total_requests += 1
            model_perf.last_used = time.time()
        
        return operation_id
        
    def end_operation(self, operation_id: str, success: bool = True, 
                     output_data: str = "", error_message: str = None,
                     tokens_used: int = 0, memory_usage: float = 0.0,
                     model_name: str = None):
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
                    'error': error_message
                })
                self.current_metrics['total_errors'] += 1
            
            self.latency_history.append(metrics.duration)
            self.current_metrics['total_requests'] += 1
            
            # Update model performance
            if model_name and model_name in self.model_performance:
                model_perf = self.model_performance[model_name]
                if success:
                    model_perf.successful_requests += 1
                
                # Update average latency
                total_latency = model_perf.average_latency * (model_perf.total_requests - 1)
                model_perf.average_latency = (total_latency + metrics.duration) / model_perf.total_requests
                
                # Update average tokens
                total_tokens = model_perf.average_tokens * (model_perf.total_requests - 1)
                model_perf.average_tokens = (total_tokens + tokens_used) / model_perf.total_requests
                
                # Update error rate
                model_perf.error_rate = 1 - (model_perf.successful_requests / model_perf.total_requests)
            
            self._update_real_time_metrics(metrics)
    
    def _update_real_time_metrics(self, metrics: PerformanceMetrics):
        """Update real-time metrics based on operation results."""
        # Update average latency
        if self.latency_history:
            self.current_metrics['average_latency'] = statistics.mean(self.latency_history)
    
    def record_intent_classification(self, category: str, confidence: float, input_text: str):
        """Record intent classification metrics."""
        if not self.metrics_enabled:
            return
            
        intent_metrics = {
            'timestamp': time.time(),
            'category': category,
            'confidence': confidence,
            'input_length': len(input_text),
            'success': confidence > 0.5
        }
        
        # Store intent classification metrics
        intent_file = os.path.join(self.metrics_dir, 'intent_classifications.jsonl')
        try:
            with open(intent_file, 'a') as f:
                f.write(json.dumps(intent_metrics) + '\n')
        except Exception as e:
            logger.error(f"Error recording intent classification: {e}")
    
    def record_request_metrics(self, category: str, pipeline_length: int, agent_results: Dict[str, Any]):
        """Record comprehensive request metrics."""
        if not self.metrics_enabled:
            return
        
        # Convert AgentResult objects to serializable dictionaries
        serializable_results = {}
        for agent_name, result in agent_results.items():
            if hasattr(result, '__dict__'):
                # Convert AgentResult to dict
                serializable_results[agent_name] = {
                    'agent_name': getattr(result, 'agent_name', agent_name),
                    'success': getattr(result, 'success', False),
                    'execution_time': getattr(result, 'execution_time', 0.0),
                    'error': getattr(result, 'error', None),
                    'metadata': getattr(result, 'metadata', {})
                }
            else:
                # Already a dict or other serializable type
                serializable_results[agent_name] = result
            
        request_metrics = {
            'timestamp': time.time(),
            'category': category,
            'pipeline_length': pipeline_length,
            'agent_results': serializable_results,
            'success': all(result.get('success', False) for result in serializable_results.values() if isinstance(result, dict))
        }
        
        # Store request metrics
        request_file = os.path.join(self.metrics_dir, 'request_metrics.jsonl')
        try:
            with open(request_file, 'a') as f:
                f.write(json.dumps(request_metrics) + '\n')
        except Exception as e:
            logger.error(f"Error recording request metrics: {e}")
    
    def get_performance_summary(self, operation: str = None, 
                              time_window: int = 3600) -> Dict[str, Any]:
        """Get performance summary for specified time window."""
        cutoff_time = time.time() - time_window
        relevant_metrics = []
        
        for operation_id, metrics_list in self.performance_history.items():
            for metrics in metrics_list:
                if metrics.start_time >= cutoff_time:
                    if operation is None or metrics.operation == operation:
                        relevant_metrics.append(metrics)
        
        if not relevant_metrics:
            return {
                'total_operations': 0,
                'success_rate': 0.0,
                'average_latency': 0.0,
                'total_tokens': 0,
                'error_count': 0
            }
        
        successful_ops = sum(1 for m in relevant_metrics if m.success)
        total_ops = len(relevant_metrics)
        total_tokens = sum(m.tokens_used for m in relevant_metrics)
        error_count = sum(1 for m in relevant_metrics if not m.success)
        
        return {
            'total_operations': total_ops,
            'success_rate': successful_ops / total_ops if total_ops > 0 else 0.0,
            'average_latency': statistics.mean([m.duration for m in relevant_metrics]),
            'total_tokens': total_tokens,
            'error_count': error_count,
            'time_window_seconds': time_window
        }
    
    def get_error_summary(self, time_window: int = 3600) -> Dict[str, Any]:
        """Get error summary for specified time window."""
        cutoff_time = time.time() - time_window
        recent_errors = [error for error in self.error_history 
                        if error['timestamp'] >= cutoff_time]
        
        if not recent_errors:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'common_errors': [],
                'error_by_operation': {}
            }
        
        # Group errors by operation
        error_by_operation = defaultdict(int)
        for error in recent_errors:
            error_by_operation[error['operation']] += 1
        
        # Find common errors
        error_messages = [error['error'] for error in recent_errors]
        error_counts = defaultdict(int)
        for error_msg in error_messages:
            error_counts[error_msg] += 1
        
        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_errors': len(recent_errors),
            'error_rate': len(recent_errors) / max(len(self.latency_history), 1),
            'common_errors': common_errors,
            'error_by_operation': dict(error_by_operation),
            'time_window_seconds': time_window
        }
    
    def get_latency_analysis(self, operation: str = None, 
                           time_window: int = 3600) -> Dict[str, Any]:
        """Get detailed latency analysis."""
        cutoff_time = time.time() - time_window
        relevant_metrics = []
        
        for operation_id, metrics_list in self.performance_history.items():
            for metrics in metrics_list:
                if metrics.start_time >= cutoff_time:
                    if operation is None or metrics.operation == operation:
                        relevant_metrics.append(metrics)
        
        if not relevant_metrics:
            return {
                'average_latency': 0.0,
                'p95_latency': 0.0,
                'p99_latency': 0.0,
                'min_latency': 0.0,
                'max_latency': 0.0,
                'latency_distribution': []
            }
        
        latencies = [m.duration for m in relevant_metrics]
        latencies.sort()
        
        return {
            'average_latency': statistics.mean(latencies),
            'p95_latency': latencies[int(len(latencies) * 0.95)] if latencies else 0.0,
            'p99_latency': latencies[int(len(latencies) * 0.99)] if latencies else 0.0,
            'min_latency': min(latencies),
            'max_latency': max(latencies),
            'latency_distribution': self._calculate_percentiles(latencies)
        }
    
    def _calculate_percentiles(self, values: List[float]) -> List[Dict[str, float]]:
        """Calculate percentile distribution."""
        if not values:
            return []
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        distribution = []
        
        for p in percentiles:
            index = int(len(values) * p / 100)
            distribution.append({
                'percentile': p,
                'value': values[index] if index < len(values) else values[-1]
            })
        
        return distribution
    
    def get_model_performance(self, model_name: str = None) -> Dict[str, Any]:
        """Get model-specific performance metrics."""
        if model_name:
            if model_name in self.model_performance:
                model_perf = self.model_performance[model_name]
                return {
                    'model_name': model_perf.model_name,
                    'total_requests': model_perf.total_requests,
                    'successful_requests': model_perf.successful_requests,
                    'average_latency': model_perf.average_latency,
                    'average_tokens': model_perf.average_tokens,
                    'error_rate': model_perf.error_rate,
                    'last_used': model_perf.last_used,
                    'uptime_hours': (time.time() - model_perf.last_used) / 3600 if model_perf.last_used > 0 else 0
                }
            else:
                return {'error': f'Model {model_name} not found'}
        
        # Return all model performance
        return {
            model_name: {
                'total_requests': perf.total_requests,
                'successful_requests': perf.successful_requests,
                'average_latency': perf.average_latency,
                'error_rate': perf.error_rate,
                'last_used': perf.last_used
            }
            for model_name, perf in self.model_performance.items()
        }
    
    def get_accuracy_metrics(self, predictions: List[Dict], 
                           ground_truth: List[Dict]) -> Dict[str, float]:
        """Calculate accuracy metrics for predictions."""
        if len(predictions) != len(ground_truth):
            return {'error': 'Predictions and ground truth must have same length'}
        
        correct = 0
        total = len(predictions)
        
        for pred, truth in zip(predictions, ground_truth):
            if pred.get('prediction') == truth.get('expected'):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct_predictions': correct,
            'total_predictions': total,
            'error_rate': 1 - accuracy
        }
    
    def get_user_satisfaction_metrics(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """Calculate user satisfaction metrics."""
        if not feedback_data:
            return {'error': 'No feedback data provided'}
        
        satisfaction_scores = [feedback.get('satisfaction_score', 0) for feedback in feedback_data]
        feedback_categories = defaultdict(list)
        
        for feedback in feedback_data:
            category = feedback.get('category', 'general')
            feedback_categories[category].append(feedback.get('satisfaction_score', 0))
        
        return {
            'average_satisfaction': statistics.mean(satisfaction_scores),
            'satisfaction_by_category': {
                category: statistics.mean(scores) 
                for category, scores in feedback_categories.items()
            },
            'total_feedback': len(feedback_data),
            'positive_feedback_rate': sum(1 for score in satisfaction_scores if score >= 4) / len(satisfaction_scores)
        }
    
    def export_metrics(self, filename: str = None) -> str:
        """Export all metrics to a file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.json"
        
        export_path = os.path.join(self.metrics_dir, filename)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'current_metrics': self.current_metrics,
            'model_performance': {
                name: asdict(perf) for name, perf in self.model_performance.items()
            },
            'performance_summary': self.get_performance_summary(),
            'error_summary': self.get_error_summary(),
            'latency_analysis': self.get_latency_analysis()
        }
        
        try:
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            return export_path
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            return None
    
    def load_model(self):
        """Load metrics models and components."""
        # Metrics agent doesn't need a specific model
        return True
    
    def _ensure_model(self):
        """Ensure the metrics system is initialized."""
        return True
    
    def clear_old_metrics(self, days: int = None):
        """Clear old metrics data."""
        days = days or self.retention_days
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        # Clear old performance history
        for operation_id in list(self.performance_history.keys()):
            self.performance_history[operation_id] = [
                m for m in self.performance_history[operation_id] 
                if m.start_time >= cutoff_time
            ]
            if not self.performance_history[operation_id]:
                del self.performance_history[operation_id]
        
        # Clear old error history
        self.error_history = deque(
            [error for error in self.error_history if error['timestamp'] >= cutoff_time],
            maxlen=1000
        )
        
        # Clear old latency history
        self.latency_history = deque(
            [latency for latency in self.latency_history],
            maxlen=1000
        )
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate metrics report based on prompt."""
        operation = kwargs.get('operation', 'summary')
        
        if operation == 'summary':
            return {
                'operation': 'summary',
                'current_metrics': self.current_metrics,
                'performance_summary': self.get_performance_summary(),
                'error_summary': self.get_error_summary(),
                'model_performance': self.get_model_performance()
            }
        
        elif operation == 'export':
            filename = kwargs.get('filename')
            export_path = self.export_metrics(filename)
            return {
                'operation': 'export',
                'success': export_path is not None,
                'file_path': export_path
            }
        
        elif operation == 'clear':
            days = kwargs.get('days', self.retention_days)
            self.clear_old_metrics(days)
            return {
                'operation': 'clear',
                'success': True,
                'retention_days': days
            }
        
        else:
            return {
                'operation': 'unknown',
                'error': f'Unknown operation: {operation}'
            }
    
    def shutdown(self):
        """Shutdown the metrics agent."""
        self.metrics_running = False
        if self.metrics_thread and self.metrics_thread.is_alive():
            self.metrics_thread.join(timeout=5)


class SystemMonitor:
    """Monitor system resources."""
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            memory_info = self.process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except Exception:
            return 0.0
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self.process.cpu_percent(interval=0.1)
        except Exception:
            return 0.0
    
    def get_gpu_usage(self) -> float:
        """Get current GPU usage percentage."""
        try:
            # This is a placeholder - would need GPU monitoring library like nvidia-ml-py
            return 0.0
        except Exception:
            return 0.0

