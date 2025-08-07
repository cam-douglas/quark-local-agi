# File: metrics.py
#!/usr/bin/env python3
"""
metrics.py

Basic metrics & logging for the Quark AI Assistant.
- Counts tokens in/out
- Logs timing, token usage, and category to a rolling JSON-lines file
"""
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from transformers import AutoTokenizer

# Ensure a logs/ folder exists next to this file
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Path to our JSON-lines log
LOG_PATH = os.path.join(LOG_DIR, "assistant.log")

# Touch the log file so it always exists on disk
if not os.path.exists(LOG_PATH):
    open(LOG_PATH, "w").close()

# Use a lightweight tokenizer for counting
_TOKENIZER = AutoTokenizer.from_pretrained("t5-small")


def count_tokens(text: str) -> int:
    """Return the number of input tokens (no special tokens)."""
    return len(_TOKENIZER.encode(text, add_special_tokens=False))


def log_metric(entry: dict):
    """Append a JSON-line entry to the log file."""
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


class MetricsCollector:
    """
    Comprehensive metrics collection and analysis system.
    
    Collects, stores, and analyzes various system metrics including
    performance, usage, errors, and custom metrics.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history = deque(maxlen=max_history)
        self.current_metrics = {}
        self.baseline_metrics = {}
        self.metric_types = {
            'performance': defaultdict(list),
            'usage': defaultdict(list),
            'errors': defaultdict(list),
            'custom': defaultdict(list)
        }
        
        # Initialize baseline
        self._establish_baseline()
    
    def _establish_baseline(self):
        """Establish baseline metrics for comparison"""
        self.baseline_metrics = {
            'response_time': 1.0,  # seconds
            'memory_usage': 0.5,   # percentage
            'cpu_usage': 0.3,      # percentage
            'error_rate': 0.01,    # 1%
            'success_rate': 0.95,  # 95%
            'token_usage': 100,    # tokens per request
        }
    
    def collect_metric(self, metric_type: str, metric_name: str, value: float, 
                      timestamp: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        Collect a single metric.
        
        Args:
            metric_type: Type of metric (performance, usage, errors, custom)
            metric_name: Name of the metric
            value: Metric value
            timestamp: When the metric was collected
            metadata: Additional metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_entry = {
            'type': metric_type,
            'name': metric_name,
            'value': value,
            'timestamp': timestamp.isoformat(),
            'metadata': metadata or {}
        }
        
        self.metrics_history.append(metric_entry)
        self.metric_types[metric_type][metric_name].append(value)
        
        # Keep only recent metrics in memory
        if len(self.metric_types[metric_type][metric_name]) > self.max_history:
            self.metric_types[metric_type][metric_name] = self.metric_types[metric_type][metric_name][-self.max_history:]
    
    def record_metrics(self, metrics: Dict[str, float], metric_type: str = "performance", 
                      timestamp: Optional[datetime] = None, metadata: Optional[Dict[str, Any]] = None):
        """Record multiple metrics at once"""
        if timestamp is None:
            timestamp = datetime.now()
        
        for metric_name, value in metrics.items():
            self.collect_metric(metric_type, metric_name, value, timestamp, metadata)
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            'response_time': self._get_latest_metric('performance', 'response_time', 1.0),
            'memory_usage': self._get_latest_metric('performance', 'memory_usage', 0.5),
            'cpu_usage': self._get_latest_metric('performance', 'cpu_usage', 0.3),
            'error_rate': self._get_latest_metric('errors', 'error_rate', 0.01),
            'success_rate': self._get_latest_metric('performance', 'success_rate', 0.95),
            'token_usage': self._get_latest_metric('usage', 'token_usage', 100),
        }
    
    def _get_latest_metric(self, metric_type: str, metric_name: str, default: float) -> float:
        """Get the latest value for a specific metric"""
        if metric_name in self.metric_types[metric_type]:
            values = self.metric_types[metric_type][metric_name]
            return values[-1] if values else default
        return default
    
    def get_metric_history(self, metric_type: str, metric_name: str, 
                          limit: Optional[int] = None) -> List[float]:
        """Get history for a specific metric"""
        if metric_name in self.metric_types[metric_type]:
            values = self.metric_types[metric_type][metric_name]
            if limit:
                return values[-limit:]
            return values
        return []
    
    def calculate_trend(self, metric_type: str, metric_name: str, window: int = 10) -> Dict[str, float]:
        """Calculate trend for a metric over a window"""
        values = self.get_metric_history(metric_type, metric_name, window)
        if len(values) < 2:
            return {'trend': 0.0, 'change_percent': 0.0}
        
        recent_avg = sum(values[-5:]) / min(5, len(values[-5:]))
        older_avg = sum(values[:-5]) / max(1, len(values[:-5]))
        
        if older_avg == 0:
            return {'trend': 0.0, 'change_percent': 0.0}
        
        change_percent = ((recent_avg - older_avg) / older_avg) * 100
        
        return {
            'trend': recent_avg - older_avg,
            'change_percent': change_percent,
            'current_avg': recent_avg,
            'baseline_avg': older_avg
        }
    
    def detect_anomalies(self, metric_type: str, metric_name: str, 
                        threshold: float = 2.0) -> List[Dict[str, Any]]:
        """Detect anomalies in metric values"""
        values = self.get_metric_history(metric_type, metric_name)
        if len(values) < 10:
            return []
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        
        anomalies = []
        for i, value in enumerate(values):
            if abs(value - mean) > threshold * std_dev:
                anomalies.append({
                    'index': i,
                    'value': value,
                    'deviation': abs(value - mean),
                    'z_score': (value - mean) / std_dev if std_dev > 0 else 0
                })
        
        return anomalies
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a comprehensive performance summary"""
        current_metrics = self.get_current_metrics()
        
        summary = {
            'current_metrics': current_metrics,
            'baseline_metrics': self.baseline_metrics,
            'trends': {},
            'anomalies': {},
            'recommendations': []
        }
        
        # Calculate trends for key metrics
        for metric_name in current_metrics:
            if metric_name in ['response_time', 'memory_usage', 'cpu_usage', 'error_rate']:
                trend = self.calculate_trend('performance', metric_name)
                summary['trends'][metric_name] = trend
                
                # Check for concerning trends
                if trend['change_percent'] > 20:  # 20% increase
                    summary['recommendations'].append(
                        f"High increase in {metric_name}: {trend['change_percent']:.1f}%"
                    )
        
        # Detect anomalies
        for metric_name in ['response_time', 'memory_usage', 'cpu_usage']:
            anomalies = self.detect_anomalies('performance', metric_name)
            if anomalies:
                summary['anomalies'][metric_name] = anomalies
        
        return summary
    
    def export_metrics(self, filename: Optional[str] = None) -> str:
        """Export all metrics to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_export_{timestamp}.json"
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'metrics_history': list(self.metrics_history),
            'current_metrics': self.get_current_metrics(),
            'baseline_metrics': self.baseline_metrics,
            'metric_types': dict(self.metric_types)
        }
        
        export_path = os.path.join(LOG_DIR, filename)
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_path
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        self.metrics_history.clear()
        for metric_type in self.metric_types:
            self.metric_types[metric_type].clear()
        self._establish_baseline()

