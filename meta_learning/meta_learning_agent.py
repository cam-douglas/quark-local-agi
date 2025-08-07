#!/usr/bin/env python3
"""
Meta-Learning Agent for Quark AI Assistant
Handles self-monitoring, performance introspection, and meta-learning capabilities

Part of Pillar 16: Meta-Learning & Self-Reflection
"""

import time
import json
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from agents.base import Agent
from core.safety_guardrails import SafetyGuardrails, ChangeType, ChangeSeverity

@dataclass
class PerformanceMetric:
    """A performance metric for meta-learning."""
    metric_name: str
    value: float
    timestamp: float
    context: Dict[str, Any]
    agent_name: str
    category: str

@dataclass
class LearningInsight:
    """An insight derived from meta-learning analysis."""
    insight_id: str
    insight_type: str  # performance_gap, optimization_opportunity, pattern_recognition
    description: str
    confidence: float
    actionable: bool
    priority: str  # low, medium, high, critical
    timestamp: float
    metadata: Dict[str, Any]

class MetaLearningAgent(Agent):
    def __init__(self, model_manager=None, learning_dir: str = None):
        super().__init__(model_manager)
        self.learning_dir = learning_dir or os.path.join(os.path.dirname(__file__), '..', 'meta_learning')
        os.makedirs(self.learning_dir, exist_ok=True)
        
        # Meta-learning data storage
        self.performance_metrics = []
        self.learning_insights = []
        self.optimization_history = []
        
        # Meta-learning settings
        self.meta_learning_enabled = True
        self.auto_optimization = True
        self.performance_threshold = 0.8
        self.optimization_cooldown = 3600  # 1 hour between optimizations
        
        # Performance tracking
        self.baseline_performance = {}
        self.current_performance = {}
        self.performance_trends = {}
        
        # Learning capabilities
        self.learning_capabilities = {
            'performance_monitoring': True,
            'pattern_recognition': True,
            'pipeline_optimization': True,
            'self_reflection': True,
            'adaptive_learning': True
        }
        
        # Initialize safety guardrails
        self.safety_guardrails = SafetyGuardrails()
        
        # Load existing data
        self._load_meta_learning_data()
        
    def _load_meta_learning_data(self):
        """Load existing meta-learning data from disk."""
        try:
            # Load performance metrics
            metrics_file = os.path.join(self.learning_dir, 'performance_metrics.json')
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                    self.performance_metrics = [PerformanceMetric(**m) for m in data]
            
            # Load learning insights
            insights_file = os.path.join(self.learning_dir, 'learning_insights.json')
            if os.path.exists(insights_file):
                with open(insights_file, 'r') as f:
                    data = json.load(f)
                    self.learning_insights = [LearningInsight(**i) for i in data]
                    
        except Exception as e:
            print(f"Warning: Could not load meta-learning data: {e}")
    
    def _save_meta_learning_data(self):
        """Save meta-learning data to disk."""
        try:
            # Save performance metrics
            metrics_file = os.path.join(self.learning_dir, 'performance_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump([asdict(m) for m in self.performance_metrics], f, indent=2)
            
            # Save learning insights
            insights_file = os.path.join(self.learning_dir, 'learning_insights.json')
            with open(insights_file, 'w') as f:
                json.dump([asdict(i) for i in self.learning_insights], f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save meta-learning data: {e}")
    
    def add_performance_metric(self, metric_name: str, value: float, 
                              agent_name: str, category: str = "general",
                              context: Dict[str, Any] = None) -> str:
        """Add a performance metric for meta-learning analysis."""
        if not self.meta_learning_enabled:
            return None
        
        metric_id = f"metric_{int(time.time() * 1000)}"
        metric = PerformanceMetric(
            metric_name=metric_name,
            value=value,
            timestamp=time.time(),
            context=context or {},
            agent_name=agent_name,
            category=category
        )
        
        self.performance_metrics.append(metric)
        self._save_meta_learning_data()
        
        return metric_id
    
    def analyze_performance_trends(self, time_window: int = 86400) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if not self.performance_metrics:
            return {'status': 'no_data', 'message': 'No performance metrics available'}
        
        # Filter recent metrics
        cutoff_time = time.time() - time_window
        recent_metrics = [m for m in self.performance_metrics if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return {'status': 'no_recent_data', 'message': 'No recent performance metrics'}
        
        # Group by agent and metric
        agent_metrics = defaultdict(lambda: defaultdict(list))
        for metric in recent_metrics:
            agent_metrics[metric.agent_name][metric.metric_name].append(metric.value)
        
        # Calculate trends
        trends = {}
        for agent_name, metrics in agent_metrics.items():
            agent_trends = {}
            for metric_name, values in metrics.items():
                if len(values) >= 2:
                    # Calculate trend (positive = improving, negative = declining)
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    avg_value = np.mean(values)
                    std_value = np.std(values)
                    
                    agent_trends[metric_name] = {
                        'trend': trend,
                        'average': avg_value,
                        'std': std_value,
                        'samples': len(values),
                        'improving': trend > 0,
                        'stable': abs(trend) < 0.01
                    }
            
            trends[agent_name] = agent_trends
        
        return {
            'status': 'success',
            'time_window': time_window,
            'total_metrics': len(recent_metrics),
            'trends': trends,
            'overall_health': self._calculate_overall_health(trends)
        }
    
    def _calculate_overall_health(self, trends: Dict) -> Dict[str, Any]:
        """Calculate overall system health based on trends."""
        improving_metrics = 0
        declining_metrics = 0
        stable_metrics = 0
        total_metrics = 0
        
        for agent_trends in trends.values():
            for metric_data in agent_trends.values():
                total_metrics += 1
                if metric_data['improving']:
                    improving_metrics += 1
                elif metric_data['stable']:
                    stable_metrics += 1
                else:
                    declining_metrics += 1
        
        health_score = 0.0
        if total_metrics > 0:
            health_score = (improving_metrics + stable_metrics * 0.5) / total_metrics
        
        return {
            'health_score': health_score,
            'improving_metrics': improving_metrics,
            'declining_metrics': declining_metrics,
            'stable_metrics': stable_metrics,
            'total_metrics': total_metrics,
            'status': 'healthy' if health_score > 0.7 else 'warning' if health_score > 0.5 else 'critical'
        }
    
    def generate_learning_insights(self) -> List[LearningInsight]:
        """Generate insights from performance analysis."""
        insights = []
        
        # Analyze performance trends
        trends_analysis = self.analyze_performance_trends()
        if trends_analysis['status'] != 'success':
            return insights
        
        trends = trends_analysis['trends']
        health = trends_analysis['overall_health']
        
        # Generate insights based on trends
        for agent_name, agent_trends in trends.items():
            for metric_name, metric_data in agent_trends.items():
                insight = self._generate_metric_insight(agent_name, metric_name, metric_data)
                if insight:
                    insights.append(insight)
        
        # Generate system-level insights
        system_insights = self._generate_system_insights(health, trends)
        insights.extend(system_insights)
        
        # Store insights
        self.learning_insights.extend(insights)
        self._save_meta_learning_data()
        
        return insights
    
    def _generate_metric_insight(self, agent_name: str, metric_name: str, 
                                metric_data: Dict) -> Optional[LearningInsight]:
        """Generate insight for a specific metric."""
        insight_id = f"insight_{int(time.time() * 1000)}"
        
        if metric_data['improving'] and metric_data['trend'] > 0.05:
            return LearningInsight(
                insight_id=insight_id,
                insight_type="positive_trend",
                description=f"{agent_name} {metric_name} is showing strong improvement",
                confidence=min(0.9, abs(metric_data['trend']) * 10),
                actionable=False,
                priority="low",
                timestamp=time.time(),
                metadata={'agent': agent_name, 'metric': metric_name, 'trend': metric_data['trend']}
            )
        
        elif not metric_data['improving'] and metric_data['trend'] < -0.05:
            return LearningInsight(
                insight_id=insight_id,
                insight_type="performance_decline",
                description=f"{agent_name} {metric_name} is declining and needs attention",
                confidence=min(0.9, abs(metric_data['trend']) * 10),
                actionable=True,
                priority="high",
                timestamp=time.time(),
                metadata={'agent': agent_name, 'metric': metric_name, 'trend': metric_data['trend']}
            )
        
        elif metric_data['std'] > metric_data['average'] * 0.3:
            return LearningInsight(
                insight_id=insight_id,
                insight_type="high_variability",
                description=f"{agent_name} {metric_name} shows high variability",
                confidence=0.7,
                actionable=True,
                priority="medium",
                timestamp=time.time(),
                metadata={'agent': agent_name, 'metric': metric_name, 'std': metric_data['std']}
            )
        
        return None
    
    def _generate_system_insights(self, health: Dict, trends: Dict) -> List[LearningInsight]:
        """Generate system-level insights."""
        insights = []
        
        if health['health_score'] < 0.5:
            insights.append(LearningInsight(
                insight_id=f"insight_{int(time.time() * 1000)}",
                insight_type="system_health_warning",
                description="Overall system performance is declining",
                confidence=0.8,
                actionable=True,
                priority="critical",
                timestamp=time.time(),
                metadata={'health_score': health['health_score']}
            ))
        
        if health['declining_metrics'] > health['improving_metrics']:
            insights.append(LearningInsight(
                insight_id=f"insight_{int(time.time() * 1000)}",
                insight_type="optimization_needed",
                description="More metrics are declining than improving",
                confidence=0.7,
                actionable=True,
                priority="high",
                timestamp=time.time(),
                metadata={'declining': health['declining_metrics'], 'improving': health['improving_metrics']}
            ))
        
        return insights
    
    def get_meta_learning_statistics(self) -> Dict[str, Any]:
        """Get meta-learning statistics."""
        return {
            'total_metrics': len(self.performance_metrics),
            'total_insights': len(self.learning_insights),
            'total_optimizations': len(self.optimization_history),
            'recent_insights': len([i for i in self.learning_insights if time.time() - i.timestamp < 86400]),
            'system_health': self.analyze_performance_trends().get('overall_health', {}),
            'learning_capabilities': self.learning_capabilities,
            'meta_learning_enabled': self.meta_learning_enabled,
            'auto_optimization': self.auto_optimization
        }
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Main meta-learning agent interface."""
        operation = kwargs.get('operation', 'analyze')
        
        if operation == 'add_metric':
            metric_name = kwargs.get('metric_name', 'unknown')
            value = kwargs.get('value', 0.0)
            agent_name = kwargs.get('agent_name', 'unknown')
            category = kwargs.get('category', 'general')
            context = kwargs.get('context', {})
            
            metric_id = self.add_performance_metric(metric_name, value, agent_name, category, context)
            return {
                'operation': 'add_metric',
                'metric_id': metric_id,
                'success': metric_id is not None
            }
        
        elif operation == 'analyze_trends':
            time_window = kwargs.get('time_window', 86400)
            return self.analyze_performance_trends(time_window)
        
        elif operation == 'generate_insights':
            insights = self.generate_learning_insights()
            return {
                'operation': 'generate_insights',
                'insights_count': len(insights),
                'insights': [asdict(i) for i in insights]
            }
        
        elif operation == 'statistics':
            return self.get_meta_learning_statistics()
        
        else:
            # Default: analyze trends and generate insights
            trends = self.analyze_performance_trends()
            insights = self.generate_learning_insights()
            
            return {
                'operation': 'analyze',
                'trends_analysis': trends,
                'insights_count': len(insights),
                'insights': [asdict(i) for i in insights],
                'meta_learning_status': 'active' if self.meta_learning_enabled else 'disabled'
            } 