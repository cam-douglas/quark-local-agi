#!/usr/bin/env python3
"""
Meta-Learning Orchestrator for Quark AI Assistant
Coordinates meta-learning components and orchestrates self-improvement

Part of Pillar 16: Meta-Learning & Self-Reflection
"""

import time
import json
import os
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

from .performance_monitor import PerformanceMonitor
from .pipeline_reconfigurator import PipelineReconfigurator
from .self_reflection_agent import SelfReflectionAgent

@dataclass
class MetaLearningSession:
    """A meta-learning session."""
    session_id: str
    start_time: float
    end_time: float
    components_used: List[str]
    insights_generated: int
    optimizations_applied: int
    improvement_score: float

class MetaLearningOrchestrator:
    def __init__(self, orchestrator_dir: str = None):
        self.orchestrator_dir = orchestrator_dir or os.path.join(os.path.dirname(__file__), '..', 'meta_learning')
        os.makedirs(self.orchestrator_dir, exist_ok=True)
        
        # Initialize meta-learning components
        self.performance_monitor = PerformanceMonitor()
        self.pipeline_reconfigurator = PipelineReconfigurator()
        self.self_reflection_agent = SelfReflectionAgent()
        
        # Meta-learning sessions
        self.meta_learning_sessions = []
        
        # Orchestration settings
        self.orchestration_enabled = True
        self.auto_orchestration = True
        self.orchestration_interval = 1800  # 30 minutes between orchestrations
        self.last_orchestration = 0
        
        # Component coordination
        self.component_priorities = {
            'performance_monitor': 1,
            'self_reflection': 2,
            'pipeline_reconfiguration': 3
        }
        
        # Load existing sessions
        self._load_orchestrator_data()
        
    def _load_orchestrator_data(self):
        """Load existing orchestrator data."""
        try:
            sessions_file = os.path.join(self.orchestrator_dir, 'meta_learning_sessions.json')
            if os.path.exists(sessions_file):
                with open(sessions_file, 'r') as f:
                    data = json.load(f)
                    self.meta_learning_sessions = [MetaLearningSession(**s) for s in data]
        except Exception as e:
            print(f"Warning: Could not load orchestrator data: {e}")
    
    def _save_orchestrator_data(self):
        """Save orchestrator data to disk."""
        try:
            sessions_file = os.path.join(self.orchestrator_dir, 'meta_learning_sessions.json')
            with open(sessions_file, 'w') as f:
                json.dump([asdict(s) for s in self.meta_learning_sessions], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save orchestrator data: {e}")
    
    def run_meta_learning_session(self, session_type: str = "comprehensive") -> Dict[str, Any]:
        """Run a comprehensive meta-learning session."""
        if not self.orchestration_enabled:
            return {'status': 'disabled', 'message': 'Meta-learning orchestration is disabled'}
        
        # Check if enough time has passed
        if time.time() - self.last_orchestration < self.orchestration_interval:
            return {'status': 'too_soon', 'message': 'Orchestration interval not met'}
        
        session_id = f"meta_session_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Initialize session tracking
        components_used = []
        insights_generated = 0
        optimizations_applied = 0
        
        # Step 1: Performance Monitoring
        performance_results = self._run_performance_monitoring()
        if performance_results['status'] == 'success':
            components_used.append('performance_monitor')
            insights_generated += performance_results.get('insights_count', 0)
        
        # Step 2: Self-Reflection
        reflection_results = self._run_self_reflection()
        if reflection_results['status'] == 'completed':
            components_used.append('self_reflection')
            insights_generated += reflection_results.get('insights_count', 0)
        
        # Step 3: Pipeline Reconfiguration
        reconfiguration_results = self._run_pipeline_reconfiguration()
        if reconfiguration_results['status'] == 'completed':
            components_used.append('pipeline_reconfiguration')
            optimizations_applied += reconfiguration_results.get('applied_actions', 0)
        
        # Calculate overall improvement score
        improvement_score = self._calculate_session_improvement_score(
            performance_results, reflection_results, reconfiguration_results
        )
        
        # Create meta-learning session
        session = MetaLearningSession(
            session_id=session_id,
            start_time=start_time,
            end_time=time.time(),
            components_used=components_used,
            insights_generated=insights_generated,
            optimizations_applied=optimizations_applied,
            improvement_score=improvement_score
        )
        
        self.meta_learning_sessions.append(session)
        self.last_orchestration = time.time()
        
        self._save_orchestrator_data()
        
        return {
            'status': 'completed',
            'session_id': session_id,
            'session_type': session_type,
            'components_used': components_used,
            'insights_generated': insights_generated,
            'optimizations_applied': optimizations_applied,
            'improvement_score': improvement_score,
            'performance_results': performance_results,
            'reflection_results': reflection_results,
            'reconfiguration_results': reconfiguration_results
        }
    
    def _run_performance_monitoring(self) -> Dict[str, Any]:
        """Run performance monitoring component."""
        try:
            # Get performance summary
            performance_summary = self.performance_monitor.get_performance_summary()
            
            # Record some sample metrics for demonstration
            sample_metrics = {
                'response_time': 2.5,
                'accuracy': 0.85,
                'throughput': 15.0,
                'error_rate': 0.02
            }
            
            # Record performance for a sample agent
            self.performance_monitor.record_agent_performance('NLU', sample_metrics)
            
            return {
                'status': 'success',
                'performance_summary': performance_summary,
                'insights_count': 2,  # Sample insights
                'alerts_generated': 0
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'insights_count': 0,
                'alerts_generated': 0
            }
    
    def _run_self_reflection(self) -> Dict[str, Any]:
        """Run self-reflection component."""
        try:
            # Run a general reflection session
            reflection_results = self.self_reflection_agent.run_reflection_session(
                reflection_type="general"
            )
            
            return reflection_results
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'insights_count': 0
            }
    
    def _run_pipeline_reconfiguration(self) -> Dict[str, Any]:
        """Run pipeline reconfiguration component."""
        try:
            # Create a sample pipeline configuration if none exists
            if not self.pipeline_reconfigurator.pipeline_configurations:
                config_id = self.pipeline_reconfigurator.create_pipeline_configuration(
                    agent_sequence=['NLU', 'Retrieval', 'Reasoning'],
                    parameters={'optimization_level': 'balanced'}
                )
            else:
                config_id = self.pipeline_reconfigurator.pipeline_configurations[0].config_id
            
            # Analyze pipeline performance
            sample_performance = {
                'accuracy': 0.82,
                'response_time': 3.2,
                'throughput': 12.0,
                'user_satisfaction': 0.78
            }
            
            analysis_results = self.pipeline_reconfigurator.analyze_pipeline_performance(
                config_id, sample_performance
            )
            
            # Apply reconfiguration if needed
            applied_actions = 0
            if analysis_results.get('reconfiguration_needed', False):
                recommendations = analysis_results.get('recommendations', [])
                reconfiguration_results = self.pipeline_reconfigurator.apply_reconfiguration(
                    config_id, recommendations
                )
                applied_actions = reconfiguration_results.get('applied_actions', 0)
            
            return {
                'status': 'completed',
                'config_id': config_id,
                'analysis_results': analysis_results,
                'applied_actions': applied_actions
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'applied_actions': 0
            }
    
    def _calculate_session_improvement_score(self, performance_results: Dict[str, Any],
                                           reflection_results: Dict[str, Any],
                                           reconfiguration_results: Dict[str, Any]) -> float:
        """Calculate overall improvement score for the session."""
        score = 0.0
        total_weight = 0.0
        
        # Performance monitoring contribution
        if performance_results['status'] == 'success':
            performance_score = 0.7  # Placeholder
            score += performance_score * 0.3
            total_weight += 0.3
        
        # Self-reflection contribution
        if reflection_results['status'] == 'completed':
            reflection_score = reflection_results.get('improvement_score', 0.0)
            score += reflection_score * 0.4
            total_weight += 0.4
        
        # Pipeline reconfiguration contribution
        if reconfiguration_results['status'] == 'completed':
            reconfig_score = min(1.0, reconfiguration_results.get('applied_actions', 0) * 0.2)
            score += reconfig_score * 0.3
            total_weight += 0.3
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def get_component_status(self) -> Dict[str, Any]:
        """Get status of all meta-learning components."""
        return {
            'performance_monitor': {
                'enabled': self.performance_monitor.monitoring_enabled,
                'statistics': self.performance_monitor.get_monitor_statistics()
            },
            'pipeline_reconfigurator': {
                'enabled': self.pipeline_reconfigurator.auto_reconfiguration,
                'statistics': self.pipeline_reconfigurator.get_reconfiguration_statistics()
            },
            'self_reflection_agent': {
                'enabled': self.self_reflection_agent.reflection_enabled,
                'statistics': self.self_reflection_agent.get_reflection_statistics()
            },
            'orchestrator': {
                'enabled': self.orchestration_enabled,
                'auto_orchestration': self.auto_orchestration,
                'last_orchestration': self.last_orchestration,
                'total_sessions': len(self.meta_learning_sessions)
            }
        }
    
    def get_meta_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive meta-learning statistics."""
        if not self.meta_learning_sessions:
            return {'status': 'no_data', 'message': 'No meta-learning sessions available'}
        
        # Calculate session statistics
        total_sessions = len(self.meta_learning_sessions)
        total_insights = sum(s.insights_generated for s in self.meta_learning_sessions)
        total_optimizations = sum(s.optimizations_applied for s in self.meta_learning_sessions)
        avg_improvement_score = sum(s.improvement_score for s in self.meta_learning_sessions) / total_sessions
        
        # Get recent sessions (last 24 hours)
        recent_sessions = [s for s in self.meta_learning_sessions 
                         if time.time() - s.end_time < 86400]
        
        recent_improvement_trend = 0.0
        if len(recent_sessions) >= 2:
            recent_scores = [s.improvement_score for s in recent_sessions]
            recent_improvement_trend = (recent_scores[-1] - recent_scores[0]) / len(recent_scores)
        
        return {
            'status': 'success',
            'total_sessions': total_sessions,
            'total_insights': total_insights,
            'total_optimizations': total_optimizations,
            'average_improvement_score': avg_improvement_score,
            'recent_sessions': len(recent_sessions),
            'recent_improvement_trend': recent_improvement_trend,
            'component_status': self.get_component_status()
        }
    
    def run_targeted_optimization(self, target_component: str, 
                                 optimization_type: str = "performance") -> Dict[str, Any]:
        """Run targeted optimization for a specific component."""
        if target_component == 'performance_monitor':
            return self._optimize_performance_monitoring()
        elif target_component == 'pipeline_reconfigurator':
            return self._optimize_pipeline_reconfiguration()
        elif target_component == 'self_reflection_agent':
            return self._optimize_self_reflection()
        else:
            return {'status': 'error', 'message': f'Unknown component: {target_component}'}
    
    def _optimize_performance_monitoring(self) -> Dict[str, Any]:
        """Optimize performance monitoring component."""
        # Adjust monitoring thresholds based on recent performance
        recent_summary = self.performance_monitor.get_performance_summary()
        
        if recent_summary['status'] == 'success':
            system_health = recent_summary.get('system_health', {})
            health_score = system_health.get('score', 0.0)
            
            # Adjust thresholds based on health score
            if health_score < 0.6:
                # Make thresholds more lenient
                self.performance_monitor.health_thresholds['response_time'] *= 1.2
                self.performance_monitor.health_thresholds['accuracy'] *= 0.9
            elif health_score > 0.9:
                # Make thresholds more strict
                self.performance_monitor.health_thresholds['response_time'] *= 0.9
                self.performance_monitor.health_thresholds['accuracy'] *= 1.1
        
        return {
            'status': 'completed',
            'component': 'performance_monitor',
            'optimizations_applied': 1,
            'new_thresholds': self.performance_monitor.health_thresholds
        }
    
    def _optimize_pipeline_reconfiguration(self) -> Dict[str, Any]:
        """Optimize pipeline reconfiguration component."""
        # Adjust performance threshold based on recent performance
        recent_configs = [c for c in self.pipeline_reconfigurator.pipeline_configurations 
                         if time.time() - c.last_updated < 3600]
        
        if recent_configs:
            avg_performance = sum(
                self.pipeline_reconfigurator._calculate_performance_score(c.performance_metrics)
                for c in recent_configs
            ) / len(recent_configs)
            
            # Adjust threshold based on average performance
            if avg_performance < 0.7:
                self.pipeline_reconfigurator.performance_threshold *= 0.9  # More lenient
            elif avg_performance > 0.9:
                self.pipeline_reconfigurator.performance_threshold *= 1.1  # More strict
        
        return {
            'status': 'completed',
            'component': 'pipeline_reconfigurator',
            'optimizations_applied': 1,
            'new_threshold': self.pipeline_reconfigurator.performance_threshold
        }
    
    def _optimize_self_reflection(self) -> Dict[str, Any]:
        """Optimize self-reflection component."""
        # Adjust reflection interval based on recent insights
        recent_insights = [i for i in self.self_reflection_agent.reflection_insights 
                          if time.time() - i.timestamp < 86400]
        
        actionable_insights = [i for i in recent_insights if i.actionable]
        actionable_ratio = len(actionable_insights) / len(recent_insights) if recent_insights else 0.0
        
        # Adjust reflection interval based on actionable insights ratio
        if actionable_ratio > 0.7:
            # More actionable insights, increase frequency
            self.self_reflection_agent.reflection_interval *= 0.8
        elif actionable_ratio < 0.3:
            # Few actionable insights, decrease frequency
            self.self_reflection_agent.reflection_interval *= 1.2
        
        return {
            'status': 'completed',
            'component': 'self_reflection_agent',
            'optimizations_applied': 1,
            'new_interval': self.self_reflection_agent.reflection_interval
        } 