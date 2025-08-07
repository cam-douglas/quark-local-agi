#!/usr/bin/env python3
"""
Tests for Meta-Learning System (Pillar 16)
Tests the meta-learning components and orchestration
"""

import pytest
import time
import tempfile
import os
from unittest.mock import Mock, patch

from meta_learning.meta_learning_orchestrator import MetaLearningOrchestrator
from meta_learning.performance_monitor import PerformanceMonitor
from meta_learning.pipeline_reconfigurator import PipelineReconfigurator
from meta_learning.self_reflection_agent import SelfReflectionAgent

class TestMetaLearningOrchestrator:
    """Test the meta-learning orchestrator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.orchestrator = MetaLearningOrchestrator(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_orchestrator_initialization(self):
        """Test that the orchestrator initializes correctly."""
        assert self.orchestrator.orchestration_enabled == True
        assert self.orchestrator.auto_orchestration == True
        assert self.orchestrator.orchestration_interval == 1800
        assert self.orchestrator.performance_monitor is not None
        assert self.orchestrator.pipeline_reconfigurator is not None
        assert self.orchestrator.self_reflection_agent is not None
    
    def test_meta_learning_session(self):
        """Test running a meta-learning session."""
        # Mock the component methods to avoid actual execution
        with patch.object(self.orchestrator.performance_monitor, 'get_performance_summary') as mock_perf:
            with patch.object(self.orchestrator.self_reflection_agent, 'run_reflection_session') as mock_reflection:
                with patch.object(self.orchestrator.pipeline_reconfigurator, 'analyze_pipeline_performance') as mock_reconfig:
                    
                    mock_perf.return_value = {'status': 'success', 'insights_count': 2}
                    mock_reflection.return_value = {
                        'status': 'completed',
                        'session_id': 'test_session',
                        'insights_count': 3,
                        'improvement_score': 0.75
                    }
                    mock_reconfig.return_value = {
                        'status': 'success',
                        'config_id': 'test_config',
                        'performance_score': 0.8,
                        'reconfiguration_needed': False,
                        'recommendations': []
                    }
                    
                    result = self.orchestrator.run_meta_learning_session("comprehensive")
                    
                    assert result['status'] == 'completed'
                    assert 'session_id' in result
                    assert 'components_used' in result
                    assert 'insights_generated' in result
                    assert 'optimizations_applied' in result
                    assert 'improvement_score' in result
    
    def test_component_status(self):
        """Test getting component status."""
        status = self.orchestrator.get_component_status()
        
        assert 'performance_monitor' in status
        assert 'pipeline_reconfigurator' in status
        assert 'self_reflection_agent' in status
        assert 'orchestrator' in status
        
        # Check that each component has expected keys
        for component, data in status.items():
            if component == 'orchestrator':
                assert 'enabled' in data
                assert 'auto_orchestration' in data
                assert 'total_sessions' in data
            else:
                assert 'enabled' in data
                assert 'statistics' in data
    
    def test_targeted_optimization(self):
        """Test targeted optimization for components."""
        # Test performance monitor optimization
        result = self.orchestrator.run_targeted_optimization('performance_monitor')
        assert result['status'] == 'completed'
        assert 'optimizations_applied' in result
        
        # Test pipeline reconfigurator optimization
        result = self.orchestrator.run_targeted_optimization('pipeline_reconfigurator')
        assert result['status'] == 'completed'
        assert 'optimizations_applied' in result
        
        # Test self-reflection agent optimization
        result = self.orchestrator.run_targeted_optimization('self_reflection_agent')
        assert result['status'] == 'completed'
        assert 'optimizations_applied' in result

class TestPerformanceMonitor:
    """Test the performance monitor."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.monitor = PerformanceMonitor()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_monitor_initialization(self):
        """Test that the monitor initializes correctly."""
        assert self.monitor.monitoring_enabled == True
        assert self.monitor.alert_enabled == True
        assert self.monitor.auto_recovery == True
        assert 'response_time' in self.monitor.health_thresholds
        assert 'accuracy' in self.monitor.health_thresholds
    
    def test_record_agent_performance(self):
        """Test recording agent performance metrics."""
        metrics = {
            'response_time': 2.5,
            'accuracy': 0.85,
            'throughput': 15.0,
            'error_rate': 0.02
        }
        
        metric_id = self.monitor.record_agent_performance('NLU', metrics)
        assert metric_id is not None
        assert len(self.monitor.performance_history) > 0
    
    def test_get_performance_summary(self):
        """Test getting performance summary."""
        # Add some test data
        metrics = {'response_time': 2.5, 'accuracy': 0.85}
        self.monitor.record_agent_performance('NLU', metrics)
        
        summary = self.monitor.get_performance_summary()
        assert 'status' in summary
        assert 'total_snapshots' in summary
    
    def test_monitor_statistics(self):
        """Test getting monitor statistics."""
        stats = self.monitor.get_monitor_statistics()
        
        assert 'total_snapshots' in stats
        assert 'total_alerts' in stats
        assert 'monitoring_enabled' in stats
        assert 'alert_enabled' in stats
        assert 'auto_recovery' in stats
        assert 'health_thresholds' in stats

class TestPipelineReconfigurator:
    """Test the pipeline reconfigurator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.reconfigurator = PipelineReconfigurator(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_reconfigurator_initialization(self):
        """Test that the reconfigurator initializes correctly."""
        assert self.reconfigurator.auto_reconfiguration == True
        assert self.reconfigurator.testing_mode == False
        assert self.reconfigurator.rollback_enabled == True
        assert self.reconfigurator.performance_threshold == 0.8
        assert 'NLU' in self.reconfigurator.agent_capabilities
        assert 'Retrieval' in self.reconfigurator.agent_capabilities
    
    def test_create_pipeline_configuration(self):
        """Test creating a pipeline configuration."""
        agent_sequence = ['NLU', 'Retrieval', 'Reasoning']
        config_id = self.reconfigurator.create_pipeline_configuration(agent_sequence)
        
        assert config_id is not None
        assert len(self.reconfigurator.pipeline_configurations) > 0
        
        config = self.reconfigurator.pipeline_configurations[0]
        assert config.agent_sequence == agent_sequence
        assert config.status == 'testing'
    
    def test_analyze_pipeline_performance(self):
        """Test analyzing pipeline performance."""
        # Create a test configuration
        agent_sequence = ['NLU', 'Retrieval']
        config_id = self.reconfigurator.create_pipeline_configuration(agent_sequence)
        
        # Analyze performance
        performance_data = {
            'accuracy': 0.82,
            'response_time': 3.2,
            'throughput': 12.0
        }
        
        result = self.reconfigurator.analyze_pipeline_performance(config_id, performance_data)
        
        assert result['status'] == 'success'
        assert 'performance_score' in result
        assert 'reconfiguration_needed' in result
        assert 'recommendations' in result
    
    def test_reconfiguration_statistics(self):
        """Test getting reconfiguration statistics."""
        stats = self.reconfigurator.get_reconfiguration_statistics()
        
        assert 'total_configurations' in stats
        assert 'auto_reconfiguration' in stats
        assert 'testing_mode' in stats
        assert 'total_reconfigurations' in stats
        assert 'performance_threshold' in stats

class TestSelfReflectionAgent:
    """Test the self-reflection agent."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.reflection_agent = SelfReflectionAgent(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_reflection_agent_initialization(self):
        """Test that the reflection agent initializes correctly."""
        assert self.reflection_agent.reflection_enabled == True
        assert self.reflection_agent.auto_reflection == True
        assert self.reflection_agent.reflection_interval == 3600
        assert 'performance_analysis' in self.reflection_agent.reflection_capabilities
        assert 'behavior_patterns' in self.reflection_agent.reflection_capabilities
    
    def test_run_reflection_session(self):
        """Test running a reflection session."""
        # Mock the reflection methods to avoid actual execution
        with patch.object(self.reflection_agent, '_reflect_on_performance') as mock_perf:
            with patch.object(self.reflection_agent, '_reflect_on_behavior') as mock_behavior:
                with patch.object(self.reflection_agent, '_reflect_on_capabilities') as mock_capabilities:
                    with patch.object(self.reflection_agent, '_reflect_on_strategy') as mock_strategy:
                        
                        mock_perf.return_value = []
                        mock_behavior.return_value = []
                        mock_capabilities.return_value = []
                        mock_strategy.return_value = []
                        
                        result = self.reflection_agent.run_reflection_session("general")
                        
                        assert result['status'] == 'completed'
                        assert 'session_id' in result
                        assert 'reflection_type' in result
                        assert 'insights_count' in result
                        assert 'actionable_insights' in result
                        assert 'improvement_score' in result
    
    def test_reflection_statistics(self):
        """Test getting reflection statistics."""
        stats = self.reflection_agent.get_reflection_statistics()
        
        assert 'total_sessions' in stats
        assert 'total_insights' in stats
        assert 'actionable_insights' in stats
        assert 'reflection_enabled' in stats
        assert 'auto_reflection' in stats
        assert 'reflection_capabilities' in stats

if __name__ == '__main__':
    pytest.main([__file__]) 