"""
META-LEARNING MODULE
===================

This module implements meta-learning capabilities for the Quark AI Assistant.
It includes self-monitoring agents, performance introspection, and pipeline reconfiguration.

Part of Pillar 16: Meta-Learning & Self-Reflection
"""

from .meta_learning_agent import MetaLearningAgent
from .performance_monitor import PerformanceMonitor
from .pipeline_reconfigurator import PipelineReconfigurator
from .self_reflection_agent import SelfReflectionAgent
from .meta_learning_orchestrator import MetaLearningOrchestrator

__all__ = [
    'MetaLearningAgent',
    'PerformanceMonitor',
    'PipelineReconfigurator',
    'SelfReflectionAgent',
    'MetaLearningOrchestrator'
] 