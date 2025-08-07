"""
Reasoning Module for Quark AI Assistant
==========================================

Implements advanced reasoning capabilities including logical inference,
causal reasoning, analogical reasoning, and multi-step problem solving.

Part of Pillar 18: Generalized Reasoning
"""

from .generalized_reasoning import (
    GeneralizedReasoning,
    ReasoningType,
    ReasoningStep,
    ReasoningChain,
    LogicalRule,
    ConfidenceLevel
)

__all__ = [
    'GeneralizedReasoning',
    'ReasoningType',
    'ReasoningStep',
    'ReasoningChain',
    'LogicalRule',
    'ConfidenceLevel'
] 