"""
Autonomy Module for Quark AI Assistant
=========================================

Implements autonomous goal generation, self-directed learning, intrinsic motivation,
and independent decision-making capabilities.

Part of Pillar 20: Autonomous Goals
"""

from .autonomous_goals import (
    AutonomousGoals,
    GoalType,
    GoalPriority,
    MotivationType,
    Goal,
    LearningObjective,
    MotivationState,
    DecisionContext
)

__all__ = [
    'AutonomousGoals',
    'GoalType',
    'GoalPriority',
    'MotivationType',
    'Goal',
    'LearningObjective',
    'MotivationState',
    'DecisionContext'
] 