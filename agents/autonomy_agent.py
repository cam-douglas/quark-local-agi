#!/usr/bin/env python3
"""
Autonomy Agent for Quark AI Assistant
=========================================

Manages autonomous goal generation, self-directed learning, intrinsic motivation,
and independent decision-making capabilities.

Part of Pillar 20: Autonomous Goals
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from agents.base import Agent
from autonomy.autonomous_goals import (
    AutonomousGoals, GoalType, GoalPriority, MotivationType,
    Goal, LearningObjective, MotivationState, DecisionContext
)


class AutonomyAgent(Agent):
    """Advanced autonomy agent for autonomous goal management and self-directed learning."""
    
    def __init__(self, model_name: str = "autonomy_agent", autonomy_dir: str = None):
        super().__init__(model_name)
        self.autonomy_dir = autonomy_dir or os.path.join(os.path.dirname(__file__), '..', 'autonomy_data')
        os.makedirs(self.autonomy_dir, exist_ok=True)
        
        # Initialize autonomous goals system
        self.autonomy_engine = AutonomousGoals(self.autonomy_dir)
        
        # Autonomy operation settings
        self.goal_generation_enabled = True
        self.self_learning_enabled = True
        self.autonomous_decision_making = True
        self.motivation_threshold = 0.6
        
        # Autonomy tracking
        self.autonomy_operations = []
        self.autonomy_performance = {
            'total_operations': 0,
            'goals_generated': 0,
            'learning_sessions': 0,
            'autonomous_decisions': 0,
            'motivation_updates': 0
        }
    
    def load_model(self):
        """Load autonomy models and components."""
        try:
            # Initialize autonomous goals system
            self.autonomy_engine = AutonomousGoals(self.autonomy_dir)
            
            return True
        except Exception as e:
            print(f"Error loading autonomy models: {e}")
            return False
    
    def generate(self, input_data: str, operation: str = "generate_goal", **kwargs) -> Dict[str, Any]:
        """
        Generate autonomy operations or perform autonomy management.
        
        Args:
            input_data: Context or goal description
            operation: Autonomy operation to perform
            **kwargs: Additional parameters
            
        Returns:
            Autonomy operation result
        """
        try:
            if operation == "generate_goal":
                return self._generate_autonomous_goal(input_data, **kwargs)
            elif operation == "create_learning_objective":
                return self._create_learning_objective(input_data, **kwargs)
            elif operation == "update_motivation":
                return self._update_motivation_state(input_data, **kwargs)
            elif operation == "make_decision":
                return self._make_autonomous_decision(input_data, **kwargs)
            elif operation == "update_goal_progress":
                return self._update_goal_progress(input_data, **kwargs)
            elif operation == "get_autonomy_stats":
                return self._get_autonomy_stats()
            elif operation == "export_autonomy_data":
                return self._export_autonomy_data(**kwargs)
            elif operation == "analyze_goals":
                return self._analyze_goals(input_data, **kwargs)
            elif operation == "prioritize_goals":
                return self._prioritize_goals(input_data, **kwargs)
            else:
                return {"error": f"Unknown autonomy operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Autonomy operation failed: {str(e)}"}
    
    def _generate_autonomous_goal(self, context: str, goal_type: str = None) -> Dict[str, Any]:
        """Generate an autonomous goal based on context."""
        try:
            # Parse goal type if provided
            goal_type_enum = None
            if goal_type:
                try:
                    goal_type_enum = GoalType(goal_type)
                except ValueError:
                    return {"error": f"Invalid goal type: {goal_type}"}
            
            # Generate goal
            result = self.autonomy_engine.generate_autonomous_goal(context, goal_type_enum)
            
            # Track operation
            self.autonomy_operations.append({
                'operation': 'generate_goal',
                'context': context,
                'goal_type': goal_type,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.autonomy_performance['goals_generated'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to generate autonomous goal: {str(e)}"}
    
    def _create_learning_objective(self, learning_data: str, goal_id: str = None, 
                                  skill_name: str = None, target_level: float = 1.0) -> Dict[str, Any]:
        """Create a learning objective for skill development."""
        try:
            # Parse learning data
            if isinstance(learning_data, str):
                # Extract skill name from learning data
                if not skill_name:
                    skill_name = self._extract_skill_name(learning_data)
                if not goal_id:
                    goal_id = f"goal_{int(time.time())}"
            else:
                skill_name = learning_data.get('skill_name', 'general_skill')
                goal_id = learning_data.get('goal_id', f"goal_{int(time.time())}")
                target_level = learning_data.get('target_level', 1.0)
            
            # Create learning objective
            result = self.autonomy_engine.create_learning_objective(goal_id, skill_name, target_level)
            
            # Track operation
            self.autonomy_operations.append({
                'operation': 'create_learning_objective',
                'skill_name': skill_name,
                'goal_id': goal_id,
                'target_level': target_level,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.autonomy_performance['learning_sessions'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to create learning objective: {str(e)}"}
    
    def _extract_skill_name(self, learning_data: str) -> str:
        """Extract skill name from learning data."""
        # Simple skill extraction
        learning_lower = learning_data.lower()
        
        if any(word in learning_lower for word in ['programming', 'coding', 'code']):
            return 'programming'
        elif any(word in learning_lower for word in ['communication', 'speaking', 'writing']):
            return 'communication'
        elif any(word in learning_lower for word in ['analysis', 'analytics', 'data']):
            return 'data_analysis'
        elif any(word in learning_lower for word in ['safety', 'security', 'alignment']):
            return 'safety_engineering'
        elif any(word in learning_lower for word in ['collaboration', 'teamwork', 'cooperation']):
            return 'collaboration'
        else:
            return 'general_skill'
    
    def _update_motivation_state(self, motivation_data: str, agent_id: str = None,
                                motivation_type: str = None, intensity: float = 0.7) -> Dict[str, Any]:
        """Update motivation state for an agent."""
        try:
            # Parse motivation data
            if isinstance(motivation_data, str):
                # Extract motivation type from text
                if not motivation_type:
                    motivation_type = self._extract_motivation_type(motivation_data)
                if not agent_id:
                    agent_id = 'autonomy_agent'
            else:
                motivation_type = motivation_data.get('motivation_type', 'curiosity')
                agent_id = motivation_data.get('agent_id', 'autonomy_agent')
                intensity = motivation_data.get('intensity', 0.7)
            
            # Convert motivation type to enum
            try:
                motivation_type_enum = MotivationType(motivation_type)
            except ValueError:
                return {"error": f"Invalid motivation type: {motivation_type}"}
            
            # Update motivation state
            result = self.autonomy_engine.update_motivation_state(agent_id, motivation_type_enum, intensity)
            
            # Track operation
            self.autonomy_operations.append({
                'operation': 'update_motivation',
                'agent_id': agent_id,
                'motivation_type': motivation_type,
                'intensity': intensity,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.autonomy_performance['motivation_updates'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to update motivation state: {str(e)}"}
    
    def _extract_motivation_type(self, motivation_data: str) -> str:
        """Extract motivation type from text."""
        motivation_lower = motivation_data.lower()
        
        if any(word in motivation_lower for word in ['learn', 'explore', 'discover', 'curious']):
            return 'curiosity'
        elif any(word in motivation_lower for word in ['improve', 'master', 'excel', 'perfect']):
            return 'mastery'
        elif any(word in motivation_lower for word in ['independent', 'autonomous', 'self-directed']):
            return 'autonomy'
        elif any(word in motivation_lower for word in ['contribute', 'help', 'purpose', 'meaningful']):
            return 'purpose'
        elif any(word in motivation_lower for word in ['collaborate', 'connect', 'social', 'team']):
            return 'social'
        else:
            return 'curiosity'  # Default motivation
    
    def _make_autonomous_decision(self, decision_data: str, options: List[str] = None,
                                 criteria: List[str] = None) -> Dict[str, Any]:
        """Make an autonomous decision based on context and options."""
        try:
            # Parse decision data
            if isinstance(decision_data, str):
                context = decision_data
                if not options:
                    options = self._generate_decision_options(context)
            else:
                context = decision_data.get('context', 'general decision')
                options = decision_data.get('options', [])
                criteria = decision_data.get('criteria', [])
            
            # Make decision
            result = self.autonomy_engine.make_autonomous_decision(context, options, criteria)
            
            # Track operation
            self.autonomy_operations.append({
                'operation': 'make_decision',
                'context': context,
                'options_count': len(options),
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.autonomy_performance['autonomous_decisions'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to make autonomous decision: {str(e)}"}
    
    def _generate_decision_options(self, context: str) -> List[str]:
        """Generate decision options based on context."""
        context_lower = context.lower()
        
        if any(word in context_lower for word in ['safety', 'security']):
            return ['Implement safety measures', 'Conduct risk assessment', 'Review protocols']
        elif any(word in context_lower for word in ['performance', 'efficiency']):
            return ['Optimize processes', 'Enhance capabilities', 'Improve algorithms']
        elif any(word in context_lower for word in ['learning', 'improvement']):
            return ['Study new techniques', 'Practice skills', 'Experiment with approaches']
        elif any(word in context_lower for word in ['collaboration', 'teamwork']):
            return ['Coordinate with team', 'Share knowledge', 'Build partnerships']
        else:
            return ['Proceed with caution', 'Take calculated risk', 'Maintain current approach']
    
    def _update_goal_progress(self, progress_data: str, goal_id: str = None,
                             progress: float = None) -> Dict[str, Any]:
        """Update the progress of a goal."""
        try:
            # Parse progress data
            if isinstance(progress_data, str):
                # Extract progress from text
                if not progress:
                    progress = self._extract_progress(progress_data)
                if not goal_id:
                    goal_id = f"goal_{int(time.time())}"
            else:
                goal_id = progress_data.get('goal_id', f"goal_{int(time.time())}")
                progress = progress_data.get('progress', 0.5)
            
            # Update goal progress
            result = self.autonomy_engine.update_goal_progress(goal_id, progress)
            
            # Track operation
            self.autonomy_operations.append({
                'operation': 'update_goal_progress',
                'goal_id': goal_id,
                'progress': progress,
                'result': result,
                'timestamp': time.time()
            })
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to update goal progress: {str(e)}"}
    
    def _extract_progress(self, progress_data: str) -> float:
        """Extract progress value from text."""
        progress_lower = progress_data.lower()
        
        if any(word in progress_lower for word in ['complete', 'finished', 'done', '100%']):
            return 1.0
        elif any(word in progress_lower for word in ['almost', 'nearly', '90%', '95%']):
            return 0.9
        elif any(word in progress_lower for word in ['half', '50%', 'midway']):
            return 0.5
        elif any(word in progress_lower for word in ['start', 'begin', '10%', '20%']):
            return 0.2
        else:
            return 0.5  # Default progress
    
    def _get_autonomy_stats(self) -> Dict[str, Any]:
        """Get comprehensive autonomy statistics."""
        try:
            # Get engine stats
            engine_stats = self.autonomy_engine.get_autonomy_stats()
            
            # Get agent performance stats
            performance_stats = {
                'total_operations': self.autonomy_performance['total_operations'],
                'goals_generated': self.autonomy_performance['goals_generated'],
                'learning_sessions': self.autonomy_performance['learning_sessions'],
                'autonomous_decisions': self.autonomy_performance['autonomous_decisions'],
                'motivation_updates': self.autonomy_performance['motivation_updates']
            }
            
            # Calculate success rates
            if performance_stats['total_operations'] > 0:
                performance_stats['goal_generation_rate'] = (
                    performance_stats['goals_generated'] / performance_stats['total_operations']
                )
                performance_stats['learning_success_rate'] = (
                    performance_stats['learning_sessions'] / performance_stats['total_operations']
                )
                performance_stats['decision_success_rate'] = (
                    performance_stats['autonomous_decisions'] / performance_stats['total_operations']
                )
            else:
                performance_stats.update({
                    'goal_generation_rate': 0.0,
                    'learning_success_rate': 0.0,
                    'decision_success_rate': 0.0
                })
            
            # Get recent operations
            recent_operations = self.autonomy_operations[-10:] if self.autonomy_operations else []
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "engine_stats": engine_stats,
                "performance_stats": performance_stats,
                "recent_operations": recent_operations,
                "settings": {
                    "goal_generation_enabled": self.goal_generation_enabled,
                    "self_learning_enabled": self.self_learning_enabled,
                    "autonomous_decision_making": self.autonomous_decision_making,
                    "motivation_threshold": self.motivation_threshold
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to get autonomy stats: {str(e)}"}
    
    def _export_autonomy_data(self, filename: str = None) -> Dict[str, Any]:
        """Export autonomy data to JSON."""
        try:
            export_file = self.autonomy_engine.export_autonomy_data(filename)
            
            return {
                "status": "success",
                "export_file": export_file,
                "export_timestamp": datetime.now().isoformat(),
                "message": f"Autonomy data exported to: {export_file}"
            }
            
        except Exception as e:
            return {"error": f"Failed to export autonomy data: {str(e)}"}
    
    def _analyze_goals(self, analysis_data: str) -> Dict[str, Any]:
        """Analyze current goals and their status."""
        try:
            # Analyze active goals
            active_goals = self.autonomy_engine.active_goals
            goal_analysis = {
                'total_active_goals': len(active_goals),
                'goal_types': {},
                'priority_distribution': {},
                'average_progress': 0.0,
                'completion_estimates': []
            }
            
            if active_goals:
                # Analyze goal types
                for goal in active_goals.values():
                    goal_type = goal.goal_type.value
                    goal_analysis['goal_types'][goal_type] = goal_analysis['goal_types'].get(goal_type, 0) + 1
                    
                    # Analyze priority distribution
                    priority = goal.priority.value
                    goal_analysis['priority_distribution'][priority] = goal_analysis['priority_distribution'].get(priority, 0) + 1
                
                # Calculate average progress
                total_progress = sum(goal.progress for goal in active_goals.values())
                goal_analysis['average_progress'] = total_progress / len(active_goals)
                
                # Estimate completion times
                for goal in active_goals.values():
                    remaining_progress = 1.0 - goal.progress
                    estimated_hours = remaining_progress * goal.estimated_duration
                    goal_analysis['completion_estimates'].append({
                        'goal_id': goal.goal_id,
                        'title': goal.title,
                        'estimated_hours': estimated_hours
                    })
            
            return {
                "status": "success",
                "analysis": goal_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze goals: {str(e)}"}
    
    def _prioritize_goals(self, priority_data: str) -> Dict[str, Any]:
        """Prioritize goals based on various criteria."""
        try:
            # Get active goals
            active_goals = list(self.autonomy_engine.active_goals.values())
            
            if not active_goals:
                return {
                    "status": "success",
                    "prioritized_goals": [],
                    "message": "No active goals to prioritize"
                }
            
            # Sort goals by priority and progress
            prioritized_goals = sorted(active_goals, key=lambda g: (
                g.priority.value,  # Priority first
                -g.progress,       # Higher progress first
                g.created_at       # Older goals first
            ))
            
            # Create priority list
            priority_list = []
            for i, goal in enumerate(prioritized_goals):
                priority_list.append({
                    'rank': i + 1,
                    'goal_id': goal.goal_id,
                    'title': goal.title,
                    'priority': goal.priority.value,
                    'progress': goal.progress,
                    'estimated_remaining': (1.0 - goal.progress) * goal.estimated_duration
                })
            
            return {
                "status": "success",
                "prioritized_goals": priority_list,
                "total_goals": len(priority_list),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to prioritize goals: {str(e)}"}
    
    def _update_performance_stats(self, result: Dict[str, Any]):
        """Update autonomy performance statistics."""
        self.autonomy_performance['total_operations'] += 1
    
    def get_autonomy_recommendations(self, context: str = None) -> Dict[str, Any]:
        """Get autonomy-related recommendations."""
        try:
            recommendations = []
            
            # Check goal generation performance
            goal_rate = (self.autonomy_performance['goals_generated'] / 
                        self.autonomy_performance['total_operations'] 
                        if self.autonomy_performance['total_operations'] > 0 else 0)
            
            if goal_rate < 0.3:
                recommendations.append({
                    'type': 'goal_generation',
                    'priority': 'medium',
                    'message': f'Goal generation rate is {goal_rate:.1%}, consider improving goal generation strategies',
                    'action': 'enhance_goal_generation'
                })
            
            # Check learning performance
            learning_rate = (self.autonomy_performance['learning_sessions'] / 
                           self.autonomy_performance['total_operations'] 
                           if self.autonomy_performance['total_operations'] > 0 else 0)
            
            if learning_rate < 0.2:
                recommendations.append({
                    'type': 'learning',
                    'priority': 'high',
                    'message': f'Learning session rate is {learning_rate:.1%}, consider increasing self-directed learning',
                    'action': 'increase_learning_activities'
                })
            
            # Check decision making performance
            decision_rate = (self.autonomy_performance['autonomous_decisions'] / 
                           self.autonomy_performance['total_operations'] 
                           if self.autonomy_performance['total_operations'] > 0 else 0)
            
            if decision_rate < 0.4:
                recommendations.append({
                    'type': 'decision_making',
                    'priority': 'high',
                    'message': f'Autonomous decision rate is {decision_rate:.1%}, consider improving decision-making capabilities',
                    'action': 'enhance_decision_making'
                })
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "recommendations_count": len(recommendations)
            }
            
        except Exception as e:
            return {"error": f"Failed to get recommendations: {str(e)}"} 