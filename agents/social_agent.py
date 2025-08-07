#!/usr/bin/env python3
"""
Social Intelligence Agent for Quark AI Assistant
==================================================

Manages social intelligence operations including emotional intelligence,
social context understanding, theory of mind, and multi-agent collaboration.

Part of Pillar 19: Social Intelligence
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from agents.base import Agent
from social.social_intelligence import (
    SocialIntelligence, EmotionType, SocialContext, TheoryOfMindLevel,
    EmotionalState, SocialInteraction, MentalState, SocialRelationship
)


class SocialAgent(Agent):
    """Advanced social intelligence agent for emotional and social operations."""
    
    def __init__(self, model_name: str = "social_agent", social_dir: str = None):
        super().__init__(model_name)
        self.social_dir = social_dir or os.path.join(os.path.dirname(__file__), '..', 'social_data')
        os.makedirs(self.social_dir, exist_ok=True)
        
        # Initialize social intelligence system
        self.social_engine = SocialIntelligence(self.social_dir)
        
        # Social operation settings
        self.emotion_sensitivity = 0.7
        self.context_adaptation = True
        self.collaboration_preference = 0.8
        self.relationship_tracking = True
        
        # Social tracking
        self.social_operations = []
        self.social_performance = {
            'total_operations': 0,
            'emotion_recognition_success': 0,
            'context_adaptation_success': 0,
            'collaboration_success': 0,
            'relationship_management_success': 0
        }
    
    def load_model(self):
        """Load social intelligence models and components."""
        try:
            # Initialize social intelligence system
            self.social_engine = SocialIntelligence(self.social_dir)
            
            return True
        except Exception as e:
            print(f"Error loading social intelligence models: {e}")
            return False
    
    def generate(self, input_data: str, operation: str = "recognize_emotion", **kwargs) -> Dict[str, Any]:
        """
        Generate social intelligence operations or perform social management.
        
        Args:
            input_data: Text or interaction data
            operation: Social operation to perform
            **kwargs: Additional parameters
            
        Returns:
            Social operation result
        """
        try:
            if operation == "recognize_emotion":
                return self._recognize_emotion(input_data, **kwargs)
            elif operation == "adapt_to_context":
                return self._adapt_to_social_context(input_data, **kwargs)
            elif operation == "understand_mental_state":
                return self._understand_mental_state(input_data, **kwargs)
            elif operation == "facilitate_collaboration":
                return self._facilitate_collaboration(input_data, **kwargs)
            elif operation == "manage_relationship":
                return self._manage_relationship(input_data, **kwargs)
            elif operation == "get_social_stats":
                return self._get_social_stats()
            elif operation == "export_social_data":
                return self._export_social_data(**kwargs)
            elif operation == "analyze_social_dynamics":
                return self._analyze_social_dynamics(input_data, **kwargs)
            elif operation == "resolve_conflict":
                return self._resolve_conflict(input_data, **kwargs)
            else:
                return {"error": f"Unknown social operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Social operation failed: {str(e)}"}
    
    def _recognize_emotion(self, text: str, agent_id: str = None) -> Dict[str, Any]:
        """Recognize emotions in text or from an agent."""
        try:
            # Perform emotion recognition
            result = self.social_engine.recognize_emotion(text, agent_id)
            
            # Track operation
            self.social_operations.append({
                'operation': 'recognize_emotion',
                'text_length': len(text),
                'agent_id': agent_id,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.social_performance['emotion_recognition_success'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to recognize emotion: {str(e)}"}
    
    def _adapt_to_social_context(self, context_data: str, current_style: str = None) -> Dict[str, Any]:
        """Adapt behavior and communication style to social context."""
        try:
            # Parse context data
            if isinstance(context_data, str):
                # Try to determine context from text
                context = self._determine_social_context(context_data)
            else:
                context = context_data
            
            # Perform context adaptation
            result = self.social_engine.adapt_to_social_context(context, current_style)
            
            # Track operation
            self.social_operations.append({
                'operation': 'adapt_to_context',
                'context': context.value if hasattr(context, 'value') else str(context),
                'current_style': current_style,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.social_performance['context_adaptation_success'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to adapt to social context: {str(e)}"}
    
    def _determine_social_context(self, text: str) -> SocialContext:
        """Determine social context from text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['formal', 'professional', 'business', 'official']):
            return SocialContext.FORMAL
        elif any(word in text_lower for word in ['casual', 'friendly', 'relaxed', 'informal']):
            return SocialContext.INFORMAL
        elif any(word in text_lower for word in ['collaborate', 'teamwork', 'cooperation', 'joint']):
            return SocialContext.COLLABORATIVE
        elif any(word in text_lower for word in ['compete', 'rivalry', 'contest', 'competition']):
            return SocialContext.COMPETITIVE
        elif any(word in text_lower for word in ['learn', 'teach', 'education', 'instruction']):
            return SocialContext.EDUCATIONAL
        elif any(word in text_lower for word in ['support', 'counsel', 'heal', 'therapy']):
            return SocialContext.THERAPEUTIC
        else:
            return SocialContext.INFORMAL  # Default context
    
    def _understand_mental_state(self, agent_info: str, available_information: Dict[str, Any] = None) -> Dict[str, Any]:
        """Understand another agent's mental state using theory of mind."""
        try:
            # Parse agent information
            if isinstance(agent_info, str):
                agent_id = agent_info
            else:
                agent_id = agent_info.get('agent_id', 'unknown_agent')
            
            # Use provided information or default
            if not available_information:
                available_information = {
                    'beliefs': [],
                    'desires': [],
                    'intentions': [],
                    'knowledge': [],
                    'emotional_state': None
                }
            
            # Perform theory of mind analysis
            result = self.social_engine.understand_mental_state(agent_id, available_information)
            
            # Track operation
            self.social_operations.append({
                'operation': 'understand_mental_state',
                'agent_id': agent_id,
                'information_available': len(available_information),
                'result': result,
                'timestamp': time.time()
            })
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to understand mental state: {str(e)}"}
    
    def _facilitate_collaboration(self, collaboration_data: str, strategy: str = None) -> Dict[str, Any]:
        """Facilitate collaboration between multiple agents."""
        try:
            # Parse collaboration data
            if isinstance(collaboration_data, str):
                # Simple parsing for demonstration
                agents = ['agent_1', 'agent_2']  # Default agents
                goals = [collaboration_data]  # Use input as goal
            else:
                agents = collaboration_data.get('agents', [])
                goals = collaboration_data.get('goals', [])
            
            # Perform collaboration facilitation
            result = self.social_engine.facilitate_collaboration(agents, goals, strategy)
            
            # Track operation
            self.social_operations.append({
                'operation': 'facilitate_collaboration',
                'agents_count': len(agents),
                'goals_count': len(goals),
                'strategy': strategy,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.social_performance['collaboration_success'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to facilitate collaboration: {str(e)}"}
    
    def _manage_relationship(self, relationship_data: str, interaction_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Manage and update social relationships."""
        try:
            # Parse relationship data
            if isinstance(relationship_data, str):
                # Simple parsing for demonstration
                parts = relationship_data.split('_')
                if len(parts) >= 2:
                    agent_id = parts[0]
                    other_agent_id = parts[1]
                else:
                    agent_id = 'agent_1'
                    other_agent_id = 'agent_2'
            else:
                agent_id = relationship_data.get('agent_id', 'agent_1')
                other_agent_id = relationship_data.get('other_agent_id', 'agent_2')
            
            # Use provided interaction data or default
            if not interaction_data:
                interaction_data = {
                    'quality': 0.7,
                    'type': 'cooperative'
                }
            
            # Perform relationship management
            result = self.social_engine.manage_relationships(agent_id, other_agent_id, interaction_data)
            
            # Track operation
            self.social_operations.append({
                'operation': 'manage_relationship',
                'agent_id': agent_id,
                'other_agent_id': other_agent_id,
                'interaction_quality': interaction_data.get('quality', 0.5),
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.social_performance['relationship_management_success'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to manage relationship: {str(e)}"}
    
    def _get_social_stats(self) -> Dict[str, Any]:
        """Get comprehensive social intelligence statistics."""
        try:
            # Get engine stats
            engine_stats = self.social_engine.get_social_stats()
            
            # Get agent performance stats
            performance_stats = {
                'total_operations': self.social_performance['total_operations'],
                'emotion_recognition_success': self.social_performance['emotion_recognition_success'],
                'context_adaptation_success': self.social_performance['context_adaptation_success'],
                'collaboration_success': self.social_performance['collaboration_success'],
                'relationship_management_success': self.social_performance['relationship_management_success']
            }
            
            # Calculate success rates
            if performance_stats['total_operations'] > 0:
                performance_stats['emotion_recognition_rate'] = (
                    performance_stats['emotion_recognition_success'] / performance_stats['total_operations']
                )
                performance_stats['context_adaptation_rate'] = (
                    performance_stats['context_adaptation_success'] / performance_stats['total_operations']
                )
                performance_stats['collaboration_success_rate'] = (
                    performance_stats['collaboration_success'] / performance_stats['total_operations']
                )
                performance_stats['relationship_management_rate'] = (
                    performance_stats['relationship_management_success'] / performance_stats['total_operations']
                )
            else:
                performance_stats.update({
                    'emotion_recognition_rate': 0.0,
                    'context_adaptation_rate': 0.0,
                    'collaboration_success_rate': 0.0,
                    'relationship_management_rate': 0.0
                })
            
            # Get recent operations
            recent_operations = self.social_operations[-10:] if self.social_operations else []
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "engine_stats": engine_stats,
                "performance_stats": performance_stats,
                "recent_operations": recent_operations,
                "settings": {
                    "emotion_sensitivity": self.emotion_sensitivity,
                    "context_adaptation": self.context_adaptation,
                    "collaboration_preference": self.collaboration_preference,
                    "relationship_tracking": self.relationship_tracking
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to get social stats: {str(e)}"}
    
    def _export_social_data(self, filename: str = None) -> Dict[str, Any]:
        """Export social intelligence data to JSON."""
        try:
            export_file = self.social_engine.export_social_data(filename)
            
            return {
                "status": "success",
                "export_file": export_file,
                "export_timestamp": datetime.now().isoformat(),
                "message": f"Social data exported to: {export_file}"
            }
            
        except Exception as e:
            return {"error": f"Failed to export social data: {str(e)}"}
    
    def _analyze_social_dynamics(self, interaction_data: str) -> Dict[str, Any]:
        """Analyze social dynamics in a group or interaction."""
        try:
            # Simple social dynamics analysis
            analysis = {
                "interaction_patterns": [],
                "emotional_climate": "neutral",
                "collaboration_potential": 0.7,
                "conflict_indicators": [],
                "relationship_insights": []
            }
            
            # Analyze interaction data
            if isinstance(interaction_data, str):
                text_lower = interaction_data.lower()
                
                # Check for emotional indicators
                if any(word in text_lower for word in ['happy', 'excited', 'pleased']):
                    analysis['emotional_climate'] = 'positive'
                elif any(word in text_lower for word in ['angry', 'frustrated', 'sad']):
                    analysis['emotional_climate'] = 'negative'
                
                # Check for collaboration indicators
                if any(word in text_lower for word in ['collaborate', 'teamwork', 'cooperation']):
                    analysis['collaboration_potential'] = 0.9
                elif any(word in text_lower for word in ['compete', 'conflict', 'disagree']):
                    analysis['collaboration_potential'] = 0.3
            
            return {
                "status": "success",
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze social dynamics: {str(e)}"}
    
    def _resolve_conflict(self, conflict_data: str) -> Dict[str, Any]:
        """Resolve conflicts between agents or parties."""
        try:
            # Simple conflict resolution
            resolution_plan = {
                "phase_1": "Identify conflict sources",
                "phase_2": "Understand perspectives",
                "phase_3": "Generate solutions",
                "phase_4": "Implement resolution"
            }
            
            # Analyze conflict data
            if isinstance(conflict_data, str):
                text_lower = conflict_data.lower()
                
                if any(word in text_lower for word in ['disagree', 'conflict', 'argument']):
                    resolution_plan['urgency'] = 'high'
                else:
                    resolution_plan['urgency'] = 'medium'
            
            return {
                "status": "success",
                "resolution_plan": resolution_plan,
                "estimated_duration": "2-4 hours",
                "success_probability": 0.8,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to resolve conflict: {str(e)}"}
    
    def _update_performance_stats(self, result: Dict[str, Any]):
        """Update social performance statistics."""
        self.social_performance['total_operations'] += 1
    
    def get_social_recommendations(self, context: str = None) -> Dict[str, Any]:
        """Get social intelligence-related recommendations."""
        try:
            recommendations = []
            
            # Check emotion recognition performance
            emotion_rate = (self.social_performance['emotion_recognition_success'] / 
                          self.social_performance['total_operations'] 
                          if self.social_performance['total_operations'] > 0 else 0)
            
            if emotion_rate < 0.7:
                recommendations.append({
                    'type': 'emotion_recognition',
                    'priority': 'medium',
                    'message': f'Emotion recognition rate is {emotion_rate:.1%}, consider improving recognition patterns',
                    'action': 'enhance_emotion_patterns'
                })
            
            # Check collaboration success
            collaboration_rate = (self.social_performance['collaboration_success'] / 
                               self.social_performance['total_operations'] 
                               if self.social_performance['total_operations'] > 0 else 0)
            
            if collaboration_rate < 0.6:
                recommendations.append({
                    'type': 'collaboration',
                    'priority': 'high',
                    'message': f'Collaboration success rate is {collaboration_rate:.1%}, consider improving facilitation strategies',
                    'action': 'improve_collaboration_strategies'
                })
            
            # Check relationship management
            relationship_rate = (self.social_performance['relationship_management_success'] / 
                              self.social_performance['total_operations'] 
                              if self.social_performance['total_operations'] > 0 else 0)
            
            if relationship_rate < 0.5:
                recommendations.append({
                    'type': 'relationship_management',
                    'priority': 'medium',
                    'message': f'Relationship management rate is {relationship_rate:.1%}, consider improving relationship tracking',
                    'action': 'enhance_relationship_tracking'
                })
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "recommendations_count": len(recommendations)
            }
            
        except Exception as e:
            return {"error": f"Failed to get recommendations: {str(e)}"} 