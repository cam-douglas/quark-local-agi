#!/usr/bin/env python3
"""
Social Intelligence Engine for Quark AI Assistant
===================================================

Implements advanced social intelligence capabilities including emotional intelligence,
social context understanding, theory of mind, and multi-agent collaboration.

Part of Pillar 19: Social Intelligence
"""

import os
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
from collections import defaultdict

from agents.base import Agent


class EmotionType(Enum):
    """Types of emotions that can be recognized and expressed."""
    JOY = "joy"                    # Happiness, excitement, pleasure
    SADNESS = "sadness"            # Grief, disappointment, melancholy
    ANGER = "anger"                # Frustration, irritation, rage
    FEAR = "fear"                  # Anxiety, worry, terror
    SURPRISE = "surprise"          # Astonishment, amazement, shock
    DISGUST = "disgust"            # Revulsion, aversion, contempt
    TRUST = "trust"                # Confidence, faith, reliance
    ANTICIPATION = "anticipation"  # Expectation, hope, eagerness
    NEUTRAL = "neutral"            # Calm, indifferent, balanced


class SocialContext(Enum):
    """Types of social contexts and situations."""
    FORMAL = "formal"              # Professional, business, official
    INFORMAL = "informal"          # Casual, friendly, relaxed
    INTIMATE = "intimate"          # Close, personal, private
    PUBLIC = "public"              # Open, shared, communal
    COLLABORATIVE = "collaborative" # Teamwork, cooperation, joint effort
    COMPETITIVE = "competitive"    # Rivalry, competition, contest
    EDUCATIONAL = "educational"    # Learning, teaching, instruction
    THERAPEUTIC = "therapeutic"    # Support, counseling, healing


class TheoryOfMindLevel(Enum):
    """Levels of theory of mind understanding."""
    ZERO_ORDER = "zero_order"      # No mental state attribution
    FIRST_ORDER = "first_order"    # Understanding others' mental states
    SECOND_ORDER = "second_order"  # Understanding others' understanding of mental states
    HIGHER_ORDER = "higher_order"  # Complex recursive mental state reasoning


@dataclass
class EmotionalState:
    """Represents an emotional state with intensity and context."""
    emotion: EmotionType
    intensity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    context: str
    triggers: List[str]
    timestamp: float
    duration: Optional[float] = None  # How long the emotion has been present


@dataclass
class SocialInteraction:
    """Represents a social interaction between agents or entities."""
    interaction_id: str
    participants: List[str]
    context: SocialContext
    emotional_states: Dict[str, EmotionalState]  # Participant -> EmotionalState
    communication_style: str
    goals: List[str]
    outcomes: List[str]
    timestamp: float
    duration: float


@dataclass
class MentalState:
    """Represents a mental state for theory of mind."""
    agent_id: str
    beliefs: List[str]
    desires: List[str]
    intentions: List[str]
    knowledge: List[str]
    emotional_state: Optional[EmotionalState]
    confidence: float
    timestamp: float


@dataclass
class SocialRelationship:
    """Represents a social relationship between entities."""
    relationship_id: str
    participants: List[str]
    relationship_type: str  # "friend", "colleague", "mentor", etc.
    strength: float  # 0.0 to 1.0
    trust_level: float  # 0.0 to 1.0
    history: List[SocialInteraction]
    shared_goals: List[str]
    conflicts: List[str]
    created_at: float
    last_updated: float


class SocialIntelligence:
    """Advanced social intelligence engine with emotional and social capabilities."""
    
    def __init__(self, social_dir: str = None):
        self.social_dir = social_dir or os.path.join(os.path.dirname(__file__), '..', 'social_data')
        os.makedirs(self.social_dir, exist_ok=True)
        
        # Social intelligence components
        self.emotional_states = {}  # Agent ID -> EmotionalState
        self.social_interactions = []
        self.mental_states = {}  # Agent ID -> MentalState
        self.relationships = {}
        self.social_networks = nx.Graph()
        
        # Social intelligence settings
        self.emotion_recognition_threshold = 0.6
        self.social_context_sensitivity = 0.7
        self.theory_of_mind_depth = TheoryOfMindLevel.FIRST_ORDER
        self.collaboration_preference = 0.8
        
        # Social tracking
        self.social_stats = {
            'total_interactions': 0,
            'emotional_recognition_accuracy': 0.0,
            'social_context_adaptations': 0,
            'collaboration_success_rate': 0.0,
            'relationship_management_score': 0.0
        }
        
        # Load social intelligence components
        self._load_emotion_patterns()
        self._load_social_context_rules()
        self._load_collaboration_strategies()
    
    def _load_emotion_patterns(self):
        """Load patterns for emotion recognition."""
        self.emotion_patterns = {
            EmotionType.JOY: {
                'keywords': ['happy', 'excited', 'pleased', 'delighted', 'thrilled'],
                'expressions': ['😊', '😄', '🎉', '👍'],
                'intensity_indicators': ['very', 'extremely', 'so', 'really']
            },
            EmotionType.SADNESS: {
                'keywords': ['sad', 'depressed', 'disappointed', 'melancholy', 'grief'],
                'expressions': ['😢', '😭', '😔', '💔'],
                'intensity_indicators': ['very', 'extremely', 'so', 'really']
            },
            EmotionType.ANGER: {
                'keywords': ['angry', 'furious', 'irritated', 'frustrated', 'mad'],
                'expressions': ['😠', '😡', '💢', '🤬'],
                'intensity_indicators': ['very', 'extremely', 'so', 'really']
            },
            EmotionType.FEAR: {
                'keywords': ['afraid', 'scared', 'anxious', 'worried', 'terrified'],
                'expressions': ['😨', '😰', '😱', '😓'],
                'intensity_indicators': ['very', 'extremely', 'so', 'really']
            },
            EmotionType.SURPRISE: {
                'keywords': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
                'expressions': ['😲', '😳', '😱', '🤯'],
                'intensity_indicators': ['very', 'extremely', 'so', 'really']
            },
            EmotionType.DISGUST: {
                'keywords': ['disgusted', 'revolted', 'appalled', 'sickened'],
                'expressions': ['🤢', '🤮', '😷', '🤧'],
                'intensity_indicators': ['very', 'extremely', 'so', 'really']
            },
            EmotionType.TRUST: {
                'keywords': ['trust', 'confident', 'reliable', 'faithful', 'dependable'],
                'expressions': ['🤝', '👍', '💪', '🙏'],
                'intensity_indicators': ['very', 'extremely', 'so', 'really']
            },
            EmotionType.ANTICIPATION: {
                'keywords': ['excited', 'eager', 'hopeful', 'expectant', 'looking forward'],
                'expressions': ['🤩', '😊', '✨', '🌟'],
                'intensity_indicators': ['very', 'extremely', 'so', 'really']
            }
        }
    
    def _load_social_context_rules(self):
        """Load rules for social context adaptation."""
        self.social_context_rules = {
            SocialContext.FORMAL: {
                'communication_style': 'professional',
                'emotional_expression': 'restrained',
                'formality_level': 'high',
                'appropriate_topics': ['work', 'business', 'professional matters'],
                'inappropriate_topics': ['personal', 'casual', 'informal']
            },
            SocialContext.INFORMAL: {
                'communication_style': 'casual',
                'emotional_expression': 'moderate',
                'formality_level': 'low',
                'appropriate_topics': ['personal', 'casual', 'general'],
                'inappropriate_topics': ['highly personal', 'controversial']
            },
            SocialContext.COLLABORATIVE: {
                'communication_style': 'cooperative',
                'emotional_expression': 'supportive',
                'formality_level': 'medium',
                'appropriate_topics': ['shared goals', 'teamwork', 'cooperation'],
                'inappropriate_topics': ['conflict', 'competition', 'individual goals']
            },
            SocialContext.COMPETITIVE: {
                'communication_style': 'assertive',
                'emotional_expression': 'controlled',
                'formality_level': 'medium',
                'appropriate_topics': ['performance', 'achievement', 'goals'],
                'inappropriate_topics': ['weakness', 'failure', 'personal issues']
            }
        }
    
    def _load_collaboration_strategies(self):
        """Load strategies for multi-agent collaboration."""
        self.collaboration_strategies = {
            'consensus_building': {
                'description': 'Build agreement among all participants',
                'techniques': ['active listening', 'compromise', 'shared goals'],
                'success_indicators': ['agreement reached', 'all participants satisfied']
            },
            'task_delegation': {
                'description': 'Assign tasks based on capabilities and preferences',
                'techniques': ['skill assessment', 'preference matching', 'load balancing'],
                'success_indicators': ['tasks completed', 'efficiency improved']
            },
            'conflict_resolution': {
                'description': 'Resolve disagreements and conflicts',
                'techniques': ['mediation', 'compromise', 'win-win solutions'],
                'success_indicators': ['conflict resolved', 'relationships maintained']
            },
            'knowledge_sharing': {
                'description': 'Share information and expertise',
                'techniques': ['documentation', 'mentoring', 'best practices'],
                'success_indicators': ['knowledge transferred', 'capabilities improved']
            }
        }
    
    def recognize_emotion(self, text: str, agent_id: str = None) -> Dict[str, Any]:
        """Recognize emotions in text or from an agent."""
        try:
            detected_emotions = []
            text_lower = text.lower()
            
            # Analyze text for emotional patterns
            for emotion_type, patterns in self.emotion_patterns.items():
                emotion_score = 0.0
                intensity = 0.5  # Default intensity
                
                # Check for keywords
                for keyword in patterns['keywords']:
                    if keyword in text_lower:
                        emotion_score += 0.3
                
                # Check for expressions
                for expression in patterns['expressions']:
                    if expression in text:
                        emotion_score += 0.4
                
                # Check for intensity indicators
                for indicator in patterns['intensity_indicators']:
                    if indicator in text_lower:
                        intensity = min(1.0, intensity + 0.3)
                
                # If emotion detected above threshold
                if emotion_score >= self.emotion_recognition_threshold:
                    detected_emotions.append({
                        'emotion': emotion_type,
                        'intensity': intensity,
                        'confidence': emotion_score,
                        'triggers': [word for word in text_lower.split() if word in patterns['keywords']]
                    })
            
            # Select the most confident emotion
            if detected_emotions:
                best_emotion = max(detected_emotions, key=lambda x: x['confidence'])
                
                emotional_state = EmotionalState(
                    emotion=best_emotion['emotion'],
                    intensity=best_emotion['intensity'],
                    confidence=best_emotion['confidence'],
                    context=text,
                    triggers=best_emotion['triggers'],
                    timestamp=time.time()
                )
                
                # Store emotional state if agent_id provided
                if agent_id:
                    self.emotional_states[agent_id] = emotional_state
                
                return {
                    "status": "success",
                    "emotion": best_emotion['emotion'].value,
                    "intensity": best_emotion['intensity'],
                    "confidence": best_emotion['confidence'],
                    "triggers": best_emotion['triggers']
                }
            else:
                # Default to neutral if no emotion detected
                return {
                    "status": "success",
                    "emotion": EmotionType.NEUTRAL.value,
                    "intensity": 0.5,
                    "confidence": 0.8,
                    "triggers": []
                }
                
        except Exception as e:
            return {"error": f"Emotion recognition failed: {str(e)}"}
    
    def adapt_to_social_context(self, context: SocialContext, 
                               current_style: str = None) -> Dict[str, Any]:
        """Adapt behavior and communication style to social context."""
        try:
            if context not in self.social_context_rules:
                return {"error": f"Unknown social context: {context}"}
            
            rules = self.social_context_rules[context]
            
            adaptation = {
                "context": context.value,
                "recommended_style": rules['communication_style'],
                "emotional_expression": rules['emotional_expression'],
                "formality_level": rules['formality_level'],
                "appropriate_topics": rules['appropriate_topics'],
                "inappropriate_topics": rules['inappropriate_topics'],
                "adaptation_score": 0.0
            }
            
            # Calculate adaptation score based on current style
            if current_style:
                if current_style == rules['communication_style']:
                    adaptation['adaptation_score'] = 1.0
                elif current_style in ['casual', 'professional'] and rules['communication_style'] in ['casual', 'professional']:
                    adaptation['adaptation_score'] = 0.7
                else:
                    adaptation['adaptation_score'] = 0.3
            
            # Update social stats
            self.social_stats['social_context_adaptations'] += 1
            
            return {
                "status": "success",
                "adaptation": adaptation
            }
            
        except Exception as e:
            return {"error": f"Social context adaptation failed: {str(e)}"}
    
    def understand_mental_state(self, agent_id: str, 
                              available_information: Dict[str, Any]) -> Dict[str, Any]:
        """Implement theory of mind to understand another agent's mental state."""
        try:
            # Extract information about the agent
            beliefs = available_information.get('beliefs', [])
            desires = available_information.get('desires', [])
            intentions = available_information.get('intentions', [])
            knowledge = available_information.get('knowledge', [])
            emotional_state = available_information.get('emotional_state')
            
            # Create mental state model
            mental_state = MentalState(
                agent_id=agent_id,
                beliefs=beliefs,
                desires=desires,
                intentions=intentions,
                knowledge=knowledge,
                emotional_state=emotional_state,
                confidence=0.7,  # Default confidence
                timestamp=time.time()
            )
            
            # Store mental state
            self.mental_states[agent_id] = mental_state
            
            # Analyze mental state complexity
            complexity_score = len(beliefs) + len(desires) + len(intentions) + len(knowledge)
            
            # Determine theory of mind level
            if complexity_score == 0:
                tom_level = TheoryOfMindLevel.ZERO_ORDER
            elif complexity_score <= 5:
                tom_level = TheoryOfMindLevel.FIRST_ORDER
            elif complexity_score <= 10:
                tom_level = TheoryOfMindLevel.SECOND_ORDER
            else:
                tom_level = TheoryOfMindLevel.HIGHER_ORDER
            
            return {
                "status": "success",
                "agent_id": agent_id,
                "mental_state": {
                    "beliefs": beliefs,
                    "desires": desires,
                    "intentions": intentions,
                    "knowledge": knowledge,
                    "emotional_state": emotional_state.value if emotional_state else None
                },
                "theory_of_mind_level": tom_level.value,
                "complexity_score": complexity_score,
                "confidence": mental_state.confidence
            }
            
        except Exception as e:
            return {"error": f"Theory of mind analysis failed: {str(e)}"}
    
    def facilitate_collaboration(self, agents: List[str], 
                               goals: List[str], 
                               strategy: str = None) -> Dict[str, Any]:
        """Facilitate collaboration between multiple agents."""
        try:
            if not strategy:
                strategy = 'consensus_building'  # Default strategy
            
            if strategy not in self.collaboration_strategies:
                return {"error": f"Unknown collaboration strategy: {strategy}"}
            
            strategy_info = self.collaboration_strategies[strategy]
            
            # Create collaboration session
            collaboration_id = f"collab_{int(time.time())}"
            
            # Analyze agent relationships and emotional states
            agent_analysis = {}
            for agent_id in agents:
                emotional_state = self.emotional_states.get(agent_id)
                mental_state = self.mental_states.get(agent_id)
                
                agent_analysis[agent_id] = {
                    "emotional_state": emotional_state.emotion.value if emotional_state else EmotionType.NEUTRAL.value,
                    "mental_state_available": mental_state is not None,
                    "collaboration_readiness": 0.8  # Default readiness
                }
            
            # Generate collaboration plan
            collaboration_plan = {
                "strategy": strategy,
                "description": strategy_info['description'],
                "techniques": strategy_info['techniques'],
                "success_indicators": strategy_info['success_indicators'],
                "agent_analysis": agent_analysis,
                "recommended_approach": self._generate_collaboration_approach(agents, goals, strategy)
            }
            
            # Update social stats
            self.social_stats['total_interactions'] += 1
            
            return {
                "status": "success",
                "collaboration_id": collaboration_id,
                "strategy": strategy,
                "agents": agents,
                "goals": goals,
                "plan": collaboration_plan
            }
            
        except Exception as e:
            return {"error": f"Collaboration facilitation failed: {str(e)}"}
    
    def _generate_collaboration_approach(self, agents: List[str], 
                                       goals: List[str], 
                                       strategy: str) -> Dict[str, Any]:
        """Generate a specific approach for collaboration."""
        if strategy == 'consensus_building':
            return {
                "phase_1": "Individual goal sharing",
                "phase_2": "Common ground identification",
                "phase_3": "Compromise development",
                "phase_4": "Agreement finalization"
            }
        elif strategy == 'task_delegation':
            return {
                "phase_1": "Skill assessment",
                "phase_2": "Task analysis",
                "phase_3": "Assignment optimization",
                "phase_4": "Execution coordination"
            }
        elif strategy == 'conflict_resolution':
            return {
                "phase_1": "Conflict identification",
                "phase_2": "Perspective sharing",
                "phase_3": "Solution brainstorming",
                "phase_4": "Resolution implementation"
            }
        else:
            return {
                "phase_1": "Initial assessment",
                "phase_2": "Strategy implementation",
                "phase_3": "Progress monitoring",
                "phase_4": "Outcome evaluation"
            }
    
    def manage_relationships(self, agent_id: str, 
                           other_agent_id: str, 
                           interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Manage and update social relationships."""
        try:
            relationship_id = f"{agent_id}_{other_agent_id}"
            
            # Get or create relationship
            if relationship_id not in self.relationships:
                relationship = SocialRelationship(
                    relationship_id=relationship_id,
                    participants=[agent_id, other_agent_id],
                    relationship_type="acquaintance",  # Default type
                    strength=0.5,  # Default strength
                    trust_level=0.5,  # Default trust
                    history=[],
                    shared_goals=[],
                    conflicts=[],
                    created_at=time.time(),
                    last_updated=time.time()
                )
                self.relationships[relationship_id] = relationship
            else:
                relationship = self.relationships[relationship_id]
            
            # Update relationship based on interaction
            interaction_quality = interaction_data.get('quality', 0.5)
            interaction_type = interaction_data.get('type', 'neutral')
            
            # Adjust relationship strength
            if interaction_quality > 0.7:
                relationship.strength = min(1.0, relationship.strength + 0.1)
            elif interaction_quality < 0.3:
                relationship.strength = max(0.0, relationship.strength - 0.1)
            
            # Adjust trust level
            if interaction_type == 'cooperative':
                relationship.trust_level = min(1.0, relationship.trust_level + 0.1)
            elif interaction_type == 'conflict':
                relationship.trust_level = max(0.0, relationship.trust_level - 0.1)
            
            # Update relationship type based on strength
            if relationship.strength > 0.8:
                relationship.relationship_type = "close_friend"
            elif relationship.strength > 0.6:
                relationship.relationship_type = "friend"
            elif relationship.strength > 0.4:
                relationship.relationship_type = "acquaintance"
            else:
                relationship.relationship_type = "stranger"
            
            relationship.last_updated = time.time()
            
            return {
                "status": "success",
                "relationship_id": relationship_id,
                "relationship_type": relationship.relationship_type,
                "strength": relationship.strength,
                "trust_level": relationship.trust_level,
                "last_updated": relationship.last_updated
            }
            
        except Exception as e:
            return {"error": f"Relationship management failed: {str(e)}"}
    
    def get_social_stats(self) -> Dict[str, Any]:
        """Get comprehensive social intelligence statistics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_interactions': self.social_stats['total_interactions'],
            'emotional_recognition_accuracy': self.social_stats['emotional_recognition_accuracy'],
            'social_context_adaptations': self.social_stats['social_context_adaptations'],
            'collaboration_success_rate': self.social_stats['collaboration_success_rate'],
            'relationship_management_score': self.social_stats['relationship_management_score'],
            'active_emotional_states': len(self.emotional_states),
            'tracked_mental_states': len(self.mental_states),
            'managed_relationships': len(self.relationships),
            'social_network_size': len(self.social_networks.nodes)
        }
    
    def export_social_data(self, filename: str = None) -> str:
        """Export social intelligence data to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"social_export_{timestamp}.json"
        
        filepath = os.path.join(self.social_dir, filename)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'social_stats': self.get_social_stats(),
            'emotional_states': {k: asdict(v) for k, v in self.emotional_states.items()},
            'mental_states': {k: asdict(v) for k, v in self.mental_states.items()},
            'relationships': {k: asdict(v) for k, v in self.relationships.items()},
            'social_interactions': [asdict(interaction) for interaction in self.social_interactions]
        }
        
        # Convert enum values to strings
        for emotional_state in export_data['emotional_states'].values():
            emotional_state['emotion'] = emotional_state['emotion'].value
        
        for mental_state in export_data['mental_states'].values():
            if mental_state['emotional_state']:
                mental_state['emotional_state']['emotion'] = mental_state['emotional_state']['emotion'].value
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath 