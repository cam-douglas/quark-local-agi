#!/usr/bin/env python3
"""
Advanced Social Understanding Agent - Pillar 33
Advanced social intelligence and interpersonal understanding
"""

import os
import sys
import time
import logging
import json
import random
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.base import Agent
from agents.emotional_intelligence_agent import (
    EmotionType, EmotionIntensity, EmotionalState, 
    Emotion, EmotionalContext, EmotionalAnalysis
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SocialContext(Enum):
    """Types of social contexts"""
    ONE_ON_ONE = "one_on_one"
    GROUP = "group"
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FAMILY = "family"
    ROMANTIC = "romantic"
    ONLINE = "online"
    PUBLIC = "public"

class RelationshipType(Enum):
    """Types of relationships"""
    STRANGER = "stranger"
    ACQUAINTANCE = "acquaintance"
    FRIEND = "friend"
    CLOSE_FRIEND = "close_friend"
    FAMILY = "family"
    ROMANTIC = "romantic"
    COLLEAGUE = "colleague"
    MENTOR = "mentor"
    STUDENT = "student"

class CommunicationStyle(Enum):
    """Communication styles"""
    ASSERTIVE = "assertive"
    PASSIVE = "passive"
    AGGRESSIVE = "aggressive"
    PASSIVE_AGGRESSIVE = "passive_aggressive"
    COLLABORATIVE = "collaborative"
    ANALYTICAL = "analytical"
    EXPRESSIVE = "expressive"
    RESERVED = "reserved"

class SocialIntelligence(Enum):
    """Social intelligence dimensions"""
    EMOTIONAL_AWARENESS = "emotional_awareness"
    SOCIAL_PERCEPTION = "social_perception"
    INTERPERSONAL_SKILLS = "interpersonal_skills"
    CONFLICT_RESOLUTION = "conflict_resolution"
    EMPATHY = "empathy"
    PERSUASION = "persuasion"
    LEADERSHIP = "leadership"
    TEAMWORK = "teamwork"

@dataclass
class SocialInteraction:
    """A social interaction context"""
    participants: List[str]
    context: SocialContext
    relationship_types: Dict[str, RelationshipType]
    communication_styles: Dict[str, CommunicationStyle]
    power_dynamics: Dict[str, float]  # 0.0 to 1.0 scale
    cultural_backgrounds: Dict[str, str]
    emotional_states: Dict[str, EmotionalState]
    conversation_history: List[Dict[str, Any]]
    goals: List[str]
    constraints: List[str]
    timestamp: datetime

@dataclass
class SocialAnalysis:
    """Complete social analysis"""
    interaction: SocialInteraction
    emotional_dynamics: Dict[str, EmotionalAnalysis]
    relationship_insights: Dict[str, Any]
    communication_patterns: Dict[str, Any]
    power_analysis: Dict[str, Any]
    cultural_considerations: Dict[str, Any]
    conflict_potential: float
    collaboration_opportunities: List[str]
    social_recommendations: List[str]
    empathy_score: float
    social_intelligence_score: float
    timestamp: datetime

@dataclass
class SocialMetrics:
    """Metrics for social intelligence performance"""
    total_analyses: int
    successful_analyses: int
    average_empathy_score: float
    average_social_intelligence_score: float
    relationship_understanding_accuracy: float
    communication_effectiveness: float
    conflict_resolution_success: float
    cultural_sensitivity: float

class SocialUnderstandingAgent(Agent):
    """Advanced social understanding and interpersonal intelligence agent"""
    
    def __init__(self, model_name: str = "social_understanding_agent"):
        super().__init__(model_name)
        self.model_name = model_name
        self.social_analyses = []
        self.metrics = SocialMetrics(
            total_analyses=0,
            successful_analyses=0,
            average_empathy_score=0.0,
            average_social_intelligence_score=0.0,
            relationship_understanding_accuracy=0.0,
            communication_effectiveness=0.0,
            conflict_resolution_success=0.0,
            cultural_sensitivity=0.0
        )
        
        # Cultural and social knowledge bases
        self.cultural_norms = {
            "western": {
                "communication_style": "direct",
                "personal_space": "moderate",
                "time_orientation": "linear",
                "power_distance": "low"
            },
            "eastern": {
                "communication_style": "indirect",
                "personal_space": "close",
                "time_orientation": "circular",
                "power_distance": "high"
            },
            "middle_eastern": {
                "communication_style": "expressive",
                "personal_space": "close",
                "time_orientation": "flexible",
                "power_distance": "high"
            }
        }
        
        self.relationship_patterns = {
            "friendship": {
                "communication_frequency": "high",
                "emotional_support": "high",
                "conflict_resolution": "collaborative"
            },
            "professional": {
                "communication_frequency": "moderate",
                "emotional_support": "low",
                "conflict_resolution": "formal"
            },
            "romantic": {
                "communication_frequency": "very_high",
                "emotional_support": "very_high",
                "conflict_resolution": "intimate"
            }
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the social understanding model"""
        try:
            # Initialize with basic social understanding capabilities
            self.model_loaded = True
            logger.info("Social understanding model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading social understanding model: {e}")
            self.model_loaded = False
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate social understanding analysis"""
        try:
            # Parse the social interaction request
            interaction = self._parse_social_request(prompt, kwargs)
            
            # Perform comprehensive social analysis
            analysis = self._perform_social_analysis(interaction)
            
            # Format the result
            result = self._format_social_result(analysis)
            
            # Update metrics
            self._update_metrics(analysis)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in social understanding generation: {e}")
            return f"Social analysis error: {e}"
    
    def _parse_social_request(self, prompt: str, kwargs: Dict[str, Any]) -> SocialInteraction:
        """Parse social interaction request"""
        # Extract participants from prompt or kwargs
        participants = kwargs.get('participants', ['user', 'other'])
        
        # Determine context from prompt
        context = self._determine_social_context(prompt)
        
        # Extract relationship information
        relationship_types = self._extract_relationship_types(participants, prompt)
        
        # Analyze communication styles
        communication_styles = self._analyze_communication_styles(participants, prompt)
        
        # Assess power dynamics
        power_dynamics = self._assess_power_dynamics(participants, prompt)
        
        # Extract cultural backgrounds
        cultural_backgrounds = self._extract_cultural_backgrounds(participants, prompt)
        
        # Analyze emotional states
        emotional_states = self._analyze_emotional_states(participants, prompt)
        
        return SocialInteraction(
            participants=participants,
            context=context,
            relationship_types=relationship_types,
            communication_styles=communication_styles,
            power_dynamics=power_dynamics,
            cultural_backgrounds=cultural_backgrounds,
            emotional_states=emotional_states,
            conversation_history=kwargs.get('conversation_history', []),
            goals=kwargs.get('goals', []),
            constraints=kwargs.get('constraints', []),
            timestamp=datetime.now()
        )
    
    def _determine_social_context(self, prompt: str) -> SocialContext:
        """Determine the social context from the prompt"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['meeting', 'work', 'office', 'business']):
            return SocialContext.PROFESSIONAL
        elif any(word in prompt_lower for word in ['family', 'parent', 'child', 'sibling']):
            return SocialContext.FAMILY
        elif any(word in prompt_lower for word in ['date', 'romantic', 'partner', 'love']):
            return SocialContext.ROMANTIC
        elif any(word in prompt_lower for word in ['group', 'team', 'crowd', 'party']):
            return SocialContext.GROUP
        elif any(word in prompt_lower for word in ['online', 'chat', 'message', 'social media']):
            return SocialContext.ONLINE
        elif any(word in prompt_lower for word in ['public', 'audience', 'speech']):
            return SocialContext.PUBLIC
        else:
            return SocialContext.ONE_ON_ONE
    
    def _extract_relationship_types(self, participants: List[str], prompt: str) -> Dict[str, RelationshipType]:
        """Extract relationship types between participants"""
        relationship_types = {}
        prompt_lower = prompt.lower()
        
        for participant in participants:
            if 'friend' in prompt_lower or 'buddy' in prompt_lower:
                relationship_types[participant] = RelationshipType.FRIEND
            elif 'family' in prompt_lower or 'parent' in prompt_lower or 'child' in prompt_lower:
                relationship_types[participant] = RelationshipType.FAMILY
            elif 'colleague' in prompt_lower or 'work' in prompt_lower:
                relationship_types[participant] = RelationshipType.COLLEAGUE
            elif 'stranger' in prompt_lower or 'unknown' in prompt_lower:
                relationship_types[participant] = RelationshipType.STRANGER
            else:
                relationship_types[participant] = RelationshipType.ACQUAINTANCE
        
        return relationship_types
    
    def _analyze_communication_styles(self, participants: List[str], prompt: str) -> Dict[str, CommunicationStyle]:
        """Analyze communication styles of participants"""
        communication_styles = {}
        prompt_lower = prompt.lower()
        
        for participant in participants:
            if any(word in prompt_lower for word in ['assertive', 'confident', 'direct']):
                communication_styles[participant] = CommunicationStyle.ASSERTIVE
            elif any(word in prompt_lower for word in ['passive', 'quiet', 'shy']):
                communication_styles[participant] = CommunicationStyle.PASSIVE
            elif any(word in prompt_lower for word in ['aggressive', 'angry', 'hostile']):
                communication_styles[participant] = CommunicationStyle.AGGRESSIVE
            elif any(word in prompt_lower for word in ['analytical', 'logical', 'detailed']):
                communication_styles[participant] = CommunicationStyle.ANALYTICAL
            elif any(word in prompt_lower for word in ['expressive', 'emotional', 'enthusiastic']):
                communication_styles[participant] = CommunicationStyle.EXPRESSIVE
            else:
                communication_styles[participant] = CommunicationStyle.COLLABORATIVE
        
        return communication_styles
    
    def _assess_power_dynamics(self, participants: List[str], prompt: str) -> Dict[str, float]:
        """Assess power dynamics between participants"""
        power_dynamics = {}
        prompt_lower = prompt.lower()
        
        # Check if any participant name contains power-related words
        for participant in participants:
            participant_lower = participant.lower()
            if any(word in participant_lower for word in ['boss', 'manager', 'leader', 'authority']):
                power_dynamics[participant] = 0.8
            elif any(word in participant_lower for word in ['subordinate', 'employee', 'student']):
                power_dynamics[participant] = 0.3
            elif any(word in participant_lower for word in ['equal', 'peer', 'friend']):
                power_dynamics[participant] = 0.5
            else:
                power_dynamics[participant] = 0.5
        
        return power_dynamics
    
    def _extract_cultural_backgrounds(self, participants: List[str], prompt: str) -> Dict[str, str]:
        """Extract cultural backgrounds of participants"""
        cultural_backgrounds = {}
        prompt_lower = prompt.lower()
        
        for participant in participants:
            participant_lower = participant.lower()
            if any(word in participant_lower for word in ['asian', 'chinese', 'japanese', 'korean']):
                cultural_backgrounds[participant] = "eastern"
            elif any(word in participant_lower for word in ['middle_eastern', 'arabic', 'islamic']):
                cultural_backgrounds[participant] = "middle_eastern"
            else:
                cultural_backgrounds[participant] = "western"
        
        return cultural_backgrounds
    
    def _analyze_emotional_states(self, participants: List[str], prompt: str) -> Dict[str, EmotionalState]:
        """Analyze emotional states of participants"""
        emotional_states = {}
        prompt_lower = prompt.lower()
        
        for participant in participants:
            if any(word in prompt_lower for word in ['happy', 'joy', 'excited', 'positive']):
                emotional_states[participant] = EmotionalState.POSITIVE
            elif any(word in prompt_lower for word in ['sad', 'angry', 'frustrated', 'negative']):
                emotional_states[participant] = EmotionalState.NEGATIVE
            elif any(word in prompt_lower for word in ['confused', 'mixed', 'complex']):
                emotional_states[participant] = EmotionalState.MIXED
            else:
                emotional_states[participant] = EmotionalState.NEUTRAL
        
        return emotional_states
    
    def _perform_social_analysis(self, interaction: SocialInteraction) -> SocialAnalysis:
        """Perform comprehensive social analysis"""
        # Analyze emotional dynamics
        emotional_dynamics = self._analyze_emotional_dynamics(interaction)
        
        # Analyze relationship insights
        relationship_insights = self._analyze_relationships(interaction)
        
        # Analyze communication patterns
        communication_patterns = self._analyze_communication_patterns(interaction)
        
        # Analyze power dynamics
        power_analysis = self._analyze_power_dynamics(interaction)
        
        # Analyze cultural considerations
        cultural_considerations = self._analyze_cultural_factors(interaction)
        
        # Assess conflict potential
        conflict_potential = self._assess_conflict_potential(interaction)
        
        # Identify collaboration opportunities
        collaboration_opportunities = self._identify_collaboration_opportunities(interaction)
        
        # Generate social recommendations
        social_recommendations = self._generate_social_recommendations(interaction)
        
        # Calculate empathy score
        empathy_score = self._calculate_empathy_score(interaction)
        
        # Calculate social intelligence score
        social_intelligence_score = self._calculate_social_intelligence_score(interaction)
        
        return SocialAnalysis(
            interaction=interaction,
            emotional_dynamics=emotional_dynamics,
            relationship_insights=relationship_insights,
            communication_patterns=communication_patterns,
            power_analysis=power_analysis,
            cultural_considerations=cultural_considerations,
            conflict_potential=conflict_potential,
            collaboration_opportunities=collaboration_opportunities,
            social_recommendations=social_recommendations,
            empathy_score=empathy_score,
            social_intelligence_score=social_intelligence_score,
            timestamp=datetime.now()
        )
    
    def _analyze_emotional_dynamics(self, interaction: SocialInteraction) -> Dict[str, EmotionalAnalysis]:
        """Analyze emotional dynamics between participants"""
        emotional_dynamics = {}
        
        for participant in interaction.participants:
            # Create emotional context for each participant
            emotional_context = EmotionalContext(
                text=f"Social interaction involving {participant}",
                speaker=participant,
                listener="system",
                situation=interaction.context.value,
                relationship=interaction.relationship_types.get(participant, RelationshipType.ACQUAINTANCE).value,
                cultural_context=interaction.cultural_backgrounds.get(participant, "western"),
                emotional_history=[],
                current_mood=interaction.emotional_states.get(participant, EmotionalState.NEUTRAL).value,
                stress_level=0.5,
                social_context=interaction.context.value
            )
            
            # Perform emotional analysis
            emotional_analysis = EmotionalAnalysis(
                primary_emotion=Emotion(
                    type=EmotionType.JOY if interaction.emotional_states.get(participant) == EmotionalState.POSITIVE else EmotionType.SADNESS,
                    intensity=EmotionIntensity.MODERATE,
                    confidence=0.7,
                    triggers=[],
                    expressions=[],
                    physiological_responses=[],
                    cognitive_effects=[],
                    behavioral_implications=[],
                    timestamp=datetime.now()
                ),
                secondary_emotions=[],
                emotional_state=interaction.emotional_states.get(participant, EmotionalState.NEUTRAL),
                empathy_score=0.8,
                emotional_intelligence_score=0.8,
                recommendations=[],
                emotional_insights=[],
                response_suggestions=[],
                timestamp=datetime.now()
            )
            
            emotional_dynamics[participant] = emotional_analysis
        
        return emotional_dynamics
    
    def _analyze_relationships(self, interaction: SocialInteraction) -> Dict[str, Any]:
        """Analyze relationship dynamics"""
        insights = {
            'relationship_types': interaction.relationship_types,
            'trust_levels': {},
            'communication_frequency': {},
            'emotional_support': {},
            'conflict_history': {},
            'relationship_strength': {}
        }
        
        for participant in interaction.participants:
            relationship_type = interaction.relationship_types.get(participant, RelationshipType.ACQUAINTANCE)
            
            # Assess trust levels based on relationship type
            if relationship_type == RelationshipType.CLOSE_FRIEND:
                insights['trust_levels'][participant] = 0.9
            elif relationship_type == RelationshipType.FRIEND:
                insights['trust_levels'][participant] = 0.7
            elif relationship_type == RelationshipType.COLLEAGUE:
                insights['trust_levels'][participant] = 0.5
            else:
                insights['trust_levels'][participant] = 0.3
            
            # Assess communication frequency
            if relationship_type in [RelationshipType.CLOSE_FRIEND, RelationshipType.ROMANTIC]:
                insights['communication_frequency'][participant] = "high"
            elif relationship_type == RelationshipType.COLLEAGUE:
                insights['communication_frequency'][participant] = "moderate"
            else:
                insights['communication_frequency'][participant] = "low"
        
        return insights
    
    def _analyze_communication_patterns(self, interaction: SocialInteraction) -> Dict[str, Any]:
        """Analyze communication patterns"""
        patterns = {
            'dominant_speakers': [],
            'communication_style_matches': {},
            'conversation_flow': 'balanced',
            'interruption_patterns': {},
            'listening_behaviors': {},
            'nonverbal_cues': {}
        }
        
        # Identify dominant speakers based on power dynamics
        for participant, power in interaction.power_dynamics.items():
            if power > 0.7:
                patterns['dominant_speakers'].append(participant)
        
        # Analyze communication style compatibility
        for participant in interaction.participants:
            style = interaction.communication_styles.get(participant, CommunicationStyle.COLLABORATIVE)
            patterns['communication_style_matches'][participant] = {
                'style': style.value,
                'effectiveness': 0.8 if style == CommunicationStyle.COLLABORATIVE else 0.6
            }
        
        return patterns
    
    def _analyze_power_dynamics(self, interaction: SocialInteraction) -> Dict[str, Any]:
        """Analyze power dynamics in the interaction"""
        power_analysis = {
            'power_imbalance': 0.0,
            'dominant_participants': [],
            'subordinate_participants': [],
            'power_conflicts': [],
            'collaboration_potential': 0.8
        }
        
        # Calculate power imbalance
        power_values = list(interaction.power_dynamics.values())
        if len(power_values) > 1:
            power_analysis['power_imbalance'] = max(power_values) - min(power_values)
        
        # Identify dominant and subordinate participants
        for participant, power in interaction.power_dynamics.items():
            if power > 0.7:
                power_analysis['dominant_participants'].append(participant)
            elif power <= 0.3:
                power_analysis['subordinate_participants'].append(participant)
        
        return power_analysis
    
    def _analyze_cultural_factors(self, interaction: SocialInteraction) -> Dict[str, Any]:
        """Analyze cultural factors in the interaction"""
        cultural_analysis = {
            'cultural_differences': [],
            'communication_barriers': [],
            'cultural_sensitivity': 0.8,
            'adaptation_recommendations': []
        }
        
        # Identify cultural differences
        cultural_backgrounds = list(interaction.cultural_backgrounds.values())
        if len(set(cultural_backgrounds)) > 1:
            cultural_analysis['cultural_differences'] = list(set(cultural_backgrounds))
        
        # Assess communication barriers
        for background in cultural_analysis['cultural_differences']:
            if background in self.cultural_norms:
                norms = self.cultural_norms[background]
                if norms['communication_style'] == 'indirect':
                    cultural_analysis['communication_barriers'].append('indirect_communication')
        
        return cultural_analysis
    
    def _assess_conflict_potential(self, interaction: SocialInteraction) -> float:
        """Assess potential for conflict in the interaction"""
        conflict_factors = 0.0
        
        # Check for emotional conflicts
        negative_emotions = sum(1 for state in interaction.emotional_states.values() 
                              if state == EmotionalState.NEGATIVE)
        if negative_emotions > 0:
            conflict_factors += 0.3
        
        # Check for power conflicts
        power_values = list(interaction.power_dynamics.values())
        if len(power_values) > 1 and max(power_values) - min(power_values) > 0.5:
            conflict_factors += 0.2
        
        # Check for communication style conflicts
        aggressive_styles = sum(1 for style in interaction.communication_styles.values() 
                              if style == CommunicationStyle.AGGRESSIVE)
        if aggressive_styles > 0:
            conflict_factors += 0.3
        
        return min(conflict_factors, 1.0)
    
    def _identify_collaboration_opportunities(self, interaction: SocialInteraction) -> List[str]:
        """Identify opportunities for collaboration"""
        opportunities = []
        
        # Check for complementary skills
        if len(interaction.participants) > 1:
            opportunities.append("Leverage complementary strengths")
        
        # Check for shared goals
        if interaction.goals:
            opportunities.append("Align on shared objectives")
        
        # Check for positive emotional states
        positive_emotions = sum(1 for state in interaction.emotional_states.values() 
                              if state == EmotionalState.POSITIVE)
        if positive_emotions > 0:
            opportunities.append("Build on positive emotional energy")
        
        # Check for collaborative communication styles
        collaborative_styles = sum(1 for style in interaction.communication_styles.values() 
                                if style == CommunicationStyle.COLLABORATIVE)
        if collaborative_styles > 0:
            opportunities.append("Foster collaborative communication")
        
        return opportunities
    
    def _generate_social_recommendations(self, interaction: SocialInteraction) -> List[str]:
        """Generate social recommendations"""
        recommendations = []
        
        # Recommendations based on context
        if interaction.context == SocialContext.PROFESSIONAL:
            recommendations.extend([
                "Maintain professional boundaries",
                "Focus on shared objectives",
                "Use clear, direct communication"
            ])
        elif interaction.context == SocialContext.FAMILY:
            recommendations.extend([
                "Show emotional support",
                "Practice active listening",
                "Be patient and understanding"
            ])
        
        # Recommendations based on relationship types
        for participant, relationship_type in interaction.relationship_types.items():
            if relationship_type == RelationshipType.STRANGER:
                recommendations.append("Establish trust gradually")
            elif relationship_type == RelationshipType.CLOSE_FRIEND:
                recommendations.append("Leverage existing trust and rapport")
        
        # Recommendations based on communication styles
        for participant, style in interaction.communication_styles.items():
            if style == CommunicationStyle.AGGRESSIVE:
                recommendations.append("De-escalate aggressive communication")
            elif style == CommunicationStyle.PASSIVE:
                recommendations.append("Encourage more assertive expression")
        
        return recommendations
    
    def _calculate_empathy_score(self, interaction: SocialInteraction) -> float:
        """Calculate empathy score for the interaction"""
        base_score = 0.5
        
        # Add score for emotional awareness
        emotional_states = list(interaction.emotional_states.values())
        if EmotionalState.POSITIVE in emotional_states:
            base_score += 0.2
        if EmotionalState.NEGATIVE in emotional_states:
            base_score += 0.1  # Recognition of negative emotions
        
        # Add score for relationship understanding
        if RelationshipType.CLOSE_FRIEND in interaction.relationship_types.values():
            base_score += 0.2
        
        # Add score for cultural sensitivity
        cultural_backgrounds = list(interaction.cultural_backgrounds.values())
        if len(set(cultural_backgrounds)) > 1:
            base_score += 0.1  # Recognition of cultural diversity
        
        return min(base_score, 1.0)
    
    def _calculate_social_intelligence_score(self, interaction: SocialInteraction) -> float:
        """Calculate social intelligence score"""
        base_score = 0.5
        
        # Add score for relationship understanding
        relationship_types = list(interaction.relationship_types.values())
        if RelationshipType.CLOSE_FRIEND in relationship_types:
            base_score += 0.2
        
        # Add score for communication style awareness
        communication_styles = list(interaction.communication_styles.values())
        if CommunicationStyle.COLLABORATIVE in communication_styles:
            base_score += 0.2
        
        # Add score for power dynamics understanding
        power_values = list(interaction.power_dynamics.values())
        if len(power_values) > 1:
            power_imbalance = max(power_values) - min(power_values)
            if power_imbalance < 0.3:  # Balanced power dynamics
                base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _format_social_result(self, analysis: SocialAnalysis) -> str:
        """Format social analysis result"""
        result = f"""
Advanced Social Understanding Analysis
====================================

Interaction Context: {analysis.interaction.context.value}
Participants: {', '.join(analysis.interaction.participants)}
Timestamp: {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Emotional Dynamics:
- Primary emotional states: {[f'{p}: {s.value}' for p, s in analysis.interaction.emotional_states.items()]}
- Empathy score: {analysis.empathy_score:.2%}
- Social intelligence score: {analysis.social_intelligence_score:.2%}

Relationship Insights:
- Relationship types: {[f'{p}: {r.value}' for p, r in analysis.interaction.relationship_types.items()]}
- Trust levels: {analysis.relationship_insights.get('trust_levels', {})}
- Communication frequency: {analysis.relationship_insights.get('communication_frequency', {})}

Communication Patterns:
- Dominant speakers: {analysis.communication_patterns.get('dominant_speakers', [])}
- Communication styles: {[f'{p}: {s.value}' for p, s in analysis.interaction.communication_styles.items()]}

Power Dynamics:
- Power imbalance: {analysis.power_analysis.get('power_imbalance', 0.0):.2f}
- Dominant participants: {analysis.power_analysis.get('dominant_participants', [])}
- Collaboration potential: {analysis.power_analysis.get('collaboration_potential', 0.0):.2f}

Cultural Considerations:
- Cultural backgrounds: {[f'{p}: {c}' for p, c in analysis.interaction.cultural_backgrounds.items()]}
- Cultural differences: {analysis.cultural_considerations.get('cultural_differences', [])}
- Cultural sensitivity: {analysis.cultural_considerations.get('cultural_sensitivity', 0.0):.2f}

Conflict Assessment:
- Conflict potential: {analysis.conflict_potential:.2f}
- Collaboration opportunities: {analysis.collaboration_opportunities}

Social Recommendations:
{chr(10).join(f'- {rec}' for rec in analysis.social_recommendations)}

Overall Assessment:
The interaction shows {'high' if analysis.social_intelligence_score > 0.7 else 'moderate' if analysis.social_intelligence_score > 0.4 else 'low'} social intelligence 
with {'strong' if analysis.empathy_score > 0.7 else 'moderate' if analysis.empathy_score > 0.4 else 'limited'} empathy capabilities.
"""
        return result
    
    def _update_metrics(self, analysis: SocialAnalysis):
        """Update performance metrics"""
        self.metrics.total_analyses += 1
        self.metrics.successful_analyses += 1
        
        # Update average scores
        if self.metrics.total_analyses > 0:
            self.metrics.average_empathy_score = (
                (self.metrics.average_empathy_score * (self.metrics.total_analyses - 1) + analysis.empathy_score) 
                / self.metrics.total_analyses
            )
            self.metrics.average_social_intelligence_score = (
                (self.metrics.average_social_intelligence_score * (self.metrics.total_analyses - 1) + analysis.social_intelligence_score) 
                / self.metrics.total_analyses
            )
        
        # Store analysis
        self.social_analyses.append(analysis)
    
    def get_social_history(self) -> List[Dict[str, Any]]:
        """Get social interaction history"""
        history = []
        for analysis in self.social_analyses[-10:]:  # Last 10 analyses
            history.append({
                'timestamp': analysis.timestamp.isoformat(),
                'participants': analysis.interaction.participants,
                'context': analysis.interaction.context.value,
                'empathy_score': analysis.empathy_score,
                'social_intelligence_score': analysis.social_intelligence_score,
                'conflict_potential': analysis.conflict_potential
            })
        return history
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            'total_analyses': self.metrics.total_analyses,
            'successful_analyses': self.metrics.successful_analyses,
            'average_empathy_score': self.metrics.average_empathy_score,
            'average_social_intelligence_score': self.metrics.average_social_intelligence_score,
            'relationship_understanding_accuracy': self.metrics.relationship_understanding_accuracy,
            'communication_effectiveness': self.metrics.communication_effectiveness,
            'conflict_resolution_success': self.metrics.conflict_resolution_success,
            'cultural_sensitivity': self.metrics.cultural_sensitivity
        } 