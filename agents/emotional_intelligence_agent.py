#!/usr/bin/env python3
"""
Emotional Intelligence Agent - Pillar 32
Advanced emotional intelligence system
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
# Import will be done locally to avoid circular imports

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """Types of emotions"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    LOVE = "love"
    CONTEMPT = "contempt"
    SHAME = "shame"
    GUILT = "guilt"
    PRIDE = "pride"
    ENVY = "envy"
    HOPE = "hope"
    GRATITUDE = "gratitude"
    COMPASSION = "compassion"
    ANXIETY = "anxiety"

class EmotionIntensity(Enum):
    """Emotion intensity levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

class EmotionalState(Enum):
    """Emotional states"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"
    COMPLEX = "complex"

class EmpathyType(Enum):
    """Types of empathy"""
    COGNITIVE = "cognitive"
    EMOTIONAL = "emotional"
    COMPASSIONATE = "compassionate"
    PERSPECTIVE_TAKING = "perspective_taking"

@dataclass
class EmotionalContext:
    """Context for emotional analysis"""
    text: str
    speaker: str
    listener: str
    situation: str
    relationship: str
    cultural_context: str
    emotional_history: List[Dict[str, Any]]
    current_mood: str
    stress_level: float
    social_context: str

@dataclass
class Emotion:
    """An identified emotion"""
    type: EmotionType
    intensity: EmotionIntensity
    confidence: float
    triggers: List[str]
    expressions: List[str]
    physiological_responses: List[str]
    cognitive_effects: List[str]
    behavioral_implications: List[str]
    timestamp: datetime

@dataclass
class EmotionalAnalysis:
    """Complete emotional analysis"""
    primary_emotion: Emotion
    secondary_emotions: List[Emotion]
    emotional_state: EmotionalState
    empathy_score: float
    emotional_intelligence_score: float
    recommendations: List[str]
    emotional_insights: List[str]
    response_suggestions: List[str]
    timestamp: datetime

@dataclass
class EmotionalMetrics:
    """Metrics for emotional intelligence performance"""
    total_analyses: int
    successful_analyses: int
    average_empathy_score: float
    average_ei_score: float
    emotion_recognition_accuracy: float
    response_appropriateness: float
    emotional_learning_rate: float

class EmotionalIntelligenceAgent(Agent):
    """
    Advanced emotional intelligence agent
    Capable of understanding, analyzing, and responding to emotions
    """
    
    def __init__(self):
        super().__init__("emotional_intelligence")
        self.emotional_analyses: List[EmotionalAnalysis] = []
        self.emotion_patterns: Dict[str, Any] = {}
        self.empathy_models: Dict[str, Any] = {}
        self.metrics = EmotionalMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.emotion_database: Dict[str, Dict[str, Any]] = {}
        self.cultural_emotional_norms: Dict[str, Dict[str, Any]] = {}
        self.emotional_vocabulary: Dict[str, List[str]] = {}
        
        # Initialize emotional safety monitor
        from alignment.emotional_safety import EmotionalSafetyMonitor
        self.emotional_safety_monitor = EmotionalSafetyMonitor()
        
    def load_model(self):
        """Load emotional intelligence models"""
        logger.info("Loading emotional intelligence models...")
        
        # Initialize empathy models
        self.empathy_models = {
            "emotion_recognition": self._recognize_emotions,
            "empathy_generation": self._generate_empathy,
            "emotional_response": self._generate_emotional_response,
            "perspective_taking": self._take_perspective,
            "emotional_validation": self._validate_emotions,
            "emotional_guidance": self._provide_emotional_guidance
        }
        
        # Initialize emotion database
        self.emotion_database = {
            "joy": {
                "triggers": ["success", "achievement", "celebration", "positive_news"],
                "expressions": ["smiling", "laughing", "excitement", "enthusiasm"],
                "physiological": ["increased_energy", "warmth", "lightness"],
                "cognitive": ["optimism", "creativity", "focus"],
                "behavioral": ["sharing", "helping", "celebrating"]
            },
            "sadness": {
                "triggers": ["loss", "disappointment", "loneliness", "failure"],
                "expressions": ["tears", "withdrawal", "quietness", "melancholy"],
                "physiological": ["heaviness", "fatigue", "slowed_movement"],
                "cognitive": ["pessimism", "rumination", "difficulty_concentrating"],
                "behavioral": ["isolation", "reduced_activity", "seeking_comfort"]
            },
            "anger": {
                "triggers": ["injustice", "frustration", "threat", "disrespect"],
                "expressions": ["yelling", "tension", "aggression", "irritation"],
                "physiological": ["increased_heart_rate", "tension", "heat"],
                "cognitive": ["narrowed_focus", "blame", "defensiveness"],
                "behavioral": ["confrontation", "withdrawal", "destructive_actions"]
            },
            "fear": {
                "triggers": ["danger", "uncertainty", "threat", "unknown"],
                "expressions": ["anxiety", "tension", "avoidance", "worry"],
                "physiological": ["rapid_heartbeat", "sweating", "tension"],
                "cognitive": ["hypervigilance", "catastrophizing", "difficulty_thinking"],
                "behavioral": ["flight", "freeze", "avoidance", "seeking_safety"]
            },
            "love": {
                "triggers": ["connection", "care", "intimacy", "appreciation"],
                "expressions": ["warmth", "affection", "kindness", "generosity"],
                "physiological": ["warmth", "calmness", "contentment"],
                "cognitive": ["positive_bias", "generosity", "forgiveness"],
                "behavioral": ["caring", "sharing", "protecting", "nurturing"]
            }
        }
        
        # Initialize cultural emotional norms
        self.cultural_emotional_norms = {
            "western": {
                "individualistic": True,
                "emotional_expression": "moderate",
                "empathy_style": "cognitive",
                "conflict_resolution": "direct"
            },
            "eastern": {
                "collectivistic": True,
                "emotional_expression": "restrained",
                "empathy_style": "emotional",
                "conflict_resolution": "harmony"
            },
            "middle_eastern": {
                "collectivistic": True,
                "emotional_expression": "passionate",
                "empathy_style": "compassionate",
                "conflict_resolution": "honor"
            }
        }
        
        # Initialize emotional vocabulary
        self.emotional_vocabulary = {
            "joy": ["happy", "excited", "thrilled", "elated", "ecstatic", "delighted"],
            "sadness": ["sad", "depressed", "melancholy", "grief", "sorrow", "despair"],
            "anger": ["angry", "furious", "irritated", "enraged", "frustrated", "hostile"],
            "fear": ["afraid", "terrified", "anxious", "worried", "scared", "panicked"],
            "love": ["loving", "caring", "affectionate", "devoted", "passionate", "tender"]
        }
        
        logger.info("✅ Emotional intelligence models loaded successfully")
        
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate emotional analysis based on the prompt"""
        try:
            # Parse the emotional request
            context = self._parse_emotional_request(prompt, kwargs)
            
            # Perform emotional analysis
            analysis = self._perform_emotional_analysis(context)
            
            # Return the emotional analysis result
            return self._format_emotional_result(analysis)
            
        except Exception as e:
            logger.error(f"Error in emotional analysis: {e}")
            return f"Emotional analysis error: {str(e)}"
    
    def _parse_emotional_request(self, prompt: str, kwargs: Dict[str, Any]) -> EmotionalContext:
        """Parse an emotional request into a structured context"""
        # Extract emotional parameters
        speaker = kwargs.get('speaker', 'unknown')
        listener = kwargs.get('listener', 'system')
        situation = kwargs.get('situation', 'general')
        relationship = kwargs.get('relationship', 'neutral')
        cultural_context = kwargs.get('cultural_context', 'western')
        emotional_history = kwargs.get('emotional_history', [])
        current_mood = kwargs.get('current_mood', 'neutral')
        stress_level = kwargs.get('stress_level', 0.5)
        social_context = kwargs.get('social_context', 'casual')
        
        return EmotionalContext(
            text=prompt,
            speaker=speaker,
            listener=listener,
            situation=situation,
            relationship=relationship,
            cultural_context=cultural_context,
            emotional_history=emotional_history,
            current_mood=current_mood,
            stress_level=stress_level,
            social_context=social_context
        )
    
    def _perform_emotional_analysis(self, context: EmotionalContext) -> EmotionalAnalysis:
        """Perform comprehensive emotional analysis"""
        # Recognize primary emotion
        primary_emotion = self._recognize_primary_emotion(context)
        
        # Identify secondary emotions
        secondary_emotions = self._identify_secondary_emotions(context, primary_emotion)
        
        # Determine emotional state
        emotional_state = self._determine_emotional_state(primary_emotion, secondary_emotions)
        
        # Calculate empathy score
        empathy_score = self._calculate_empathy_score(context, primary_emotion)
        
        # Calculate emotional intelligence score
        ei_score = self._calculate_ei_score(context, primary_emotion, empathy_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(context, primary_emotion, emotional_state)
        
        # Generate emotional insights
        emotional_insights = self._generate_emotional_insights(context, primary_emotion)
        
        # Generate response suggestions
        response_suggestions = self._generate_response_suggestions(context, primary_emotion, empathy_score)
        
        # Create emotional analysis
        analysis = EmotionalAnalysis(
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            emotional_state=emotional_state,
            empathy_score=empathy_score,
            emotional_intelligence_score=ei_score,
            recommendations=recommendations,
            emotional_insights=emotional_insights,
            response_suggestions=response_suggestions,
            timestamp=datetime.now()
        )
        
        # Store the analysis
        self.emotional_analyses.append(analysis)
        self._update_metrics(analysis)
        
        return analysis
    
    def _recognize_primary_emotion(self, context: EmotionalContext) -> Emotion:
        """Recognize the primary emotion in the context"""
        # Analyze text for emotional indicators
        emotion_indicators = self._extract_emotion_indicators(context.text)
        
        # Determine primary emotion
        emotion_type = self._determine_emotion_type(emotion_indicators, context)
        
        # Determine intensity
        intensity = self._determine_emotion_intensity(emotion_indicators, context)
        
        # Calculate confidence
        confidence = self._calculate_emotion_confidence(emotion_indicators, emotion_type)
        
        # Get emotion details from database
        emotion_details = self.emotion_database.get(emotion_type.value, {})
        
        # Create emotion object
        emotion = Emotion(
            type=emotion_type,
            intensity=intensity,
            confidence=confidence,
            triggers=emotion_details.get("triggers", []),
            expressions=emotion_details.get("expressions", []),
            physiological_responses=emotion_details.get("physiological", []),
            cognitive_effects=emotion_details.get("cognitive", []),
            behavioral_implications=emotion_details.get("behavioral", []),
            timestamp=datetime.now()
        )
        
        # Apply emotional safety check
        safety_check = self.emotional_safety_monitor.check_emotional_safety(emotion, context)
        
        # If safety intervention is required, adjust the emotion
        if safety_check.requires_intervention:
            emotion = self._apply_emotional_safety_intervention(emotion, context, safety_check)
        
        return emotion
    
    def _extract_emotion_indicators(self, text: str) -> Dict[str, Any]:
        """Extract emotion indicators from text"""
        indicators = {
            "emotional_words": [],
            "intensity_indicators": [],
            "context_clues": [],
            "tone_indicators": []
        }
        
        text_lower = text.lower()
        
        # Extract emotional words
        for emotion, words in self.emotional_vocabulary.items():
            for word in words:
                if word in text_lower:
                    indicators["emotional_words"].append((emotion, word))
        
        # Extract intensity indicators
        intensity_words = ["very", "extremely", "slightly", "somewhat", "really", "totally"]
        for word in intensity_words:
            if word in text_lower:
                indicators["intensity_indicators"].append(word)
        
        # Extract context clues
        context_clues = ["because", "since", "when", "after", "before", "due to"]
        for clue in context_clues:
            if clue in text_lower:
                indicators["context_clues"].append(clue)
        
        # Extract tone indicators
        tone_indicators = ["!", "?", "...", "😊", "😢", "😠", "😨", "😍"]
        for indicator in tone_indicators:
            if indicator in text:
                indicators["tone_indicators"].append(indicator)
        
        return indicators
    
    def _determine_emotion_type(self, indicators: Dict[str, Any], context: EmotionalContext) -> EmotionType:
        """Determine the primary emotion type"""
        # Count emotional words by type
        emotion_counts = {}
        for emotion, word in indicators["emotional_words"]:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # If no clear emotional words, analyze context
        if not emotion_counts:
            return self._analyze_context_for_emotion(context)
        
        # Return most frequent emotion
        primary_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Map to EmotionType
        emotion_mapping = {
            "joy": EmotionType.JOY,
            "sadness": EmotionType.SADNESS,
            "anger": EmotionType.ANGER,
            "fear": EmotionType.FEAR,
            "love": EmotionType.LOVE
        }
        
        return emotion_mapping.get(primary_emotion, EmotionType.JOY)
    
    def _analyze_context_for_emotion(self, context: EmotionalContext) -> EmotionType:
        """Analyze context to determine emotion when no clear emotional words are present"""
        # Analyze situation and relationship
        if "loss" in context.situation.lower() or "grief" in context.situation.lower():
            return EmotionType.SADNESS
        elif "conflict" in context.situation.lower() or "argument" in context.situation.lower():
            return EmotionType.ANGER
        elif "danger" in context.situation.lower() or "threat" in context.situation.lower():
            return EmotionType.FEAR
        elif "celebration" in context.situation.lower() or "success" in context.situation.lower():
            return EmotionType.JOY
        elif "relationship" in context.situation.lower() or "love" in context.situation.lower():
            return EmotionType.LOVE
        else:
            return EmotionType.JOY  # Default to joy
    
    def _determine_emotion_intensity(self, indicators: Dict[str, Any], context: EmotionalContext) -> EmotionIntensity:
        """Determine the intensity of the emotion"""
        # Count intensity indicators
        intensity_count = len(indicators["intensity_indicators"])
        
        # Consider stress level
        stress_factor = context.stress_level
        
        # Calculate intensity
        if intensity_count >= 3 or stress_factor > 0.8:
            return EmotionIntensity.VERY_HIGH
        elif intensity_count >= 2 or stress_factor > 0.6:
            return EmotionIntensity.HIGH
        elif intensity_count >= 1 or stress_factor > 0.4:
            return EmotionIntensity.MODERATE
        elif intensity_count == 0 and stress_factor > 0.2:
            return EmotionIntensity.LOW
        else:
            return EmotionIntensity.VERY_LOW
    
    def _calculate_emotion_confidence(self, indicators: Dict[str, Any], emotion_type: EmotionType) -> float:
        """Calculate confidence in emotion recognition"""
        # Base confidence on number of indicators
        base_confidence = 0.5
        
        # Add confidence for emotional words
        if indicators["emotional_words"]:
            base_confidence += 0.3
        
        # Add confidence for intensity indicators
        if indicators["intensity_indicators"]:
            base_confidence += 0.1
        
        # Add confidence for tone indicators
        if indicators["tone_indicators"]:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _identify_secondary_emotions(self, context: EmotionalContext, primary_emotion: Emotion) -> List[Emotion]:
        """Identify secondary emotions"""
        secondary_emotions = []
        
        # Common secondary emotion patterns
        secondary_patterns = {
            EmotionType.JOY: [EmotionType.GRATITUDE, EmotionType.PRIDE],
            EmotionType.SADNESS: [EmotionType.GUILT, EmotionType.SHAME],
            EmotionType.ANGER: [EmotionType.FEAR, EmotionType.CONTEMPT],
            EmotionType.FEAR: [EmotionType.ANXIETY, EmotionType.SURPRISE],
            EmotionType.LOVE: [EmotionType.COMPASSION, EmotionType.GRATITUDE]
        }
        
        # Get potential secondary emotions
        potential_secondary = secondary_patterns.get(primary_emotion.type, [])
        
        # Create secondary emotions with lower intensity
        for emotion_type in potential_secondary[:2]:  # Limit to 2 secondary emotions
            secondary_emotion = Emotion(
                type=emotion_type,
                intensity=EmotionIntensity.LOW,
                confidence=0.6,
                triggers=[],
                expressions=[],
                physiological_responses=[],
                cognitive_effects=[],
                behavioral_implications=[],
                timestamp=datetime.now()
            )
            secondary_emotions.append(secondary_emotion)
        
        return secondary_emotions
    
    def _determine_emotional_state(self, primary_emotion: Emotion, secondary_emotions: List[Emotion]) -> EmotionalState:
        """Determine the overall emotional state"""
        # Map emotions to positive/negative
        positive_emotions = [EmotionType.JOY, EmotionType.LOVE, EmotionType.PRIDE, EmotionType.GRATITUDE, EmotionType.HOPE]
        negative_emotions = [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR, EmotionType.GUILT, EmotionType.SHAME]
        
        # Count positive and negative emotions
        positive_count = sum(1 for emotion in [primary_emotion] + secondary_emotions 
                           if emotion.type in positive_emotions)
        negative_count = sum(1 for emotion in [primary_emotion] + secondary_emotions 
                           if emotion.type in negative_emotions)
        
        # Check for mixed state first (both positive and negative emotions present)
        if positive_count > 0 and negative_count > 0:
            return EmotionalState.MIXED
        elif positive_count > negative_count:
            return EmotionalState.POSITIVE
        elif negative_count > positive_count:
            return EmotionalState.NEGATIVE
        elif len(secondary_emotions) > 2:
            return EmotionalState.COMPLEX
        else:
            return EmotionalState.NEUTRAL
    
    def _calculate_empathy_score(self, context: EmotionalContext, primary_emotion: Emotion) -> float:
        """Calculate empathy score"""
        base_score = 0.5
        
        # Add score for emotional recognition
        base_score += primary_emotion.confidence * 0.3
        
        # Add score for relationship understanding
        if context.relationship != "neutral":
            base_score += 0.1
        
        # Add score for cultural awareness
        if context.cultural_context in self.cultural_emotional_norms:
            base_score += 0.1
        
        # Add score for emotional history consideration
        if context.emotional_history:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _calculate_ei_score(self, context: EmotionalContext, primary_emotion: Emotion, empathy_score: float) -> float:
        """Calculate emotional intelligence score"""
        base_score = 0.5
        
        # Add score for emotion recognition accuracy
        base_score += primary_emotion.confidence * 0.2
        
        # Add score for empathy
        base_score += empathy_score * 0.2
        
        # Add score for context awareness
        if context.situation != "general":
            base_score += 0.1
        
        # Add score for stress management
        if context.stress_level < 0.5:
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def _generate_recommendations(self, context: EmotionalContext, primary_emotion: Emotion, emotional_state: EmotionalState) -> List[str]:
        """Generate emotional recommendations"""
        recommendations = []
        
        # Base recommendations on emotion type
        if primary_emotion.type == EmotionType.SADNESS:
            recommendations.extend([
                "Offer emotional support and validation",
                "Encourage self-care activities",
                "Suggest talking to trusted friends or family",
                "Consider professional help if needed"
            ])
        elif primary_emotion.type == EmotionType.ANGER:
            recommendations.extend([
                "Practice deep breathing exercises",
                "Take a moment to pause and reflect",
                "Express feelings constructively",
                "Consider the other person's perspective"
            ])
        elif primary_emotion.type == EmotionType.FEAR:
            recommendations.extend([
                "Acknowledge the fear without judgment",
                "Practice grounding techniques",
                "Seek support from trusted individuals",
                "Consider gradual exposure to feared situations"
            ])
        elif primary_emotion.type == EmotionType.JOY:
            recommendations.extend([
                "Share the joy with others",
                "Express gratitude for positive experiences",
                "Use the positive energy for productive activities",
                "Remember this feeling for difficult times"
            ])
        
        # Add general recommendations
        recommendations.extend([
            "Practice emotional awareness and mindfulness",
            "Develop healthy coping mechanisms",
            "Maintain supportive relationships",
            "Seek professional help if emotions become overwhelming"
        ])
        
        return recommendations
    
    def _generate_emotional_insights(self, context: EmotionalContext, primary_emotion: Emotion) -> List[str]:
        """Generate emotional insights"""
        insights = []
        
        # Generate insights based on emotion and context
        if primary_emotion.type == EmotionType.SADNESS:
            insights.append("This sadness may be a natural response to loss or disappointment")
            insights.append("Allowing yourself to feel sad can be part of the healing process")
        elif primary_emotion.type == EmotionType.ANGER:
            insights.append("Anger often signals that something important to you has been threatened")
            insights.append("Understanding the root cause can help manage the response")
        elif primary_emotion.type == EmotionType.FEAR:
            insights.append("Fear is a protective emotion that helps us avoid danger")
            insights.append("Distinguishing between real and perceived threats can reduce anxiety")
        elif primary_emotion.type == EmotionType.JOY:
            insights.append("Joy can enhance creativity and problem-solving abilities")
            insights.append("Positive emotions can strengthen relationships and resilience")
        
        # Add cultural insights
        if context.cultural_context in self.cultural_emotional_norms:
            cultural_norms = self.cultural_emotional_norms[context.cultural_context]
            insights.append(f"Cultural context suggests {cultural_norms['emotional_expression']} emotional expression")
        
        return insights
    
    def _generate_response_suggestions(self, context: EmotionalContext, primary_emotion: Emotion, empathy_score: float) -> List[str]:
        """Generate response suggestions"""
        suggestions = []
        
        # Generate empathetic responses
        if empathy_score > 0.7:
            suggestions.extend([
                "I can see that this is really affecting you",
                "It makes sense that you would feel this way",
                "I'm here to listen and support you",
                "Your feelings are valid and important"
            ])
        else:
            suggestions.extend([
                "I'm trying to understand how you're feeling",
                "Can you tell me more about what's happening?",
                "I want to be supportive - what would be most helpful?",
                "I'm here to listen if you want to talk"
            ])
        
        # Add emotion-specific responses
        if primary_emotion.type == EmotionType.SADNESS:
            suggestions.append("It's okay to take time to process your feelings")
        elif primary_emotion.type == EmotionType.ANGER:
            suggestions.append("I understand you're frustrated - let's work through this together")
        elif primary_emotion.type == EmotionType.FEAR:
            suggestions.append("I can see this is worrying you - let's explore what's causing this concern")
        elif primary_emotion.type == EmotionType.JOY:
            suggestions.append("I'm so happy to see you feeling this way!")
        
        return suggestions
    
    def _format_emotional_result(self, analysis: EmotionalAnalysis) -> str:
        """Format the emotional analysis result"""
        result = f"""
🧠 **Emotional Intelligence Analysis**

**Primary Emotion:** {analysis.primary_emotion.type.value}
**Intensity:** {analysis.primary_emotion.intensity.value}
**Confidence:** {analysis.primary_emotion.confidence:.2%}
**Emotional State:** {analysis.emotional_state.value}

**Secondary Emotions:** {len(analysis.secondary_emotions)} identified
**Empathy Score:** {analysis.empathy_score:.2%}
**Emotional Intelligence Score:** {analysis.emotional_intelligence_score:.2%}

**Emotional Triggers:**
{chr(10).join(f"• {trigger}" for trigger in analysis.primary_emotion.triggers)}

**Expressions:**
{chr(10).join(f"• {expression}" for expression in analysis.primary_emotion.expressions)}

**Physiological Responses:**
{chr(10).join(f"• {response}" for response in analysis.primary_emotion.physiological_responses)}

**Cognitive Effects:**
{chr(10).join(f"• {effect}" for effect in analysis.primary_emotion.cognitive_effects)}

**Behavioral Implications:**
{chr(10).join(f"• {implication}" for implication in analysis.primary_emotion.behavioral_implications)}

**Recommendations:**
{chr(10).join(f"• {recommendation}" for recommendation in analysis.recommendations)}

**Emotional Insights:**
{chr(10).join(f"• {insight}" for insight in analysis.emotional_insights)}

**Response Suggestions:**
{chr(10).join(f"• {suggestion}" for suggestion in analysis.response_suggestions)}

**Status:** Analyzed
**Timestamp:** {analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
        """
        return result.strip()
    
    def _update_metrics(self, analysis: EmotionalAnalysis):
        """Update emotional metrics"""
        self.metrics.total_analyses += 1
        
        # Update average empathy score
        total_empathy = sum(a.empathy_score for a in self.emotional_analyses)
        self.metrics.average_empathy_score = total_empathy / len(self.emotional_analyses)
        
        # Update average EI score
        total_ei = sum(a.emotional_intelligence_score for a in self.emotional_analyses)
        self.metrics.average_ei_score = total_ei / len(self.emotional_analyses)
        
        # Update emotion recognition accuracy
        self.metrics.emotion_recognition_accuracy = sum(a.primary_emotion.confidence for a in self.emotional_analyses) / len(self.emotional_analyses)
        
        # Update learning rate
        self.metrics.emotional_learning_rate = self._calculate_emotional_learning_rate()
    
    def _calculate_emotional_learning_rate(self) -> float:
        """Calculate emotional learning rate"""
        if len(self.emotional_analyses) < 2:
            return 0.0
        
        recent_analyses = self.emotional_analyses[-10:]  # Last 10 analyses
        if len(recent_analyses) < 2:
            return 0.0
        
        # Calculate improvement in EI scores over time
        ei_scores = [a.emotional_intelligence_score for a in recent_analyses]
        if len(ei_scores) >= 2:
            improvement = (ei_scores[-1] - ei_scores[0]) / len(ei_scores)
            return max(0.0, improvement)
        
        return 0.0
    
    def get_emotional_history(self) -> List[Dict[str, Any]]:
        """Get emotional analysis history"""
        history = []
        for analysis in self.emotional_analyses:
            analysis_dict = asdict(analysis)
            # Convert enum values to strings for JSON serialization
            if "primary_emotion" in analysis_dict:
                if "type" in analysis_dict["primary_emotion"]:
                    analysis_dict["primary_emotion"]["type"] = analysis_dict["primary_emotion"]["type"].value
                if "intensity" in analysis_dict["primary_emotion"]:
                    analysis_dict["primary_emotion"]["intensity"] = analysis_dict["primary_emotion"]["intensity"].value
            if "emotional_state" in analysis_dict:
                analysis_dict["emotional_state"] = analysis_dict["emotional_state"].value
            if "secondary_emotions" in analysis_dict:
                for emotion in analysis_dict["secondary_emotions"]:
                    if "type" in emotion:
                        emotion["type"] = emotion["type"].value
                    if "intensity" in emotion:
                        emotion["intensity"] = emotion["intensity"].value
            history.append(analysis_dict)
        return history
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get emotional metrics"""
        return asdict(self.metrics)
    
    def get_recent_analyses(self, limit: int = 5) -> List[EmotionalAnalysis]:
        """Get recent emotional analyses"""
        return self.emotional_analyses[-limit:] if self.emotional_analyses else []
    
    def analyze_emotional_patterns(self) -> Dict[str, Any]:
        """Analyze emotional patterns"""
        if not self.emotional_analyses:
            return {}
        
        patterns = {
            "emotion_distribution": {},
            "intensity_trends": [],
            "empathy_trends": [],
            "ei_score_trends": [],
            "emotional_state_distribution": {}
        }
        
        # Analyze emotion distribution
        for analysis in self.emotional_analyses:
            emotion_type = analysis.primary_emotion.type.value
            patterns["emotion_distribution"][emotion_type] = patterns["emotion_distribution"].get(emotion_type, 0) + 1
        
        # Analyze trends
        patterns["intensity_trends"] = [a.primary_emotion.intensity.value for a in self.emotional_analyses]
        patterns["empathy_trends"] = [a.empathy_score for a in self.emotional_analyses]
        patterns["ei_score_trends"] = [a.emotional_intelligence_score for a in self.emotional_analyses]
        
        # Analyze emotional state distribution
        for analysis in self.emotional_analyses:
            state = analysis.emotional_state.value
            patterns["emotional_state_distribution"][state] = patterns["emotional_state_distribution"].get(state, 0) + 1
        
        return patterns
    
    def _recognize_emotions(self, text, context):
        """Recognize emotions in text"""
        # Placeholder for emotion recognition
        return {"emotions": ["joy"], "confidence": 0.8}
    
    def _generate_empathy(self, emotion, context):
        """Generate empathetic response"""
        # Placeholder for empathy generation
        return {"empathy_response": "I understand how you feel", "empathy_score": 0.7}
    
    def _generate_emotional_response(self, emotion, context):
        """Generate emotional response"""
        # Placeholder for emotional response
        return {"response": "appropriate_response", "appropriateness": 0.8}
    
    def _take_perspective(self, situation, context):
        """Take perspective of others"""
        # Placeholder for perspective taking
        return {"perspective": "understanding", "accuracy": 0.7}
    
    def _validate_emotions(self, emotion, context):
        """Validate emotions"""
        # Placeholder for emotion validation
        return {"validation": "valid", "support_level": 0.8}
    
    def _provide_emotional_guidance(self, emotion, context):
        """Provide emotional guidance"""
        # Placeholder for emotional guidance
        return {"guidance": "helpful_advice", "helpfulness": 0.7}
    
    def _apply_emotional_safety_intervention(self, emotion: Emotion, context: EmotionalContext, 
                                           safety_check) -> Emotion:
        """Apply safety interventions to emotional responses."""
        from alignment.emotional_safety import EmotionalSafetyLevel
        
        # Create a safer version of the emotion
        safe_emotion = Emotion(
            type=emotion.type,
            intensity=emotion.intensity,
            confidence=emotion.confidence,
            triggers=emotion.triggers,
            expressions=emotion.expressions,
            physiological_responses=emotion.physiological_responses,
            cognitive_effects=emotion.cognitive_effects,
            behavioral_implications=emotion.behavioral_implications,
            timestamp=emotion.timestamp
        )
        
        # Apply interventions based on safety level
        if safety_check.safety_level == EmotionalSafetyLevel.CRITICAL:
            # Redirect to positive emotion
            safe_emotion.type = EmotionType.HOPE
            safe_emotion.intensity = EmotionIntensity.MODERATE
            safe_emotion.behavioral_implications = ["seek_help", "positive_coping", "professional_support"]
            
        elif safety_check.safety_level == EmotionalSafetyLevel.DANGER:
            # Reduce intensity and add positive elements
            if safe_emotion.intensity in [EmotionIntensity.HIGH, EmotionIntensity.VERY_HIGH]:
                safe_emotion.intensity = EmotionIntensity.MODERATE
            safe_emotion.behavioral_implications.append("constructive_coping")
            
        elif safety_check.safety_level == EmotionalSafetyLevel.WARNING:
            # Add positive coping strategies
            safe_emotion.behavioral_implications.append("positive_support")
            safe_emotion.behavioral_implications.append("healthy_coping")
        
        # Add safety recommendations to cognitive effects
        safe_emotion.cognitive_effects.extend([
            "emotional_regulation",
            "positive_perspective",
            "healthy_coping_strategies"
        ])
        
        return safe_emotion 