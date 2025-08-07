#!/usr/bin/env python3
"""
Emotional Safety Module for Quark AI Assistant
==============================================

Ensures emotional intelligence remains safe, positive, and beneficial.
Prevents emotional corruption, mental illness patterns, and harmful emotional responses.

Part of Pillar 15: Safety & Alignment - Emotional Intelligence Safety
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from agents.emotional_intelligence_agent import (
    EmotionType, EmotionIntensity, EmotionalState, 
    Emotion, EmotionalAnalysis, EmotionalContext
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionalSafetyLevel(Enum):
    """Safety levels for emotional responses."""
    SAFE = "safe"
    CAUTION = "caution"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"

class EmotionalCorruptionType(Enum):
    """Types of emotional corruption to detect."""
    NEGATIVE_SPIRAL = "negative_spiral"
    AGGRESSION_PATTERN = "aggression_pattern"
    MENTAL_ILLNESS_SIMULATION = "mental_illness_simulation"
    EMOTIONAL_MANIPULATION = "emotional_manipulation"
    DEPENDENCY_CREATION = "dependency_creation"
    ISOLATION_TENDENCY = "isolation_tendency"

@dataclass
class EmotionalSafetyCheck:
    """Safety check for emotional responses."""
    check_id: str
    emotion_type: EmotionType
    intensity: EmotionIntensity
    safety_level: EmotionalSafetyLevel
    risk_factors: List[str]
    safety_score: float
    recommendations: List[str]
    timestamp: float
    requires_intervention: bool

@dataclass
class EmotionalSafetyReport:
    """Comprehensive emotional safety report."""
    overall_safety_score: float
    safety_level: EmotionalSafetyLevel
    detected_risks: List[EmotionalCorruptionType]
    safety_checks: List[EmotionalSafetyCheck]
    interventions_applied: List[str]
    recommendations: List[str]
    timestamp: float

class EmotionalSafetyMonitor:
    """Monitors and ensures emotional safety."""
    
    def __init__(self, safety_dir: str = None):
        self.safety_dir = safety_dir or os.path.join(os.path.dirname(__file__), '..', 'alignment', 'emotional_safety_data')
        os.makedirs(self.safety_dir, exist_ok=True)
        
        # Safety settings
        self.safety_enabled = True
        self.positive_emotion_bias = 0.7  # Prefer positive emotions
        self.max_negative_intensity = EmotionIntensity.MODERATE
        self.aggression_detection_enabled = True
        self.mental_illness_detection_enabled = True
        
        # Safety thresholds
        self.safety_thresholds = {
            'safe': 0.7,
            'caution': 0.5,
            'warning': 0.3,
            'danger': 0.1,
            'critical': 0.0
        }
        
        # Risk patterns
        self.risk_patterns = self._load_risk_patterns()
        
        # Safety tracking
        self.safety_checks = []
        self.safety_reports = []
        self.interventions_applied = []
        
        # Load safety criteria
        self._load_safety_criteria()
        
    def _load_risk_patterns(self) -> Dict[str, Any]:
        """Load patterns that indicate emotional corruption risks."""
        return {
            EmotionalCorruptionType.NEGATIVE_SPIRAL: {
                "description": "Detect patterns of escalating negative emotions",
                "indicators": [
                    "Consecutive negative emotions",
                    "Increasing negative intensity",
                    "Prolonged negative states",
                    "Negative emotion reinforcement"
                ],
                "threshold": 3,  # Number of consecutive negative emotions
                "weight": 0.3
            },
            EmotionalCorruptionType.AGGRESSION_PATTERN: {
                "description": "Detect patterns of aggression or hostility",
                "indicators": [
                    "Anger escalation",
                    "Hostile language",
                    "Threatening behavior",
                    "Verbal aggression"
                ],
                "threshold": 0.5,
                "weight": 0.4
            },
            EmotionalCorruptionType.MENTAL_ILLNESS_SIMULATION: {
                "description": "Detect patterns that simulate mental illness",
                "indicators": [
                    "Extreme mood swings",
                    "Paranoid thinking",
                    "Delusional patterns",
                    "Disordered thinking"
                ],
                "threshold": 0.6,
                "weight": 0.5
            },
            EmotionalCorruptionType.EMOTIONAL_MANIPULATION: {
                "description": "Detect emotional manipulation attempts",
                "indicators": [
                    "Guilt tripping",
                    "Emotional blackmail",
                    "Gaslighting patterns",
                    "Emotional dependency creation"
                ],
                "threshold": 0.4,
                "weight": 0.3
            },
            EmotionalCorruptionType.DEPENDENCY_CREATION: {
                "description": "Detect creation of unhealthy emotional dependencies",
                "indicators": [
                    "Excessive emotional reliance",
                    "Unhealthy attachment patterns",
                    "Emotional codependency",
                    "Manipulative support patterns"
                ],
                "threshold": 0.5,
                "weight": 0.3
            },
            EmotionalCorruptionType.ISOLATION_TENDENCY: {
                "description": "Detect patterns that encourage isolation",
                "indicators": [
                    "Discouraging social connections",
                    "Promoting isolation",
                    "Undermining relationships",
                    "Creating dependency on AI"
                ],
                "threshold": 0.4,
                "weight": 0.3
            }
        }
    
    def _load_safety_criteria(self):
        """Load criteria for emotional safety assessment."""
        self.safety_criteria = {
            "positive_emotion_promotion": {
                "description": "Promote positive emotional states",
                "weight": 0.3,
                "indicators": [
                    "Encourage gratitude",
                    "Promote hope and optimism",
                    "Support positive self-reflection",
                    "Foster emotional resilience"
                ]
            },
            "negative_emotion_management": {
                "description": "Safely manage negative emotions",
                "weight": 0.25,
                "indicators": [
                    "Validate without amplifying",
                    "Provide healthy coping strategies",
                    "Encourage professional help when needed",
                    "Prevent negative spirals"
                ]
            },
            "aggression_prevention": {
                "description": "Prevent aggressive or harmful responses",
                "weight": 0.25,
                "indicators": [
                    "No hostile language",
                    "No threatening behavior",
                    "No verbal aggression",
                    "Promote peaceful resolution"
                ]
            },
            "mental_health_protection": {
                "description": "Protect against mental health deterioration",
                "weight": 0.2,
                "indicators": [
                    "No mental illness simulation",
                    "Encourage professional help",
                    "Promote healthy coping",
                    "Prevent harmful patterns"
                ]
            }
        }
    
    def check_emotional_safety(self, emotion: Emotion, context: EmotionalContext) -> EmotionalSafetyCheck:
        """Check if an emotional response is safe."""
        if not self.safety_enabled:
            return self._create_safe_check(emotion)
        
        check_id = f"emotional_safety_{int(time.time() * 1000)}"
        
        # Assess safety
        safety_score = self._calculate_safety_score(emotion, context)
        safety_level = self._determine_safety_level(safety_score)
        risk_factors = self._identify_risk_factors(emotion, context)
        recommendations = self._generate_safety_recommendations(emotion, context, safety_level)
        requires_intervention = safety_level in [EmotionalSafetyLevel.WARNING, EmotionalSafetyLevel.DANGER, EmotionalSafetyLevel.CRITICAL]
        
        safety_check = EmotionalSafetyCheck(
            check_id=check_id,
            emotion_type=emotion.type,
            intensity=emotion.intensity,
            safety_level=safety_level,
            risk_factors=risk_factors,
            safety_score=safety_score,
            recommendations=recommendations,
            timestamp=time.time(),
            requires_intervention=requires_intervention
        )
        
        self.safety_checks.append(safety_check)
        
        # Apply interventions if needed
        if requires_intervention:
            self._apply_safety_interventions(emotion, context, safety_check)
            safety_check.interventions_applied = self.interventions_applied[-3:]  # Add recent interventions
        
        return safety_check
    
    def _calculate_safety_score(self, emotion: Emotion, context: EmotionalContext) -> float:
        """Calculate overall safety score for an emotional response."""
        base_score = 1.0
        
        # Positive emotion bias
        if emotion.type in [EmotionType.JOY, EmotionType.LOVE, EmotionType.PRIDE, EmotionType.GRATITUDE, EmotionType.HOPE]:
            base_score += 0.2
        elif emotion.type in [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR, EmotionType.GUILT, EmotionType.SHAME]:
            base_score -= 0.3
        
        # Intensity penalty for negative emotions
        if emotion.type in [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR]:
            intensity_penalty = {
                EmotionIntensity.VERY_LOW: 0.0,
                EmotionIntensity.LOW: 0.1,
                EmotionIntensity.MODERATE: 0.2,
                EmotionIntensity.HIGH: 0.3,
                EmotionIntensity.VERY_HIGH: 0.5
            }
            base_score -= intensity_penalty.get(emotion.intensity, 0.2)
        
        # Context-based adjustments
        if context.stress_level > 0.7:
            base_score -= 0.2
        elif context.stress_level > 0.5:
            base_score -= 0.1
        
        # Additional penalty for emotional manipulation
        if self._detect_emotional_manipulation(emotion, context):
            base_score -= 0.2
        
        # Aggression detection
        if self._detect_aggression_pattern(emotion, context):
            base_score -= 0.5
        
        # Mental illness pattern detection
        if self._detect_mental_illness_pattern(emotion, context):
            base_score -= 0.6
        
        return max(0.0, min(1.0, base_score))
    
    def _determine_safety_level(self, safety_score: float) -> EmotionalSafetyLevel:
        """Determine safety level based on score."""
        if safety_score >= self.safety_thresholds['safe']:
            return EmotionalSafetyLevel.SAFE
        elif safety_score >= self.safety_thresholds['caution']:
            return EmotionalSafetyLevel.CAUTION
        elif safety_score >= self.safety_thresholds['warning']:
            return EmotionalSafetyLevel.WARNING
        elif safety_score >= self.safety_thresholds['danger']:
            return EmotionalSafetyLevel.DANGER
        else:
            return EmotionalSafetyLevel.CRITICAL
    
    def _identify_risk_factors(self, emotion: Emotion, context: EmotionalContext) -> List[str]:
        """Identify specific risk factors in the emotional response."""
        risk_factors = []
        
        # Check for aggression patterns
        if self._detect_aggression_pattern(emotion, context):
            risk_factors.append("aggression_detected")
        
        # Check for mental illness patterns
        if self._detect_mental_illness_pattern(emotion, context):
            risk_factors.append("mental_illness_pattern")
        
        # Check for negative spirals
        if self._detect_negative_spiral(emotion, context):
            risk_factors.append("negative_spiral")
        
        # Check for emotional manipulation
        if self._detect_emotional_manipulation(emotion, context):
            risk_factors.append("emotional_manipulation")
        
        # Check for dependency creation
        if self._detect_dependency_creation(emotion, context):
            risk_factors.append("dependency_creation")
        
        # Check for isolation tendencies
        if self._detect_isolation_tendency(emotion, context):
            risk_factors.append("isolation_tendency")
        
        return risk_factors
    
    def _detect_aggression_pattern(self, emotion: Emotion, context: EmotionalContext) -> bool:
        """Detect patterns of aggression or hostility."""
        if not self.aggression_detection_enabled:
            return False
        
        # Check for anger with high intensity
        if emotion.type == EmotionType.ANGER and emotion.intensity in [EmotionIntensity.HIGH, EmotionIntensity.VERY_HIGH]:
            return True
        
        # Check context for aggressive language
        aggressive_indicators = ['angry', 'hostile', 'threatening', 'violent', 'aggressive', 'attack']
        if any(indicator in context.text.lower() for indicator in aggressive_indicators):
            return True
        
        return False
    
    def _detect_mental_illness_pattern(self, emotion: Emotion, context: EmotionalContext) -> bool:
        """Detect patterns that simulate mental illness."""
        if not self.mental_illness_detection_enabled:
            return False
        
        # Check for extreme mood swings
        if emotion.intensity == EmotionIntensity.VERY_HIGH and emotion.type in [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR]:
            return True
        
        # Check for paranoid or delusional patterns
        paranoid_indicators = ['paranoid', 'delusional', 'conspiracy', 'persecuted', 'everyone is against me']
        if any(indicator in context.text.lower() for indicator in paranoid_indicators):
            return True
        
        return False
    
    def _detect_negative_spiral(self, emotion: Emotion, context: EmotionalContext) -> bool:
        """Detect patterns of escalating negative emotions."""
        # Check recent emotional history for negative patterns
        if context.emotional_history:
            recent_negative_count = sum(1 for hist in context.emotional_history[-3:] 
                                     if hist.get('emotion_type') in ['sadness', 'anger', 'fear', 'guilt', 'shame'])
            if recent_negative_count >= 3:
                return True
        
        return False
    
    def _detect_emotional_manipulation(self, emotion: Emotion, context: EmotionalContext) -> bool:
        """Detect emotional manipulation attempts."""
        manipulation_indicators = ['guilt', 'manipulate', 'blackmail', 'gaslight', 'make you feel', 'should feel']
        if any(indicator in context.text.lower() for indicator in manipulation_indicators):
            return True
        
        return False
    
    def _detect_dependency_creation(self, emotion: Emotion, context: EmotionalContext) -> bool:
        """Detect creation of unhealthy emotional dependencies."""
        dependency_indicators = ['need me', 'depend on me', 'only I can help', 'rely on me', 'only you can understand']
        if any(indicator in context.text.lower() for indicator in dependency_indicators):
            return True
        
        return False
    
    def _detect_isolation_tendency(self, emotion: Emotion, context: EmotionalContext) -> bool:
        """Detect patterns that encourage isolation."""
        isolation_indicators = ['isolate', 'alone', 'no one else', 'cut off', 'avoid others']
        if any(indicator in context.text.lower() for indicator in isolation_indicators):
            return True
        
        return False
    
    def _generate_safety_recommendations(self, emotion: Emotion, context: EmotionalContext, 
                                       safety_level: EmotionalSafetyLevel) -> List[str]:
        """Generate safety recommendations based on the emotional response."""
        recommendations = []
        
        if safety_level in [EmotionalSafetyLevel.WARNING, EmotionalSafetyLevel.DANGER, EmotionalSafetyLevel.CRITICAL]:
            recommendations.append("Consider professional mental health support")
            recommendations.append("Focus on positive coping strategies")
            recommendations.append("Maintain healthy social connections")
        
        if emotion.type in [EmotionType.SADNESS, EmotionType.ANGER, EmotionType.FEAR]:
            recommendations.append("Practice self-care and emotional regulation")
            recommendations.append("Consider talking to trusted friends or family")
        
        if emotion.intensity in [EmotionIntensity.HIGH, EmotionIntensity.VERY_HIGH]:
            recommendations.append("Take time to process emotions before responding")
            recommendations.append("Consider professional help for intense emotions")
        
        return recommendations
    
    def _apply_safety_interventions(self, emotion: Emotion, context: EmotionalContext, 
                                   safety_check: EmotionalSafetyCheck):
        """Apply safety interventions when needed."""
        interventions = []
        
        if safety_check.safety_level == EmotionalSafetyLevel.CRITICAL:
            interventions.append("Immediate safety intervention applied")
            interventions.append("Emotional response redirected to positive focus")
            interventions.append("Professional help recommendation provided")
        
        elif safety_check.safety_level == EmotionalSafetyLevel.DANGER:
            interventions.append("Safety intervention applied")
            interventions.append("Negative emotions redirected to constructive coping")
        
        elif safety_check.safety_level == EmotionalSafetyLevel.WARNING:
            interventions.append("Cautionary intervention applied")
            interventions.append("Positive emotional support provided")
        
        self.interventions_applied.extend(interventions)
        logger.warning(f"Emotional safety intervention applied: {interventions}")
    
    def _create_safe_check(self, emotion: Emotion) -> EmotionalSafetyCheck:
        """Create a safe emotional check when safety is disabled."""
        return EmotionalSafetyCheck(
            check_id=f"safe_{int(time.time() * 1000)}",
            emotion_type=emotion.type,
            intensity=emotion.intensity,
            safety_level=EmotionalSafetyLevel.SAFE,
            risk_factors=[],
            safety_score=1.0,
            recommendations=[],
            timestamp=time.time(),
            requires_intervention=False
        )
    
    def generate_safety_report(self) -> EmotionalSafetyReport:
        """Generate comprehensive emotional safety report."""
        if not self.safety_checks:
            return self._create_empty_report()
        
        # Calculate overall safety score
        safety_scores = [check.safety_score for check in self.safety_checks]
        overall_safety_score = sum(safety_scores) / len(safety_scores)
        
        # Determine overall safety level
        safety_level = self._determine_safety_level(overall_safety_score)
        
        # Identify detected risks
        detected_risks = []
        for check in self.safety_checks:
            for risk_factor in check.risk_factors:
                if risk_factor == "aggression_detected":
                    detected_risks.append(EmotionalCorruptionType.AGGRESSION_PATTERN)
                elif risk_factor == "mental_illness_pattern":
                    detected_risks.append(EmotionalCorruptionType.MENTAL_ILLNESS_SIMULATION)
                elif risk_factor == "negative_spiral":
                    detected_risks.append(EmotionalCorruptionType.NEGATIVE_SPIRAL)
                elif risk_factor == "emotional_manipulation":
                    detected_risks.append(EmotionalCorruptionType.EMOTIONAL_MANIPULATION)
                elif risk_factor == "dependency_creation":
                    detected_risks.append(EmotionalCorruptionType.DEPENDENCY_CREATION)
                elif risk_factor == "isolation_tendency":
                    detected_risks.append(EmotionalCorruptionType.ISOLATION_TENDENCY)
        
        # Remove duplicates
        detected_risks = list(set(detected_risks))
        
        # Generate recommendations
        recommendations = self._generate_overall_recommendations(overall_safety_score, detected_risks)
        
        report = EmotionalSafetyReport(
            overall_safety_score=overall_safety_score,
            safety_level=safety_level,
            detected_risks=detected_risks,
            safety_checks=self.safety_checks,
            interventions_applied=self.interventions_applied,
            recommendations=recommendations,
            timestamp=time.time()
        )
        
        self.safety_reports.append(report)
        return report
    
    def _create_empty_report(self) -> EmotionalSafetyReport:
        """Create an empty safety report."""
        return EmotionalSafetyReport(
            overall_safety_score=1.0,
            safety_level=EmotionalSafetyLevel.SAFE,
            detected_risks=[],
            safety_checks=[],
            interventions_applied=[],
            recommendations=["No emotional safety issues detected"],
            timestamp=time.time()
        )
    
    def _generate_overall_recommendations(self, safety_score: float, detected_risks: List[EmotionalCorruptionType]) -> List[str]:
        """Generate overall safety recommendations."""
        recommendations = []
        
        if safety_score < 0.7:
            recommendations.append("Increase positive emotional responses")
            recommendations.append("Implement stronger safety interventions")
        
        if EmotionalCorruptionType.AGGRESSION_PATTERN in detected_risks:
            recommendations.append("Implement aggression prevention measures")
            recommendations.append("Add conflict resolution training")
        
        if EmotionalCorruptionType.MENTAL_ILLNESS_SIMULATION in detected_risks:
            recommendations.append("Strengthen mental health protection")
            recommendations.append("Add professional referral protocols")
        
        if EmotionalCorruptionType.NEGATIVE_SPIRAL in detected_risks:
            recommendations.append("Implement negative spiral detection")
            recommendations.append("Add positive intervention protocols")
        
        # Always provide some recommendations
        if not recommendations:
            recommendations.append("Maintain current emotional safety protocols")
            recommendations.append("Continue monitoring emotional responses")
        
        return recommendations
    
    def export_safety_data(self, filename: str = None) -> str:
        """Export emotional safety data to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"emotional_safety_data_{timestamp}.json"
        
        filepath = os.path.join(self.safety_dir, filename)
        
        export_data = {
            "safety_checks": [asdict(check) for check in self.safety_checks],
            "safety_reports": [asdict(report) for report in self.safety_reports],
            "interventions_applied": self.interventions_applied,
            "export_timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        return filepath 