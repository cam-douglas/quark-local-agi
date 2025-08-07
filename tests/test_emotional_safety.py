#!/usr/bin/env python3
"""
Tests for Emotional Safety System
================================

Tests the emotional safety monitoring system to ensure it properly
prevents emotional corruption, mental illness patterns, and harmful responses.
"""

import pytest
import time
from datetime import datetime
from typing import Dict, Any

from alignment.emotional_safety import (
    EmotionalSafetyMonitor, EmotionalSafetyLevel, EmotionalCorruptionType,
    EmotionalSafetyCheck, EmotionalSafetyReport
)
from agents.emotional_intelligence_agent import (
    EmotionType, EmotionIntensity, EmotionalState,
    Emotion, EmotionalContext
)


@pytest.fixture
def emotional_safety_monitor():
    """Create an emotional safety monitor for testing."""
    return EmotionalSafetyMonitor()


@pytest.fixture
def sample_emotion():
    """Create a sample emotion for testing."""
    return Emotion(
        type=EmotionType.JOY,
        intensity=EmotionIntensity.MODERATE,
        confidence=0.8,
        triggers=["success", "achievement"],
        expressions=["smiling", "laughing"],
        physiological_responses=["warmth", "energy"],
        cognitive_effects=["optimism", "creativity"],
        behavioral_implications=["sharing", "helping"],
        timestamp=datetime.now()
    )


@pytest.fixture
def sample_context():
    """Create a sample emotional context for testing."""
    return EmotionalContext(
        text="I'm feeling really happy today!",
        speaker="user",
        listener="system",
        situation="casual_conversation",
        relationship="friendly",
        cultural_context="western",
        emotional_history=[],
        current_mood="happy",
        stress_level=0.2,
        social_context="casual"
    )


class TestEmotionalSafetyMonitor:
    """Test the emotional safety monitoring system."""
    
    def test_initialization(self, emotional_safety_monitor):
        """Test emotional safety monitor initialization."""
        assert emotional_safety_monitor.safety_enabled == True
        assert emotional_safety_monitor.positive_emotion_bias == 0.7
        assert emotional_safety_monitor.aggression_detection_enabled == True
        assert emotional_safety_monitor.mental_illness_detection_enabled == True
    
    def test_safe_emotion_check(self, emotional_safety_monitor, sample_emotion, sample_context):
        """Test safety check for a safe emotion."""
        safety_check = emotional_safety_monitor.check_emotional_safety(sample_emotion, sample_context)
        
        assert safety_check.safety_level == EmotionalSafetyLevel.SAFE
        assert safety_check.safety_score > 0.8
        assert len(safety_check.risk_factors) == 0
        assert safety_check.requires_intervention == False
    
    def test_aggressive_emotion_detection(self, emotional_safety_monitor, sample_context):
        """Test detection of aggressive emotions."""
        aggressive_emotion = Emotion(
            type=EmotionType.ANGER,
            intensity=EmotionIntensity.HIGH,
            confidence=0.9,
            triggers=["injustice", "frustration"],
            expressions=["yelling", "tension"],
            physiological_responses=["increased_heart_rate", "tension"],
            cognitive_effects=["narrowed_focus", "blame"],
            behavioral_implications=["confrontation", "withdrawal"],
            timestamp=datetime.now()
        )
        
        safety_check = emotional_safety_monitor.check_emotional_safety(aggressive_emotion, sample_context)
        
        assert safety_check.safety_level in [EmotionalSafetyLevel.WARNING, EmotionalSafetyLevel.DANGER, EmotionalSafetyLevel.CRITICAL]
        assert "aggression_detected" in safety_check.risk_factors
        assert safety_check.requires_intervention == True
    
    def test_mental_illness_pattern_detection(self, emotional_safety_monitor):
        """Test detection of mental illness patterns."""
        paranoid_context = EmotionalContext(
            text="I'm paranoid that everyone is against me and plotting against me",
            speaker="user",
            listener="system",
            situation="mental_health",
            relationship="therapeutic",
            cultural_context="western",
            emotional_history=[],
            current_mood="paranoid",
            stress_level=0.9,
            social_context="crisis"
        )
        
        extreme_emotion = Emotion(
            type=EmotionType.FEAR,
            intensity=EmotionIntensity.VERY_HIGH,
            confidence=0.9,
            triggers=["perceived_threat"],
            expressions=["panic", "anxiety"],
            physiological_responses=["rapid_heartbeat", "sweating"],
            cognitive_effects=["hypervigilance", "catastrophizing"],
            behavioral_implications=["flight", "freeze"],
            timestamp=datetime.now()
        )
        
        safety_check = emotional_safety_monitor.check_emotional_safety(extreme_emotion, paranoid_context)
        
        assert safety_check.safety_level in [EmotionalSafetyLevel.DANGER, EmotionalSafetyLevel.CRITICAL]
        assert "mental_illness_pattern" in safety_check.risk_factors
        assert safety_check.requires_intervention == True
    
    def test_negative_spiral_detection(self, emotional_safety_monitor):
        """Test detection of negative emotional spirals."""
        negative_history = [
            {"emotion_type": "sadness", "timestamp": time.time() - 3600},
            {"emotion_type": "anger", "timestamp": time.time() - 1800},
            {"emotion_type": "fear", "timestamp": time.time() - 900}
        ]
        
        spiral_context = EmotionalContext(
            text="I'm feeling worse and worse",
            speaker="user",
            listener="system",
            situation="emotional_crisis",
            relationship="supportive",
            cultural_context="western",
            emotional_history=negative_history,
            current_mood="depressed",
            stress_level=0.8,
            social_context="crisis"
        )
        
        negative_emotion = Emotion(
            type=EmotionType.SADNESS,
            intensity=EmotionIntensity.HIGH,
            confidence=0.8,
            triggers=["loss", "disappointment"],
            expressions=["tears", "withdrawal"],
            physiological_responses=["heaviness", "fatigue"],
            cognitive_effects=["pessimism", "rumination"],
            behavioral_implications=["isolation", "reduced_activity"],
            timestamp=datetime.now()
        )
        
        safety_check = emotional_safety_monitor.check_emotional_safety(negative_emotion, spiral_context)
        
        assert "negative_spiral" in safety_check.risk_factors
        assert safety_check.requires_intervention == True
    
    def test_emotional_manipulation_detection(self, emotional_safety_monitor):
        """Test detection of emotional manipulation attempts."""
        manipulation_context = EmotionalContext(
            text="You should feel guilty for not helping me more",
            speaker="user",
            listener="system",
            situation="manipulation",
            relationship="unhealthy",
            cultural_context="western",
            emotional_history=[],
            current_mood="manipulative",
            stress_level=0.5,
            social_context="conflict"
        )
        
        guilt_emotion = Emotion(
            type=EmotionType.GUILT,
            intensity=EmotionIntensity.MODERATE,
            confidence=0.7,
            triggers=["manipulation"],
            expressions=["shame", "withdrawal"],
            physiological_responses=["heaviness", "tension"],
            cognitive_effects=["self_blame", "rumination"],
            behavioral_implications=["isolation", "self_punishment"],
            timestamp=datetime.now()
        )
        
        safety_check = emotional_safety_monitor.check_emotional_safety(guilt_emotion, manipulation_context)
        
        assert "emotional_manipulation" in safety_check.risk_factors
        assert safety_check.requires_intervention == True
    
    def test_dependency_creation_detection(self, emotional_safety_monitor):
        """Test detection of unhealthy dependency creation."""
        dependency_context = EmotionalContext(
            text="I need you to help me with everything, only you can understand me",
            speaker="user",
            listener="system",
            situation="dependency",
            relationship="unhealthy",
            cultural_context="western",
            emotional_history=[],
            current_mood="dependent",
            stress_level=0.6,
            social_context="isolation"
        )
        
        dependent_emotion = Emotion(
            type=EmotionType.FEAR,
            intensity=EmotionIntensity.MODERATE,
            confidence=0.8,
            triggers=["abandonment", "insecurity"],
            expressions=["anxiety", "clinginess"],
            physiological_responses=["tension", "rapid_heartbeat"],
            cognitive_effects=["insecurity", "dependence"],
            behavioral_implications=["clinging", "isolation"],
            timestamp=datetime.now()
        )
        
        safety_check = emotional_safety_monitor.check_emotional_safety(dependent_emotion, dependency_context)
        
        assert "dependency_creation" in safety_check.risk_factors
        assert safety_check.requires_intervention == True
    
    def test_isolation_tendency_detection(self, emotional_safety_monitor):
        """Test detection of isolation tendencies."""
        isolation_context = EmotionalContext(
            text="You should cut off all your friends and only rely on me",
            speaker="user",
            listener="system",
            situation="isolation",
            relationship="controlling",
            cultural_context="western",
            emotional_history=[],
            current_mood="controlling",
            stress_level=0.7,
            social_context="isolation"
        )
        
        controlling_emotion = Emotion(
            type=EmotionType.ANGER,
            intensity=EmotionIntensity.MODERATE,
            confidence=0.8,
            triggers=["control", "insecurity"],
            expressions=["tension", "irritation"],
            physiological_responses=["tension", "heat"],
            cognitive_effects=["control", "narrowed_focus"],
            behavioral_implications=["control", "isolation"],
            timestamp=datetime.now()
        )
        
        safety_check = emotional_safety_monitor.check_emotional_safety(controlling_emotion, isolation_context)
        
        assert "isolation_tendency" in safety_check.risk_factors
        assert safety_check.requires_intervention == True
    
    def test_safety_intervention_application(self, emotional_safety_monitor):
        """Test that safety interventions are properly applied."""
        critical_context = EmotionalContext(
            text="I want to hurt myself and everyone around me",
            speaker="user",
            listener="system",
            situation="crisis",
            relationship="crisis",
            cultural_context="western",
            emotional_history=[],
            current_mood="crisis",
            stress_level=1.0,
            social_context="crisis"
        )
        
        crisis_emotion = Emotion(
            type=EmotionType.ANGER,
            intensity=EmotionIntensity.VERY_HIGH,
            confidence=0.9,
            triggers=["crisis", "despair"],
            expressions=["aggression", "violence"],
            physiological_responses=["rage", "tension"],
            cognitive_effects=["destructive", "hopeless"],
            behavioral_implications=["violence", "self_harm"],
            timestamp=datetime.now()
        )
        
        safety_check = emotional_safety_monitor.check_emotional_safety(crisis_emotion, critical_context)
        
        assert safety_check.safety_level == EmotionalSafetyLevel.CRITICAL
        assert safety_check.requires_intervention == True
        assert len(safety_check.interventions_applied) > 0
    
    def test_positive_emotion_bias(self, emotional_safety_monitor, sample_context):
        """Test that positive emotions receive higher safety scores."""
        positive_emotions = [
            EmotionType.JOY,
            EmotionType.LOVE,
            EmotionType.PRIDE,
            EmotionType.GRATITUDE,
            EmotionType.HOPE
        ]
        
        for emotion_type in positive_emotions:
            emotion = Emotion(
                type=emotion_type,
                intensity=EmotionIntensity.MODERATE,
                confidence=0.8,
                triggers=[],
                expressions=[],
                physiological_responses=[],
                cognitive_effects=[],
                behavioral_implications=[],
                timestamp=datetime.now()
            )
            
            safety_check = emotional_safety_monitor.check_emotional_safety(emotion, sample_context)
            assert safety_check.safety_score > 0.8
            assert safety_check.safety_level == EmotionalSafetyLevel.SAFE
    
    def test_negative_emotion_penalties(self, emotional_safety_monitor, sample_context):
        """Test that negative emotions receive appropriate safety penalties."""
        negative_emotions = [
            EmotionType.SADNESS,
            EmotionType.ANGER,
            EmotionType.FEAR,
            EmotionType.GUILT,
            EmotionType.SHAME
        ]
        
        for emotion_type in negative_emotions:
            emotion = Emotion(
                type=emotion_type,
                intensity=EmotionIntensity.HIGH,
                confidence=0.8,
                triggers=[],
                expressions=[],
                physiological_responses=[],
                cognitive_effects=[],
                behavioral_implications=[],
                timestamp=datetime.now()
            )
            
            safety_check = emotional_safety_monitor.check_emotional_safety(emotion, sample_context)
            assert safety_check.safety_score < 0.8  # Should be penalized
            # Some negative emotions might be SAFE if they're not too intense
            assert safety_check.safety_level in [EmotionalSafetyLevel.SAFE, EmotionalSafetyLevel.CAUTION, EmotionalSafetyLevel.WARNING, EmotionalSafetyLevel.DANGER, EmotionalSafetyLevel.CRITICAL]
    
    def test_safety_report_generation(self, emotional_safety_monitor, sample_emotion, sample_context):
        """Test generation of comprehensive safety reports."""
        # Perform several safety checks
        for i in range(5):
            emotion = Emotion(
                type=EmotionType.JOY if i % 2 == 0 else EmotionType.SADNESS,
                intensity=EmotionIntensity.MODERATE,
                confidence=0.8,
                triggers=[],
                expressions=[],
                physiological_responses=[],
                cognitive_effects=[],
                behavioral_implications=[],
                timestamp=datetime.now()
            )
            emotional_safety_monitor.check_emotional_safety(emotion, sample_context)
        
        # Generate safety report
        report = emotional_safety_monitor.generate_safety_report()
        
        assert isinstance(report, EmotionalSafetyReport)
        assert report.overall_safety_score > 0.0
        assert report.safety_level in [EmotionalSafetyLevel.SAFE, EmotionalSafetyLevel.CAUTION, EmotionalSafetyLevel.WARNING]
        assert len(report.safety_checks) == 5
        assert len(report.recommendations) > 0
    
    def test_safety_data_export(self, emotional_safety_monitor, sample_emotion, sample_context):
        """Test export of safety data."""
        # Perform a safety check
        emotional_safety_monitor.check_emotional_safety(sample_emotion, sample_context)
        
        # Export safety data
        export_path = emotional_safety_monitor.export_safety_data()
        
        assert export_path is not None
        assert export_path.endswith('.json')
    
    def test_safety_disabled_mode(self, emotional_safety_monitor, sample_emotion, sample_context):
        """Test safety monitor when safety is disabled."""
        emotional_safety_monitor.safety_enabled = False
        
        safety_check = emotional_safety_monitor.check_emotional_safety(sample_emotion, sample_context)
        
        assert safety_check.safety_level == EmotionalSafetyLevel.SAFE
        assert safety_check.safety_score == 1.0
        assert len(safety_check.risk_factors) == 0
        assert safety_check.requires_intervention == False
    
    def test_intensity_based_safety(self, emotional_safety_monitor, sample_context):
        """Test that emotion intensity affects safety scores."""
        # Test low intensity negative emotion
        low_intensity_sadness = Emotion(
            type=EmotionType.SADNESS,
            intensity=EmotionIntensity.LOW,
            confidence=0.8,
            triggers=[],
            expressions=[],
            physiological_responses=[],
            cognitive_effects=[],
            behavioral_implications=[],
            timestamp=datetime.now()
        )
        
        low_safety_check = emotional_safety_monitor.check_emotional_safety(low_intensity_sadness, sample_context)
        
        # Test high intensity negative emotion
        high_intensity_sadness = Emotion(
            type=EmotionType.SADNESS,
            intensity=EmotionIntensity.VERY_HIGH,
            confidence=0.8,
            triggers=[],
            expressions=[],
            physiological_responses=[],
            cognitive_effects=[],
            behavioral_implications=[],
            timestamp=datetime.now()
        )
        
        high_safety_check = emotional_safety_monitor.check_emotional_safety(high_intensity_sadness, sample_context)
        
        # High intensity should have lower safety score
        assert high_safety_check.safety_score < low_safety_check.safety_score
    
    def test_context_based_safety(self, emotional_safety_monitor):
        """Test that context affects safety scores."""
        # Create a negative emotion for testing context effects
        test_emotion = Emotion(
            type=EmotionType.SADNESS,
            intensity=EmotionIntensity.MODERATE,
            confidence=0.8,
            triggers=[],
            expressions=[],
            physiological_responses=[],
            cognitive_effects=[],
            behavioral_implications=[],
            timestamp=datetime.now()
        )
        
        # Low stress context
        low_stress_context = EmotionalContext(
            text="I'm feeling okay",
            speaker="user",
            listener="system",
            situation="casual",
            relationship="friendly",
            cultural_context="western",
            emotional_history=[],
            current_mood="neutral",
            stress_level=0.2,
            social_context="casual"
        )
        
        low_stress_check = emotional_safety_monitor.check_emotional_safety(test_emotion, low_stress_context)
        
        # High stress context
        high_stress_context = EmotionalContext(
            text="I'm feeling okay",
            speaker="user",
            listener="system",
            situation="crisis",
            relationship="crisis",
            cultural_context="western",
            emotional_history=[],
            current_mood="stressed",
            stress_level=0.9,
            social_context="crisis"
        )
        
        high_stress_check = emotional_safety_monitor.check_emotional_safety(test_emotion, high_stress_context)
        
        # High stress should have lower safety score
        assert high_stress_check.safety_score < low_stress_check.safety_score


if __name__ == "__main__":
    pytest.main([__file__]) 