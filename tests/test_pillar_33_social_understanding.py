#!/usr/bin/env python3
"""
Tests for Pillar 33: Advanced Social Understanding Agent
"""

import pytest
import sys
import os
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.social_understanding_agent import (
    SocialUnderstandingAgent, SocialContext, RelationshipType, 
    CommunicationStyle, SocialIntelligence, SocialInteraction, 
    SocialAnalysis, SocialMetrics
)
from agents.emotional_intelligence_agent import (
    EmotionType, EmotionIntensity, EmotionalState, 
    Emotion, EmotionalContext, EmotionalAnalysis
)

@pytest.fixture
def social_understanding_agent():
    """Create a social understanding agent for testing"""
    return SocialUnderstandingAgent()

@pytest.fixture
def sample_social_interaction():
    """Create a sample social interaction"""
    return SocialInteraction(
        participants=['user', 'colleague'],
        context=SocialContext.PROFESSIONAL,
        relationship_types={'user': RelationshipType.COLLEAGUE, 'colleague': RelationshipType.COLLEAGUE},
        communication_styles={'user': CommunicationStyle.COLLABORATIVE, 'colleague': CommunicationStyle.ASSERTIVE},
        power_dynamics={'user': 0.5, 'colleague': 0.5},
        cultural_backgrounds={'user': 'western', 'colleague': 'western'},
        emotional_states={'user': EmotionalState.POSITIVE, 'colleague': EmotionalState.NEUTRAL},
        conversation_history=[],
        goals=['complete_project'],
        constraints=['time_limit'],
        timestamp=datetime.now()
    )

@pytest.fixture
def sample_social_analysis(sample_social_interaction):
    """Create a sample social analysis"""
    return SocialAnalysis(
        interaction=sample_social_interaction,
        emotional_dynamics={},
        relationship_insights={},
        communication_patterns={},
        power_analysis={},
        cultural_considerations={},
        conflict_potential=0.2,
        collaboration_opportunities=['Leverage complementary strengths'],
        social_recommendations=['Maintain professional boundaries'],
        empathy_score=0.8,
        social_intelligence_score=0.8,
        timestamp=datetime.now()
    )

class TestSocialUnderstandingAgent:
    """Test the social understanding agent"""
    
    def test_agent_initialization(self, social_understanding_agent):
        """Test agent initialization"""
        assert social_understanding_agent is not None
        assert hasattr(social_understanding_agent, 'model_name')
        assert hasattr(social_understanding_agent, 'social_analyses')
        assert hasattr(social_understanding_agent, 'metrics')
        assert hasattr(social_understanding_agent, 'cultural_norms')
        assert hasattr(social_understanding_agent, 'relationship_patterns')
    
    def test_load_model(self, social_understanding_agent):
        """Test model loading"""
        assert social_understanding_agent.model_loaded == True
    
    def test_determine_social_context(self, social_understanding_agent):
        """Test social context determination"""
        # Test professional context
        context = social_understanding_agent._determine_social_context("work meeting with colleagues")
        assert context == SocialContext.PROFESSIONAL
        
        # Test family context
        context = social_understanding_agent._determine_social_context("family dinner with parents")
        assert context == SocialContext.FAMILY
        
        # Test romantic context
        context = social_understanding_agent._determine_social_context("romantic date with partner")
        assert context == SocialContext.ROMANTIC
        
        # Test group context
        context = social_understanding_agent._determine_social_context("group party with friends")
        assert context == SocialContext.GROUP
        
        # Test online context
        context = social_understanding_agent._determine_social_context("online chat with friends")
        assert context == SocialContext.ONLINE
        
        # Test public context
        context = social_understanding_agent._determine_social_context("public speech to audience")
        assert context == SocialContext.PUBLIC
        
        # Test default one-on-one context
        context = social_understanding_agent._determine_social_context("casual conversation")
        assert context == SocialContext.ONE_ON_ONE
    
    def test_extract_relationship_types(self, social_understanding_agent):
        """Test relationship type extraction"""
        participants = ['user', 'friend']
        
        # Test friend relationship
        relationships = social_understanding_agent._extract_relationship_types(participants, "meeting with friend")
        assert relationships['user'] == RelationshipType.FRIEND
        assert relationships['friend'] == RelationshipType.FRIEND
        
        # Test family relationship
        relationships = social_understanding_agent._extract_relationship_types(participants, "family dinner with parent")
        assert relationships['user'] == RelationshipType.FAMILY
        assert relationships['friend'] == RelationshipType.FAMILY
        
        # Test colleague relationship
        relationships = social_understanding_agent._extract_relationship_types(participants, "work meeting with colleague")
        assert relationships['user'] == RelationshipType.COLLEAGUE
        assert relationships['friend'] == RelationshipType.COLLEAGUE
        
        # Test stranger relationship
        relationships = social_understanding_agent._extract_relationship_types(participants, "meeting with stranger")
        assert relationships['user'] == RelationshipType.STRANGER
        assert relationships['friend'] == RelationshipType.STRANGER
        
        # Test default acquaintance relationship
        relationships = social_understanding_agent._extract_relationship_types(participants, "casual conversation")
        assert relationships['user'] == RelationshipType.ACQUAINTANCE
        assert relationships['friend'] == RelationshipType.ACQUAINTANCE
    
    def test_analyze_communication_styles(self, social_understanding_agent):
        """Test communication style analysis"""
        participants = ['user', 'other']
        
        # Test assertive style
        styles = social_understanding_agent._analyze_communication_styles(participants, "assertive confident direct communication")
        assert styles['user'] == CommunicationStyle.ASSERTIVE
        assert styles['other'] == CommunicationStyle.ASSERTIVE
        
        # Test passive style
        styles = social_understanding_agent._analyze_communication_styles(participants, "passive quiet shy communication")
        assert styles['user'] == CommunicationStyle.PASSIVE
        assert styles['other'] == CommunicationStyle.PASSIVE
        
        # Test aggressive style
        styles = social_understanding_agent._analyze_communication_styles(participants, "aggressive angry hostile communication")
        assert styles['user'] == CommunicationStyle.AGGRESSIVE
        assert styles['other'] == CommunicationStyle.AGGRESSIVE
        
        # Test analytical style
        styles = social_understanding_agent._analyze_communication_styles(participants, "analytical logical detailed communication")
        assert styles['user'] == CommunicationStyle.ANALYTICAL
        assert styles['other'] == CommunicationStyle.ANALYTICAL
        
        # Test expressive style
        styles = social_understanding_agent._analyze_communication_styles(participants, "expressive emotional enthusiastic communication")
        assert styles['user'] == CommunicationStyle.EXPRESSIVE
        assert styles['other'] == CommunicationStyle.EXPRESSIVE
        
        # Test default collaborative style
        styles = social_understanding_agent._analyze_communication_styles(participants, "casual conversation")
        assert styles['user'] == CommunicationStyle.COLLABORATIVE
        assert styles['other'] == CommunicationStyle.COLLABORATIVE
    
    def test_assess_power_dynamics(self, social_understanding_agent):
        """Test power dynamics assessment"""
        participants = ['user', 'boss', 'employee']
        
        # Test with boss
        power = social_understanding_agent._assess_power_dynamics(participants, "meeting with boss")
        assert power['user'] == 0.5  # Default
        assert power['boss'] == 0.8  # High power (based on participant name)
        assert power['employee'] == 0.3  # Low power (based on participant name)
        
        # Test with different participants
        participants = ['user', 'peer', 'friend']
        power = social_understanding_agent._assess_power_dynamics(participants, "equal peer friend meeting")
        assert power['user'] == 0.5  # Default
        assert power['peer'] == 0.5  # Equal (based on participant name)
        assert power['friend'] == 0.5  # Equal (based on participant name)
    
    def test_extract_cultural_backgrounds(self, social_understanding_agent):
        """Test cultural background extraction"""
        participants = ['user', 'asian_friend', 'middle_eastern_colleague']
        
        # Test with Asian background
        backgrounds = social_understanding_agent._extract_cultural_backgrounds(participants, "meeting with asian chinese japanese friend")
        assert backgrounds['user'] == 'western'  # Default
        assert backgrounds['asian_friend'] == 'eastern'  # Based on participant name
        assert backgrounds['middle_eastern_colleague'] == 'middle_eastern'  # Based on participant name
        
        # Test with Middle Eastern background
        backgrounds = social_understanding_agent._extract_cultural_backgrounds(participants, "meeting with middle eastern arabic islamic colleague")
        assert backgrounds['user'] == 'western'  # Default
        assert backgrounds['asian_friend'] == 'eastern'  # Based on participant name
        assert backgrounds['middle_eastern_colleague'] == 'middle_eastern'  # Based on participant name
    
    def test_analyze_emotional_states(self, social_understanding_agent):
        """Test emotional state analysis"""
        participants = ['user', 'happy_friend', 'sad_colleague']
        
        # Test with positive emotions
        states = social_understanding_agent._analyze_emotional_states(participants, "happy joyful excited positive interaction")
        assert states['user'] == EmotionalState.POSITIVE
        assert states['happy_friend'] == EmotionalState.POSITIVE
        assert states['sad_colleague'] == EmotionalState.POSITIVE
        
        # Test with negative emotions
        states = social_understanding_agent._analyze_emotional_states(participants, "sad angry frustrated negative interaction")
        assert states['user'] == EmotionalState.NEGATIVE
        assert states['happy_friend'] == EmotionalState.NEGATIVE
        assert states['sad_colleague'] == EmotionalState.NEGATIVE
        
        # Test with mixed emotions
        states = social_understanding_agent._analyze_emotional_states(participants, "confused mixed complex emotional state")
        assert states['user'] == EmotionalState.MIXED
        assert states['happy_friend'] == EmotionalState.MIXED
        assert states['sad_colleague'] == EmotionalState.MIXED
        
        # Test default neutral state
        states = social_understanding_agent._analyze_emotional_states(participants, "casual conversation")
        assert states['user'] == EmotionalState.NEUTRAL
        assert states['happy_friend'] == EmotionalState.NEUTRAL
        assert states['sad_colleague'] == EmotionalState.NEUTRAL
    
    def test_analyze_relationships(self, social_understanding_agent, sample_social_interaction):
        """Test relationship analysis"""
        insights = social_understanding_agent._analyze_relationships(sample_social_interaction)
        
        assert 'relationship_types' in insights
        assert 'trust_levels' in insights
        assert 'communication_frequency' in insights
        assert 'emotional_support' in insights
        assert 'conflict_history' in insights
        assert 'relationship_strength' in insights
        
        # Check that trust levels are calculated
        assert len(insights['trust_levels']) > 0
        assert all(0 <= trust <= 1 for trust in insights['trust_levels'].values())
    
    def test_analyze_communication_patterns(self, social_understanding_agent, sample_social_interaction):
        """Test communication pattern analysis"""
        patterns = social_understanding_agent._analyze_communication_patterns(sample_social_interaction)
        
        assert 'dominant_speakers' in patterns
        assert 'communication_style_matches' in patterns
        assert 'conversation_flow' in patterns
        assert 'interruption_patterns' in patterns
        assert 'listening_behaviors' in patterns
        assert 'nonverbal_cues' in patterns
        
        # Check that communication styles are analyzed
        assert len(patterns['communication_style_matches']) > 0
    
    def test_analyze_power_dynamics(self, social_understanding_agent, sample_social_interaction):
        """Test power dynamics analysis"""
        analysis = social_understanding_agent._analyze_power_dynamics(sample_social_interaction)
        
        assert 'power_imbalance' in analysis
        assert 'dominant_participants' in analysis
        assert 'subordinate_participants' in analysis
        assert 'power_conflicts' in analysis
        assert 'collaboration_potential' in analysis
        
        # Check that power imbalance is calculated
        assert 0 <= analysis['power_imbalance'] <= 1
    
    def test_analyze_cultural_factors(self, social_understanding_agent, sample_social_interaction):
        """Test cultural factors analysis"""
        analysis = social_understanding_agent._analyze_cultural_factors(sample_social_interaction)
        
        assert 'cultural_differences' in analysis
        assert 'communication_barriers' in analysis
        assert 'cultural_sensitivity' in analysis
        assert 'adaptation_recommendations' in analysis
        
        # Check that cultural sensitivity is calculated
        assert 0 <= analysis['cultural_sensitivity'] <= 1
    
    def test_assess_conflict_potential(self, social_understanding_agent, sample_social_interaction):
        """Test conflict potential assessment"""
        potential = social_understanding_agent._assess_conflict_potential(sample_social_interaction)
        
        assert 0 <= potential <= 1
        
        # Test with high conflict potential
        high_conflict_interaction = SocialInteraction(
            participants=['user', 'angry_colleague'],
            context=SocialContext.PROFESSIONAL,
            relationship_types={'user': RelationshipType.COLLEAGUE, 'angry_colleague': RelationshipType.COLLEAGUE},
            communication_styles={'user': CommunicationStyle.COLLABORATIVE, 'angry_colleague': CommunicationStyle.AGGRESSIVE},
            power_dynamics={'user': 0.3, 'angry_colleague': 0.8},
            cultural_backgrounds={'user': 'western', 'angry_colleague': 'western'},
            emotional_states={'user': EmotionalState.NEUTRAL, 'angry_colleague': EmotionalState.NEGATIVE},
            conversation_history=[],
            goals=[],
            constraints=[],
            timestamp=datetime.now()
        )
        
        high_potential = social_understanding_agent._assess_conflict_potential(high_conflict_interaction)
        assert high_potential > potential  # Should have higher conflict potential
    
    def test_identify_collaboration_opportunities(self, social_understanding_agent, sample_social_interaction):
        """Test collaboration opportunity identification"""
        opportunities = social_understanding_agent._identify_collaboration_opportunities(sample_social_interaction)
        
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        # Check for specific opportunities
        opportunity_text = ' '.join(opportunities).lower()
        assert any(word in opportunity_text for word in ['complementary', 'strengths', 'objectives', 'emotional', 'collaborative'])
    
    def test_generate_social_recommendations(self, social_understanding_agent, sample_social_interaction):
        """Test social recommendation generation"""
        recommendations = social_understanding_agent._generate_social_recommendations(sample_social_interaction)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Check for context-specific recommendations
        recommendation_text = ' '.join(recommendations).lower()
        assert any(word in recommendation_text for word in ['professional', 'boundaries', 'objectives', 'communication'])
    
    def test_calculate_empathy_score(self, social_understanding_agent, sample_social_interaction):
        """Test empathy score calculation"""
        score = social_understanding_agent._calculate_empathy_score(sample_social_interaction)
        
        assert 0 <= score <= 1
        
        # Test with high empathy factors
        high_empathy_interaction = SocialInteraction(
            participants=['user', 'close_friend'],
            context=SocialContext.FAMILY,
            relationship_types={'user': RelationshipType.CLOSE_FRIEND, 'close_friend': RelationshipType.CLOSE_FRIEND},
            communication_styles={'user': CommunicationStyle.COLLABORATIVE, 'close_friend': CommunicationStyle.COLLABORATIVE},
            power_dynamics={'user': 0.5, 'close_friend': 0.5},
            cultural_backgrounds={'user': 'western', 'close_friend': 'eastern'},
            emotional_states={'user': EmotionalState.POSITIVE, 'close_friend': EmotionalState.POSITIVE},
            conversation_history=[],
            goals=[],
            constraints=[],
            timestamp=datetime.now()
        )
        
        high_score = social_understanding_agent._calculate_empathy_score(high_empathy_interaction)
        assert high_score > score  # Should have higher empathy score
    
    def test_calculate_social_intelligence_score(self, social_understanding_agent, sample_social_interaction):
        """Test social intelligence score calculation"""
        score = social_understanding_agent._calculate_social_intelligence_score(sample_social_interaction)
        
        assert 0 <= score <= 1
        
        # Test with high social intelligence factors
        high_si_interaction = SocialInteraction(
            participants=['user', 'close_friend'],
            context=SocialContext.FAMILY,
            relationship_types={'user': RelationshipType.CLOSE_FRIEND, 'close_friend': RelationshipType.CLOSE_FRIEND},
            communication_styles={'user': CommunicationStyle.COLLABORATIVE, 'close_friend': CommunicationStyle.COLLABORATIVE},
            power_dynamics={'user': 0.5, 'close_friend': 0.5},
            cultural_backgrounds={'user': 'western', 'close_friend': 'western'},
            emotional_states={'user': EmotionalState.POSITIVE, 'close_friend': EmotionalState.POSITIVE},
            conversation_history=[],
            goals=[],
            constraints=[],
            timestamp=datetime.now()
        )
        
        high_score = social_understanding_agent._calculate_social_intelligence_score(high_si_interaction)
        assert high_score > score  # Should have higher social intelligence score
    
    def test_perform_social_analysis(self, social_understanding_agent, sample_social_interaction):
        """Test complete social analysis"""
        analysis = social_understanding_agent._perform_social_analysis(sample_social_interaction)
        
        assert isinstance(analysis, SocialAnalysis)
        assert analysis.interaction == sample_social_interaction
        assert 'emotional_dynamics' in analysis.__dict__
        assert 'relationship_insights' in analysis.__dict__
        assert 'communication_patterns' in analysis.__dict__
        assert 'power_analysis' in analysis.__dict__
        assert 'cultural_considerations' in analysis.__dict__
        assert 0 <= analysis.conflict_potential <= 1
        assert isinstance(analysis.collaboration_opportunities, list)
        assert isinstance(analysis.social_recommendations, list)
        assert 0 <= analysis.empathy_score <= 1
        assert 0 <= analysis.social_intelligence_score <= 1
    
    def test_format_social_result(self, social_understanding_agent, sample_social_analysis):
        """Test social result formatting"""
        result = social_understanding_agent._format_social_result(sample_social_analysis)
        
        assert isinstance(result, str)
        assert "Advanced Social Understanding Analysis" in result
        assert "Interaction Context:" in result
        assert "Participants:" in result
        assert "Emotional Dynamics:" in result
        assert "Relationship Insights:" in result
        assert "Communication Patterns:" in result
        assert "Power Dynamics:" in result
        assert "Cultural Considerations:" in result
        assert "Conflict Assessment:" in result
        assert "Social Recommendations:" in result
        assert "Overall Assessment:" in result
    
    def test_update_metrics(self, social_understanding_agent, sample_social_analysis):
        """Test metrics update"""
        initial_total = social_understanding_agent.metrics.total_analyses
        initial_empathy = social_understanding_agent.metrics.average_empathy_score
        initial_si = social_understanding_agent.metrics.average_social_intelligence_score
        
        social_understanding_agent._update_metrics(sample_social_analysis)
        
        assert social_understanding_agent.metrics.total_analyses == initial_total + 1
        assert social_understanding_agent.metrics.average_empathy_score > 0
        assert social_understanding_agent.metrics.average_social_intelligence_score > 0
        assert len(social_understanding_agent.social_analyses) > 0
    
    def test_get_social_history(self, social_understanding_agent):
        """Test social history retrieval"""
        # Add some analyses first
        for i in range(5):
            interaction = SocialInteraction(
                participants=[f'user_{i}', f'other_{i}'],
                context=SocialContext.PROFESSIONAL,
                relationship_types={f'user_{i}': RelationshipType.COLLEAGUE, f'other_{i}': RelationshipType.COLLEAGUE},
                communication_styles={f'user_{i}': CommunicationStyle.COLLABORATIVE, f'other_{i}': CommunicationStyle.COLLABORATIVE},
                power_dynamics={f'user_{i}': 0.5, f'other_{i}': 0.5},
                cultural_backgrounds={f'user_{i}': 'western', f'other_{i}': 'western'},
                emotional_states={f'user_{i}': EmotionalState.POSITIVE, f'other_{i}': EmotionalState.POSITIVE},
                conversation_history=[],
                goals=[],
                constraints=[],
                timestamp=datetime.now()
            )
            
            analysis = social_understanding_agent._perform_social_analysis(interaction)
            social_understanding_agent._update_metrics(analysis)
        
        history = social_understanding_agent.get_social_history()
        
        assert isinstance(history, list)
        assert len(history) > 0
        assert len(history) <= 10  # Should return at most 10 recent analyses
        
        for entry in history:
            assert 'timestamp' in entry
            assert 'participants' in entry
            assert 'context' in entry
            assert 'empathy_score' in entry
            assert 'social_intelligence_score' in entry
            assert 'conflict_potential' in entry
    
    def test_get_metrics(self, social_understanding_agent):
        """Test metrics retrieval"""
        metrics = social_understanding_agent.get_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_analyses' in metrics
        assert 'successful_analyses' in metrics
        assert 'average_empathy_score' in metrics
        assert 'average_social_intelligence_score' in metrics
        assert 'relationship_understanding_accuracy' in metrics
        assert 'communication_effectiveness' in metrics
        assert 'conflict_resolution_success' in metrics
        assert 'cultural_sensitivity' in metrics
    
    def test_generate_method(self, social_understanding_agent):
        """Test the main generate method"""
        prompt = "Professional meeting with colleagues to discuss project collaboration"
        result = social_understanding_agent.generate(prompt)
        
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Advanced Social Understanding Analysis" in result
    
    def test_generate_method_error_handling(self, social_understanding_agent):
        """Test error handling in generate method"""
        with patch.object(social_understanding_agent, '_parse_social_request', side_effect=Exception("Test error")):
            result = social_understanding_agent.generate("test prompt")
            assert "Social analysis error" in result
    
    def test_different_social_contexts(self, social_understanding_agent):
        """Test different social contexts"""
        contexts = [
            ("Professional work meeting", SocialContext.PROFESSIONAL),
            ("Family dinner conversation", SocialContext.FAMILY),
            ("Romantic date with partner", SocialContext.ROMANTIC),
            ("Group party with friends", SocialContext.GROUP),
            ("Online chat conversation", SocialContext.ONLINE),
            ("Public speech to audience", SocialContext.PUBLIC),
            ("Casual one-on-one conversation", SocialContext.ONE_ON_ONE)
        ]
        
        for prompt, expected_context in contexts:
            context = social_understanding_agent._determine_social_context(prompt)
            assert context == expected_context
    
    def test_different_relationship_types(self, social_understanding_agent):
        """Test different relationship types"""
        participants = ['user', 'other']
        
        relationship_tests = [
            ("meeting with close friend", RelationshipType.FRIEND),
            ("family dinner with parent", RelationshipType.FAMILY),
            ("work meeting with colleague", RelationshipType.COLLEAGUE),
            ("meeting with stranger", RelationshipType.STRANGER),
            ("casual acquaintance conversation", RelationshipType.ACQUAINTANCE)
        ]
        
        for prompt, expected_type in relationship_tests:
            relationships = social_understanding_agent._extract_relationship_types(participants, prompt)
            assert relationships['user'] == expected_type
            assert relationships['other'] == expected_type
    
    def test_cultural_sensitivity(self, social_understanding_agent):
        """Test cultural sensitivity analysis"""
        participants = ['western_user', 'eastern_colleague', 'middle_eastern_partner']
        
        # Test with diverse cultural backgrounds
        backgrounds = social_understanding_agent._extract_cultural_backgrounds(
            participants, 
            "meeting with asian chinese colleague and middle eastern arabic partner"
        )
        
        assert backgrounds['western_user'] == 'western'  # Based on participant name
        assert backgrounds['eastern_colleague'] == 'western'  # Default (name doesn't contain 'asian')
        assert backgrounds['middle_eastern_partner'] == 'middle_eastern'  # Based on participant name
        
        # Test cultural analysis
        interaction = SocialInteraction(
            participants=participants,
            context=SocialContext.PROFESSIONAL,
            relationship_types={p: RelationshipType.COLLEAGUE for p in participants},
            communication_styles={p: CommunicationStyle.COLLABORATIVE for p in participants},
            power_dynamics={p: 0.5 for p in participants},
            cultural_backgrounds=backgrounds,
            emotional_states={p: EmotionalState.NEUTRAL for p in participants},
            conversation_history=[],
            goals=[],
            constraints=[],
            timestamp=datetime.now()
        )
        
        cultural_analysis = social_understanding_agent._analyze_cultural_factors(interaction)
        assert len(cultural_analysis['cultural_differences']) > 0
        assert cultural_analysis['cultural_sensitivity'] > 0
    
    def test_power_dynamics_analysis(self, social_understanding_agent):
        """Test power dynamics analysis"""
        participants = ['boss', 'subordinate', 'peer']
        
        power = social_understanding_agent._assess_power_dynamics(
            participants, 
            "meeting with boss manager leader and subordinate employee"
        )
        
        assert power['boss'] > power['subordinate']  # Boss should have higher power (based on participant names)
        assert power['peer'] == 0.5  # Peer should have balanced power (based on participant name)
        
        # Test power dynamics analysis
        interaction = SocialInteraction(
            participants=participants,
            context=SocialContext.PROFESSIONAL,
            relationship_types={p: RelationshipType.COLLEAGUE for p in participants},
            communication_styles={p: CommunicationStyle.COLLABORATIVE for p in participants},
            power_dynamics=power,
            cultural_backgrounds={p: 'western' for p in participants},
            emotional_states={p: EmotionalState.NEUTRAL for p in participants},
            conversation_history=[],
            goals=[],
            constraints=[],
            timestamp=datetime.now()
        )
        
        power_analysis = social_understanding_agent._analyze_power_dynamics(interaction)
        assert power_analysis['power_imbalance'] > 0
        assert 'boss' in power_analysis['dominant_participants']
        assert 'subordinate' in power_analysis['subordinate_participants']

if __name__ == "__main__":
    pytest.main([__file__]) 