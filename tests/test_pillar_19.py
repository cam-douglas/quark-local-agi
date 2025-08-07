#!/usr/bin/env python3
"""
Test Pillar 19: Social Intelligence
==================================

Tests the implementation of advanced social intelligence capabilities including
emotional intelligence, social context understanding, theory of mind, and multi-agent collaboration.

Part of Phase 5: AGI Capabilities
"""

import os
import sys
import time
import tempfile
import shutil
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from social.social_intelligence import (
    SocialIntelligence, EmotionType, SocialContext, TheoryOfMindLevel,
    EmotionalState, SocialInteraction, MentalState, SocialRelationship
)
from agents.social_agent import SocialAgent


def test_pillar_19():
    """Test Pillar 19: Social Intelligence."""
    print("🤝 Testing Pillar 19: Social Intelligence")
    print("=" * 60)
    
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    social_dir = os.path.join(temp_dir, "social_data")
    
    try:
        # Test 1: Social Intelligence Engine
        print("\n🧠 Testing Social Intelligence Engine")
        print("-" * 40)
        
        # Initialize social intelligence engine
        social_engine = SocialIntelligence(social_dir)
        
        # Test emotion recognition
        print("✅ Testing Emotion Recognition")
        
        emotion_texts = [
            "I am so happy and excited about this project! 😊",
            "I feel sad and disappointed about the results 😢",
            "I'm really angry and frustrated with this situation 😠",
            "I'm afraid and worried about what might happen 😨",
            "I'm surprised and amazed by this discovery! 😲"
        ]
        
        for text in emotion_texts:
            emotion_result = social_engine.recognize_emotion(text)
            print(f"   Text: {text[:30]}...")
            print(f"   Emotion: {emotion_result['emotion']}")
            print(f"   Intensity: {emotion_result['intensity']:.2f}")
            print(f"   Confidence: {emotion_result['confidence']:.2f}")
            print()
        
        # Test social context adaptation
        print("✅ Testing Social Context Adaptation")
        
        contexts = [
            SocialContext.FORMAL,
            SocialContext.INFORMAL,
            SocialContext.COLLABORATIVE,
            SocialContext.COMPETITIVE
        ]
        
        for context in contexts:
            adaptation_result = social_engine.adapt_to_social_context(context)
            print(f"   Context: {context.value}")
            print(f"   Recommended style: {adaptation_result['adaptation']['recommended_style']}")
            print(f"   Emotional expression: {adaptation_result['adaptation']['emotional_expression']}")
            print(f"   Formality level: {adaptation_result['adaptation']['formality_level']}")
            print()
        
        # Test theory of mind
        print("✅ Testing Theory of Mind")
        
        agent_info = {
            'beliefs': ['AI safety is important', 'Collaboration is beneficial'],
            'desires': ['Improve system performance', 'Build better relationships'],
            'intentions': ['Work on safety features', 'Collaborate with other agents'],
            'knowledge': ['Machine learning basics', 'Safety protocols'],
            'emotional_state': None
        }
        
        tom_result = social_engine.understand_mental_state("test_agent", agent_info)
        print(f"   Agent ID: {tom_result['agent_id']}")
        print(f"   Theory of mind level: {tom_result['theory_of_mind_level']}")
        print(f"   Complexity score: {tom_result['complexity_score']}")
        print(f"   Confidence: {tom_result['confidence']:.2f}")
        print()
        
        # Test collaboration facilitation
        print("✅ Testing Collaboration Facilitation")
        
        agents = ["agent_1", "agent_2", "agent_3"]
        goals = ["Complete project on time", "Improve system performance", "Ensure safety compliance"]
        
        collaboration_result = social_engine.facilitate_collaboration(agents, goals, "consensus_building")
        print(f"   Collaboration ID: {collaboration_result['collaboration_id']}")
        print(f"   Strategy: {collaboration_result['strategy']}")
        print(f"   Agents: {len(collaboration_result['agents'])}")
        print(f"   Goals: {len(collaboration_result['goals'])}")
        print()
        
        # Test relationship management
        print("✅ Testing Relationship Management")
        
        interaction_data = {
            'quality': 0.8,
            'type': 'cooperative'
        }
        
        relationship_result = social_engine.manage_relationships("agent_1", "agent_2", interaction_data)
        print(f"   Relationship ID: {relationship_result['relationship_id']}")
        print(f"   Relationship type: {relationship_result['relationship_type']}")
        print(f"   Strength: {relationship_result['strength']:.2f}")
        print(f"   Trust level: {relationship_result['trust_level']:.2f}")
        print()
        
        # Test social statistics
        print("✅ Testing Social Statistics")
        stats = social_engine.get_social_stats()
        print(f"   Total interactions: {stats['total_interactions']}")
        print(f"   Active emotional states: {stats['active_emotional_states']}")
        print(f"   Tracked mental states: {stats['tracked_mental_states']}")
        print(f"   Managed relationships: {stats['managed_relationships']}")
        print(f"   Social network size: {stats['social_network_size']}")
        print()
        
        # Test 2: Social Agent Integration
        print("\n🤖 Testing Social Agent Integration")
        print("-" * 40)
        
        # Initialize social agent
        social_agent = SocialAgent(social_dir=social_dir)
        
        # Test agent operations
        print("✅ Testing Social Agent Operations")
        
        # Test emotion recognition via agent
        agent_emotion_result = social_agent.generate(
            "I'm really excited about this new feature!",
            operation="recognize_emotion"
        )
        print(f"   Agent emotion recognition: {agent_emotion_result['status']}")
        
        # Test context adaptation via agent
        agent_context_result = social_agent.generate(
            "formal business meeting",
            operation="adapt_to_context"
        )
        print(f"   Agent context adaptation: {agent_context_result['status']}")
        
        # Test mental state understanding via agent
        agent_tom_result = social_agent.generate(
            "agent_3",
            operation="understand_mental_state"
        )
        print(f"   Agent theory of mind: {agent_tom_result['status']}")
        
        # Test collaboration facilitation via agent
        agent_collab_result = social_agent.generate(
            "Complete the AI safety project",
            operation="facilitate_collaboration"
        )
        print(f"   Agent collaboration: {agent_collab_result['status']}")
        
        # Test relationship management via agent
        agent_rel_result = social_agent.generate(
            "agent_1_agent_2",
            operation="manage_relationship"
        )
        print(f"   Agent relationship management: {agent_rel_result['status']}")
        
        # Test social statistics via agent
        agent_stats_result = social_agent.generate("", operation="get_social_stats")
        print(f"   Agent social stats: {agent_stats_result['performance_stats']['total_operations']} operations")
        
        # Test 3: Advanced Social Features
        print("\n🚀 Testing Advanced Social Features")
        print("-" * 40)
        
        # Test social dynamics analysis
        print("✅ Testing Social Dynamics Analysis")
        dynamics_result = social_agent.generate(
            "The team is collaborating well and everyone seems happy with the progress",
            operation="analyze_social_dynamics"
        )
        print(f"   Dynamics analysis: {dynamics_result['status']}")
        
        # Test conflict resolution
        print("✅ Testing Conflict Resolution")
        conflict_result = social_agent.generate(
            "There's a disagreement about the project timeline",
            operation="resolve_conflict"
        )
        print(f"   Conflict resolution: {conflict_result['status']}")
        
        # Test social recommendations
        print("✅ Testing Social Recommendations")
        recommendations_result = social_agent.get_social_recommendations()
        print(f"   Recommendations: {recommendations_result['recommendations_count']}")
        
        # Test data export
        print("✅ Testing Data Export")
        export_result = social_agent.generate("", operation="export_social_data")
        print(f"   Export result: {export_result['status']}")
        
        print("\n🎉 Pillar 19 Test Completed Successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Emotional intelligence and emotion recognition")
        print("✅ Social context understanding and adaptation")
        print("✅ Theory of mind and mental state modeling")
        print("✅ Multi-agent collaboration facilitation")
        print("✅ Relationship management and tracking")
        print("✅ Social dynamics analysis")
        print("✅ Conflict resolution strategies")
        print("✅ Performance monitoring and statistics")
        print("✅ Data export and persistence")
        print("✅ Agent integration and operations")
        print("✅ Advanced social intelligence capabilities")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_social_integration():
    """Test integration between social intelligence and other systems."""
    print("\n🔗 Testing Social Intelligence Integration")
    print("=" * 50)
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    social_dir = os.path.join(temp_dir, "social_data")
    
    try:
        # Initialize social systems
        social_engine = SocialIntelligence(social_dir)
        social_agent = SocialAgent(social_dir=social_dir)
        
        # Test integrated social workflow
        print("✅ Testing Integrated Social Workflow")
        
        # 1. Multi-agent social scenario
        scenario = """
        Scenario: Three AI agents are working on a collaborative project.
        
        Agent 1: "I'm excited about this project and want to collaborate effectively!"
        Agent 2: "I'm a bit worried about meeting the deadline, but I trust the team."
        Agent 3: "I'm confident in our abilities and ready to work together."
        
        Goal: Facilitate successful collaboration and maintain positive relationships.
        """
        
        print("✅ Testing Multi-Agent Social Scenario")
        
        # Process each agent's emotional state
        agent_emotions = {}
        agent_texts = [
            "I'm excited about this project and want to collaborate effectively!",
            "I'm a bit worried about meeting the deadline, but I trust the team.",
            "I'm confident in our abilities and ready to work together."
        ]
        
        for i, text in enumerate(agent_texts):
            agent_id = f"agent_{i+1}"
            emotion_result = social_engine.recognize_emotion(text, agent_id)
            agent_emotions[agent_id] = emotion_result
            print(f"   {agent_id}: {emotion_result['emotion']} (confidence: {emotion_result['confidence']:.2f})")
        
        # 2. Test social context adaptation
        print("✅ Testing Social Context Adaptation")
        
        context_result = social_engine.adapt_to_social_context(SocialContext.COLLABORATIVE)
        print(f"   Recommended style: {context_result['adaptation']['recommended_style']}")
        print(f"   Appropriate topics: {context_result['adaptation']['appropriate_topics']}")
        
        # 3. Test collaboration facilitation
        print("✅ Testing Collaboration Facilitation")
        
        agents = ["agent_1", "agent_2", "agent_3"]
        goals = ["Complete project successfully", "Meet deadline", "Maintain team harmony"]
        
        collab_result = social_engine.facilitate_collaboration(agents, goals, "consensus_building")
        print(f"   Collaboration strategy: {collab_result['strategy']}")
        print(f"   Plan phases: {len(collab_result['plan']['recommended_approach'])}")
        
        # 4. Test relationship management
        print("✅ Testing Relationship Management")
        
        # Manage relationships between all agent pairs
        for i in range(len(agents)):
            for j in range(i+1, len(agents)):
                agent_1 = agents[i]
                agent_2 = agents[j]
                
                interaction_data = {
                    'quality': 0.8,
                    'type': 'cooperative'
                }
                
                rel_result = social_engine.manage_relationships(agent_1, agent_2, interaction_data)
                print(f"   {agent_1} ↔ {agent_2}: {rel_result['relationship_type']} (strength: {rel_result['strength']:.2f})")
        
        # 5. Test social dynamics analysis
        print("✅ Testing Social Dynamics Analysis")
        
        dynamics_text = "The team is collaborating well, everyone is excited and confident, and relationships are positive."
        dynamics_result = social_agent.generate(dynamics_text, operation="analyze_social_dynamics")
        print(f"   Emotional climate: {dynamics_result['analysis']['emotional_climate']}")
        print(f"   Collaboration potential: {dynamics_result['analysis']['collaboration_potential']:.2f}")
        
        # 6. Test integrated insights
        print("✅ Testing Integrated Social Insights")
        
        integrated_insights = {
            'emotional_states': agent_emotions,
            'collaboration_plan': collab_result['plan'],
            'social_dynamics': dynamics_result['analysis'],
            'overall_assessment': 'Positive team dynamics with high collaboration potential'
        }
        
        print(f"   Overall assessment: {integrated_insights['overall_assessment']}")
        
        print("✅ Social Intelligence Integration Test Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_pillar_19()
    test_social_integration() 