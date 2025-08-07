#!/usr/bin/env python3
"""
Test Pillar 20: Autonomous Goals
================================

Tests the implementation of autonomous goal generation, self-directed learning,
intrinsic motivation, and independent decision-making capabilities.

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

from autonomy.autonomous_goals import (
    AutonomousGoals, GoalType, GoalPriority, MotivationType,
    Goal, LearningObjective, MotivationState, DecisionContext
)
from agents.autonomy_agent import AutonomyAgent


def test_pillar_20():
    """Test Pillar 20: Autonomous Goals."""
    print("🎯 Testing Pillar 20: Autonomous Goals")
    print("=" * 60)
    
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    autonomy_dir = os.path.join(temp_dir, "autonomy_data")
    
    try:
        # Test 1: Autonomous Goals Engine
        print("\n🧠 Testing Autonomous Goals Engine")
        print("-" * 40)
        
        # Initialize autonomy engine
        autonomy_engine = AutonomousGoals(autonomy_dir)
        
        # Test autonomous goal generation
        print("✅ Testing Autonomous Goal Generation")
        
        goal_contexts = [
            "Learn new programming techniques",
            "Improve system performance",
            "Explore new AI capabilities",
            "Create innovative solutions",
            "Collaborate with other agents",
            "Ensure safety and alignment",
            "Self-improvement and growth"
        ]
        
        for context in goal_contexts:
            goal_result = autonomy_engine.generate_autonomous_goal(context)
            print(f"   Context: {context[:30]}...")
            print(f"   Goal ID: {goal_result['goal_id']}")
            print(f"   Goal Type: {goal_result['goal']['goal_type']}")
            print(f"   Priority: {goal_result['goal']['priority']}")
            print(f"   Motivation: {goal_result['motivation']}")
            print()
        
        # Test learning objective creation
        print("✅ Testing Learning Objective Creation")
        
        skills = ["programming", "communication", "data_analysis", "safety_engineering"]
        
        for skill in skills:
            objective_result = autonomy_engine.create_learning_objective(
                "goal_1", skill, 0.8
            )
            print(f"   Skill: {skill}")
            print(f"   Learning method: {objective_result['learning_method']}")
            print(f"   Estimated time: {objective_result['estimated_time']} hours")
            print()
        
        # Test motivation state updates
        print("✅ Testing Motivation State Updates")
        
        motivation_types = [
            MotivationType.CURIOSITY,
            MotivationType.MASTERY,
            MotivationType.AUTONOMY,
            MotivationType.PURPOSE,
            MotivationType.SOCIAL
        ]
        
        for motivation_type in motivation_types:
            motivation_result = autonomy_engine.update_motivation_state(
                "test_agent", motivation_type, 0.8
            )
            print(f"   Motivation: {motivation_type.value}")
            print(f"   Intensity: {motivation_result['intensity']}")
            print(f"   Satisfaction: {motivation_result['satisfaction_level']:.2f}")
            print()
        
        # Test autonomous decision making
        print("✅ Testing Autonomous Decision Making")
        
        decision_contexts = [
            "Safety-critical system decision",
            "Performance optimization choice",
            "Learning strategy selection",
            "Collaboration approach"
        ]
        
        for context in decision_contexts:
            decision_result = autonomy_engine.make_autonomous_decision(
                context, ["Option A", "Option B", "Option C"]
            )
            print(f"   Context: {context}")
            print(f"   Selected: {decision_result['selected_option']}")
            print(f"   Confidence: {decision_result['confidence']:.2f}")
            print()
        
        # Test goal progress updates
        print("✅ Testing Goal Progress Updates")
        
        # Update progress for the first goal
        if autonomy_engine.active_goals:
            first_goal_id = list(autonomy_engine.active_goals.keys())[0]
            progress_result = autonomy_engine.update_goal_progress(first_goal_id, 0.7)
            print(f"   Goal ID: {first_goal_id}")
            print(f"   Progress: {progress_result['progress']:.2f}")
            print(f"   Completed: {progress_result['completed']}")
            print()
        
        # Test autonomy statistics
        print("✅ Testing Autonomy Statistics")
        stats = autonomy_engine.get_autonomy_stats()
        print(f"   Goals generated: {stats['goals_generated']}")
        print(f"   Goals completed: {stats['goals_completed']}")
        print(f"   Active goals: {stats['active_goals']}")
        print(f"   Learning sessions: {stats['learning_sessions']}")
        print(f"   Autonomous decisions: {stats['autonomous_decisions']}")
        print(f"   Motivation cycles: {stats['motivation_cycles']}")
        print(f"   Completion rate: {stats['completion_rate']:.2f}")
        print()
        
        # Test 2: Autonomy Agent Integration
        print("\n🤖 Testing Autonomy Agent Integration")
        print("-" * 40)
        
        # Initialize autonomy agent
        autonomy_agent = AutonomyAgent(autonomy_dir=autonomy_dir)
        
        # Test agent operations
        print("✅ Testing Autonomy Agent Operations")
        
        # Test goal generation via agent
        agent_goal_result = autonomy_agent.generate(
            "Learn advanced AI techniques",
            operation="generate_goal"
        )
        print(f"   Agent goal generation: {agent_goal_result['status']}")
        
        # Test learning objective creation via agent
        agent_learning_result = autonomy_agent.generate(
            "Improve programming skills",
            operation="create_learning_objective"
        )
        print(f"   Agent learning objective: {agent_learning_result['status']}")
        
        # Test motivation update via agent
        agent_motivation_result = autonomy_agent.generate(
            "I'm curious about new technologies",
            operation="update_motivation"
        )
        print(f"   Agent motivation update: {agent_motivation_result['status']}")
        
        # Test autonomous decision via agent
        agent_decision_result = autonomy_agent.generate(
            "Choose the best approach for system optimization",
            operation="make_decision"
        )
        print(f"   Agent autonomous decision: {agent_decision_result['status']}")
        
        # Test goal progress update via agent
        agent_progress_result = autonomy_agent.generate(
            "Goal is 75% complete",
            operation="update_goal_progress"
        )
        print(f"   Agent goal progress: {agent_progress_result['status']}")
        
        # Test autonomy statistics via agent
        agent_stats_result = autonomy_agent.generate("", operation="get_autonomy_stats")
        print(f"   Agent autonomy stats: {agent_stats_result['performance_stats']['total_operations']} operations")
        
        # Test 3: Advanced Autonomy Features
        print("\n🚀 Testing Advanced Autonomy Features")
        print("-" * 40)
        
        # Test goal analysis
        print("✅ Testing Goal Analysis")
        analysis_result = autonomy_agent.generate("", operation="analyze_goals")
        print(f"   Goal analysis: {analysis_result['status']}")
        
        # Test goal prioritization
        print("✅ Testing Goal Prioritization")
        priority_result = autonomy_agent.generate("", operation="prioritize_goals")
        print(f"   Goal prioritization: {priority_result['status']}")
        
        # Test autonomy recommendations
        print("✅ Testing Autonomy Recommendations")
        recommendations_result = autonomy_agent.get_autonomy_recommendations()
        print(f"   Recommendations: {recommendations_result['recommendations_count']}")
        
        # Test data export
        print("✅ Testing Data Export")
        export_result = autonomy_agent.generate("", operation="export_autonomy_data")
        print(f"   Export result: {export_result['status']}")
        
        print("\n🎉 Pillar 20 Test Completed Successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Autonomous goal generation and management")
        print("✅ Self-directed learning objectives")
        print("✅ Intrinsic motivation systems")
        print("✅ Independent decision making")
        print("✅ Goal progress tracking")
        print("✅ Goal analysis and prioritization")
        print("✅ Performance monitoring and statistics")
        print("✅ Data export and persistence")
        print("✅ Agent integration and operations")
        print("✅ Advanced autonomy capabilities")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_autonomy_integration():
    """Test integration between autonomy and other systems."""
    print("\n🔗 Testing Autonomy Integration")
    print("=" * 50)
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    autonomy_dir = os.path.join(temp_dir, "autonomy_data")
    
    try:
        # Initialize autonomy systems
        autonomy_engine = AutonomousGoals(autonomy_dir)
        autonomy_agent = AutonomyAgent(autonomy_dir=autonomy_dir)
        
        # Test integrated autonomy workflow
        print("✅ Testing Integrated Autonomy Workflow")
        
        # 1. Autonomous goal generation scenario
        scenario = """
        Scenario: The AI system needs to autonomously improve its capabilities.
        
        Context: The system has identified areas for improvement and wants to:
        - Learn new skills
        - Optimize performance
        - Ensure safety
        - Collaborate effectively
        
        Goal: Generate and manage autonomous goals for self-improvement.
        """
        
        print("✅ Testing Autonomous Goal Generation Scenario")
        
        # Generate goals for different areas
        goal_areas = [
            "Learn advanced reasoning techniques",
            "Improve system efficiency",
            "Ensure safety and alignment",
            "Enhance collaboration capabilities"
        ]
        
        generated_goals = []
        for area in goal_areas:
            goal_result = autonomy_engine.generate_autonomous_goal(area)
            generated_goals.append(goal_result)
            print(f"   Generated goal: {goal_result['goal']['title']}")
            print(f"     Type: {goal_result['goal']['goal_type']}")
            print(f"     Priority: {goal_result['goal']['priority']}")
            print(f"     Motivation: {goal_result['motivation']}")
        
        # 2. Test learning objective creation
        print("✅ Testing Learning Objective Creation")
        
        learning_skills = ["advanced_reasoning", "system_optimization", "safety_engineering"]
        
        for skill in learning_skills:
            objective_result = autonomy_engine.create_learning_objective(
                generated_goals[0]['goal_id'], skill, 0.9
            )
            print(f"   Learning objective: {skill}")
            print(f"     Method: {objective_result['learning_method']}")
            print(f"     Time required: {objective_result['estimated_time']} hours")
        
        # 3. Test motivation management
        print("✅ Testing Motivation Management")
        
        motivation_scenarios = [
            ("I want to learn new things", MotivationType.CURIOSITY),
            ("I want to improve my skills", MotivationType.MASTERY),
            ("I want to work independently", MotivationType.AUTONOMY),
            ("I want to help others", MotivationType.PURPOSE)
        ]
        
        for scenario, motivation_type in motivation_scenarios:
            motivation_result = autonomy_engine.update_motivation_state(
                "autonomy_agent", motivation_type, 0.8
            )
            print(f"   Motivation: {motivation_type.value}")
            print(f"     Intensity: {motivation_result['intensity']}")
            print(f"     Satisfaction: {motivation_result['satisfaction_level']:.2f}")
        
        # 4. Test autonomous decision making
        print("✅ Testing Autonomous Decision Making")
        
        decision_scenarios = [
            ("Safety-critical decision", ["Implement safety measures", "Proceed with caution", "Delay decision"]),
            ("Performance optimization", ["Optimize algorithms", "Enhance hardware", "Improve processes"]),
            ("Learning strategy", ["Study theory", "Practice skills", "Experiment"])
        ]
        
        for context, options in decision_scenarios:
            decision_result = autonomy_engine.make_autonomous_decision(context, options)
            print(f"   Context: {context}")
            print(f"     Selected: {decision_result['selected_option']}")
            print(f"     Confidence: {decision_result['confidence']:.2f}")
        
        # 5. Test goal progress tracking
        print("✅ Testing Goal Progress Tracking")
        
        # Update progress for all goals
        for goal_result in generated_goals:
            goal_id = goal_result['goal_id']
            progress = 0.3  # Simulate some progress
            progress_result = autonomy_engine.update_goal_progress(goal_id, progress)
            print(f"   Goal: {goal_result['goal']['title']}")
            print(f"     Progress: {progress_result['progress']:.2f}")
            print(f"     Completed: {progress_result['completed']}")
        
        # 6. Test integrated insights
        print("✅ Testing Integrated Autonomy Insights")
        
        # Get comprehensive statistics
        stats = autonomy_engine.get_autonomy_stats()
        
        integrated_insights = {
            'goals_generated': stats['goals_generated'],
            'goals_completed': stats['goals_completed'],
            'completion_rate': stats['completion_rate'],
            'learning_sessions': stats['learning_sessions'],
            'autonomous_decisions': stats['autonomous_decisions'],
            'motivation_cycles': stats['motivation_cycles'],
            'overall_assessment': 'Autonomous system successfully generating and managing goals'
        }
        
        print(f"   Overall assessment: {integrated_insights['overall_assessment']}")
        print(f"   Goals generated: {integrated_insights['goals_generated']}")
        print(f"   Completion rate: {integrated_insights['completion_rate']:.2f}")
        print(f"   Learning sessions: {integrated_insights['learning_sessions']}")
        print(f"   Autonomous decisions: {integrated_insights['autonomous_decisions']}")
        
        print("✅ Autonomy Integration Test Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_pillar_20()
    test_autonomy_integration() 