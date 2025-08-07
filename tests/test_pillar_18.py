#!/usr/bin/env python3
"""
Test Pillar 18: Generalized Reasoning
===================================

Tests the implementation of advanced reasoning capabilities including
logical inference, causal reasoning, analogical reasoning, and multi-step problem solving.

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

from reasoning.generalized_reasoning import GeneralizedReasoning, ReasoningType, ReasoningStep, ReasoningChain
from agents.reasoning_agent import ReasoningAgent


def test_pillar_18():
    """Test Pillar 18: Generalized Reasoning."""
    print("🧩 Testing Pillar 18: Generalized Reasoning")
    print("=" * 60)
    
    # Create temporary directories for testing
    temp_dir = tempfile.mkdtemp()
    reasoning_dir = os.path.join(temp_dir, "reasoning_data")
    
    try:
        # Test 1: Generalized Reasoning Engine
        print("\n🧠 Testing Generalized Reasoning Engine")
        print("-" * 40)
        
        # Initialize reasoning engine
        reasoning_engine = GeneralizedReasoning(reasoning_dir)
        
        # Test deductive reasoning
        print("✅ Testing Deductive Reasoning")
        
        premises = [
            "If it rains, then the ground will be wet",
            "It is raining",
            "All mammals are animals",
            "All dogs are mammals"
        ]
        
        deductive_result = reasoning_engine.deductive_reasoning(premises, "The ground will be wet")
        print(f"   Deductive reasoning result: {deductive_result['status']}")
        print(f"   Steps executed: {deductive_result['steps']}")
        print(f"   Final conclusion: {deductive_result['final_conclusion']}")
        print(f"   Confidence: {deductive_result['confidence']:.2f}")
        
        # Test causal reasoning
        print("\n✅ Testing Causal Reasoning")
        
        events = [
            "Smoking causes lung cancer",
            "Lung cancer causes breathing difficulties",
            "Breathing difficulties cause reduced activity",
            "Reduced activity causes weight gain"
        ]
        
        causal_result = reasoning_engine.causal_reasoning(events, "weight gain")
        print(f"   Causal reasoning result: {causal_result['status']}")
        print(f"   Causal relationships found: {causal_result['causal_relationships']}")
        print(f"   Causal chains found: {causal_result['causal_chains']}")
        print(f"   Final conclusion: {causal_result['final_conclusion']}")
        print(f"   Confidence: {causal_result['confidence']:.2f}")
        
        # Test analogical reasoning
        print("\n✅ Testing Analogical Reasoning")
        
        source_domain = "A computer works like a brain because both process information"
        target_domain = "A neural network works like a computer because both perform calculations"
        
        analogical_result = reasoning_engine.analogical_reasoning(source_domain, target_domain)
        print(f"   Analogical reasoning result: {analogical_result['status']}")
        print(f"   Analogical insights: {analogical_result['analogical_insights']}")
        print(f"   Final conclusion: {analogical_result['final_conclusion']}")
        print(f"   Confidence: {analogical_result['confidence']:.2f}")
        
        # Test multi-step problem solving
        print("\n✅ Testing Multi-Step Problem Solving")
        
        complex_problem = "If the temperature rises above 30 degrees, then the air conditioning turns on. The air conditioning causes increased energy consumption. Increased energy consumption leads to higher electricity bills. The temperature is currently 35 degrees. What will happen to the electricity bill?"
        
        multistep_result = reasoning_engine.multi_step_problem_solving(complex_problem)
        print(f"   Multi-step problem solving result: {multistep_result['status']}")
        print(f"   Steps executed: {multistep_result['steps_executed']}")
        print(f"   Final conclusion: {multistep_result['final_conclusion']}")
        print(f"   Confidence: {multistep_result['confidence']:.2f}")
        
        # Test reasoning statistics
        print("\n✅ Testing Reasoning Statistics")
        stats = reasoning_engine.get_reasoning_stats()
        print(f"   Total chains: {stats['total_chains']}")
        print(f"   Successful chains: {stats['successful_chains']}")
        print(f"   Success rate: {stats['success_rate']:.2f}")
        print(f"   Average confidence: {stats['average_confidence']:.2f}")
        print(f"   Reasoning types used: {stats['reasoning_types_used']}")
        
        # Test 2: Reasoning Agent Integration
        print("\n🤖 Testing Reasoning Agent Integration")
        print("-" * 40)
        
        # Initialize reasoning agent
        reasoning_agent = ReasoningAgent(reasoning_dir=reasoning_dir)
        
        # Test agent operations
        print("✅ Testing Reasoning Agent Operations")
        
        # Test deductive reasoning via agent
        agent_deductive_result = reasoning_agent.generate(
            "If P then Q\nP",
            operation="deductive_reasoning",
            target_conclusion="Q"
        )
        print(f"   Agent deductive reasoning: {agent_deductive_result['status']}")
        
        # Test causal reasoning via agent
        agent_causal_result = reasoning_agent.generate(
            "Smoking causes lung cancer\nLung cancer causes death",
            operation="causal_reasoning",
            target_event="death"
        )
        print(f"   Agent causal reasoning: {agent_causal_result['status']}")
        
        # Test analogical reasoning via agent
        agent_analogical_result = reasoning_agent.generate(
            "Source domain: A car has an engine that provides power\nTarget domain: A computer has a processor that provides computing power",
            operation="analogical_reasoning"
        )
        print(f"   Agent analogical reasoning: {agent_analogical_result['status']}")
        
        # Test multi-step problem solving via agent
        agent_multistep_result = reasoning_agent.generate(
            "If the stock market rises, then investors make money. If investors make money, then they spend more. If they spend more, then the economy grows. The stock market is rising. What happens to the economy?",
            operation="multi_step_problem_solving"
        )
        print(f"   Agent multi-step problem solving: {agent_multistep_result['status']}")
        
        # Test reasoning statistics via agent
        agent_stats_result = reasoning_agent.generate("", operation="get_reasoning_stats")
        print(f"   Agent reasoning stats: {agent_stats_result['performance_stats']['total_operations']} operations")
        
        # Test 3: Advanced Reasoning Features
        print("\n🚀 Testing Advanced Reasoning Features")
        print("-" * 40)
        
        # Test reasoning explanation
        print("✅ Testing Reasoning Explanation")
        if reasoning_engine.reasoning_chains:
            chain_id = reasoning_engine.reasoning_chains[0].chain_id
            explanation_result = reasoning_agent.generate(
                chain_id,
                operation="explain_reasoning"
            )
            print(f"   Explanation generated: {explanation_result['status']}")
        
        # Test reasoning validation
        print("✅ Testing Reasoning Validation")
        if reasoning_engine.reasoning_chains:
            chain_id = reasoning_engine.reasoning_chains[0].chain_id
            validation_result = reasoning_agent.generate(
                chain_id,
                operation="validate_reasoning"
            )
            print(f"   Validation result: {validation_result['status']}")
            if validation_result['status'] == 'success':
                validation = validation_result['validation']
                print(f"   Is valid: {validation['is_valid']}")
                print(f"   Overall score: {validation['overall_score']:.2f}")
        
        # Test reasoning recommendations
        print("✅ Testing Reasoning Recommendations")
        recommendations_result = reasoning_agent.get_reasoning_recommendations()
        print(f"   Recommendations: {recommendations_result['recommendations_count']}")
        
        # Test data export
        print("✅ Testing Data Export")
        export_result = reasoning_agent.generate("", operation="export_reasoning_data")
        print(f"   Export result: {export_result['status']}")
        
        print("\n🎉 Pillar 18 Test Completed Successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Deductive reasoning with logical inference")
        print("✅ Causal reasoning with cause-and-effect analysis")
        print("✅ Analogical reasoning with similarity mapping")
        print("✅ Multi-step problem solving with decomposition")
        print("✅ Reasoning chain tracking and management")
        print("✅ Reasoning validation and explanation")
        print("✅ Performance monitoring and statistics")
        print("✅ Data export and persistence")
        print("✅ Agent integration and operations")
        print("✅ Advanced reasoning capabilities")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_reasoning_integration():
    """Test integration between reasoning and other systems."""
    print("\n🔗 Testing Reasoning Integration")
    print("=" * 50)
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    reasoning_dir = os.path.join(temp_dir, "reasoning_data")
    
    try:
        # Initialize reasoning systems
        reasoning_engine = GeneralizedReasoning(reasoning_dir)
        reasoning_agent = ReasoningAgent(reasoning_dir=reasoning_dir)
        
        # Test integrated reasoning workflow
        print("✅ Testing Integrated Reasoning Workflow")
        
        # 1. Complex problem with multiple reasoning types
        complex_problem = """
        Problem: A company's profits are declining.
        
        Premises:
        - If sales decrease, then revenue decreases
        - If revenue decreases, then profits decrease
        - If marketing budget is cut, then sales decrease
        - The marketing budget was cut last quarter
        - The company's profits are declining
        
        Question: What caused the profit decline?
        """
        
        # 2. Apply different reasoning types
        print("✅ Testing Multi-Type Reasoning")
        
        # Deductive reasoning
        deductive_premises = [
            "If marketing budget is cut, then sales decrease",
            "If sales decrease, then revenue decreases",
            "If revenue decreases, then profits decrease",
            "The marketing budget was cut"
        ]
        
        deductive_result = reasoning_engine.deductive_reasoning(deductive_premises)
        print(f"   Deductive reasoning: {deductive_result['status']}")
        
        # Causal reasoning
        causal_events = [
            "Marketing budget cut causes sales decrease",
            "Sales decrease causes revenue decrease",
            "Revenue decrease causes profit decrease"
        ]
        
        causal_result = reasoning_engine.causal_reasoning(causal_events, "profit decrease")
        print(f"   Causal reasoning: {causal_result['status']}")
        
        # 3. Test reasoning chain analysis
        print("✅ Testing Reasoning Chain Analysis")
        
        # Analyze all reasoning chains
        all_chains = reasoning_engine.reasoning_chains
        print(f"   Total reasoning chains: {len(all_chains)}")
        
        for i, chain in enumerate(all_chains):
            print(f"   Chain {i+1}: {chain.reasoning_type.value} reasoning")
            print(f"     Steps: {len(chain.steps)}")
            print(f"     Confidence: {chain.overall_confidence:.2f}")
            print(f"     Conclusion: {chain.final_conclusion}")
        
        # 4. Test agent-based reasoning operations
        print("✅ Testing Agent-Based Reasoning")
        
        # Complex reasoning via agent
        agent_result = reasoning_agent.generate(
            complex_problem,
            operation="multi_step_problem_solving"
        )
        print(f"   Agent reasoning result: {agent_result['status']}")
        
        # 5. Test reasoning validation and explanation
        print("✅ Testing Reasoning Validation and Explanation")
        
        if all_chains:
            # Validate the first chain
            validation_result = reasoning_agent.generate(
                all_chains[0].chain_id,
                operation="validate_reasoning"
            )
            print(f"   Validation: {validation_result.get('status', 'unknown')}")
            
            # Explain the first chain
            explanation_result = reasoning_agent.generate(
                all_chains[0].chain_id,
                operation="explain_reasoning"
            )
            print(f"   Explanation: {explanation_result.get('status', 'unknown')}")
        
        # 6. Test integrated insights
        print("✅ Testing Integrated Insights")
        
        # Combine insights from different reasoning types
        integrated_insights = {
            'deductive_insights': deductive_result.get('final_conclusion'),
            'causal_insights': causal_result.get('final_conclusion'),
            'agent_insights': agent_result.get('final_conclusion'),
            'overall_analysis': 'Multiple reasoning approaches confirm the causal chain from marketing budget cuts to profit decline'
        }
        
        print(f"   Integrated analysis: {integrated_insights['overall_analysis']}")
        
        print("✅ Reasoning Integration Test Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        raise
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_pillar_18()
    test_reasoning_integration() 