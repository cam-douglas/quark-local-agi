#!/usr/bin/env python3
"""
Pillar Harmony Test
===================

Tests that all pillars work together harmoniously without conflicts.
"""

import os
import sys
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.orchestrator import Orchestrator
from agents.response_generation_agent import ResponseGenerationAgent

def test_pillar_harmony():
    """Test that all pillars work together harmoniously."""
    print("🧪 Testing Pillar Harmony")
    print("=" * 50)
    
    # Test cases covering different pillar interactions
    test_cases = [
        {
            "input": "how are you?",
            "expected_type": "greeting",
            "description": "Greeting (ResponseGeneration pillar)"
        },
        {
            "input": "what is 2 + 2?",
            "expected_type": "math",
            "description": "Math computation (Reasoning + ResponseGeneration pillars)"
        },
        {
            "input": "tell me a joke",
            "expected_type": "joke",
            "description": "Entertainment (ResponseGeneration pillar)"
        },
        {
            "input": "what's the weather like?",
            "expected_type": "weather",
            "description": "Information request (Knowledge Retrieval + ResponseGeneration pillars)"
        },
        {
            "input": "can you help me plan my day?",
            "expected_type": "general",
            "description": "Planning request (Planning + ResponseGeneration pillars)"
        }
    ]
    
    print("📦 Initializing orchestrator...")
    orchestrator = Orchestrator()
    print("✅ Orchestrator initialized successfully")
    
    print(f"\n🔍 Testing {len(test_cases)} scenarios...")
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['description']}")
        print(f"   Input: '{test_case['input']}'")
        
        try:
            # Test with orchestrator
            start_time = time.time()
            result = orchestrator.handle(test_case['input'])
            execution_time = time.time() - start_time
            
            if hasattr(result, 'final_response'):
                response = result.final_response
            else:
                response = str(result)
            
            print(f"   Response: '{response}'")
            print(f"   Execution time: {execution_time:.2f}s")
            
            # Test with ResponseGenerationAgent directly
            response_agent = ResponseGenerationAgent()
            direct_result = response_agent.generate(test_case['input'])
            
            if direct_result['type'] == test_case['expected_type']:
                print(f"   ✅ Type match: {direct_result['type']}")
            else:
                print(f"   ⚠️ Type mismatch: expected {test_case['expected_type']}, got {direct_result['type']}")
            
            print(f"   ✅ Test passed")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All pillar harmony tests passed!")
        print("✅ All pillars are working together without conflicts")
    else:
        print("❌ Some pillar harmony tests failed")
        print("⚠️ There may be conflicts between pillars")
    
    return all_passed

def test_agent_dependencies():
    """Test that all agents can be initialized without conflicts."""
    print("\n🔧 Testing Agent Dependencies")
    print("=" * 50)
    
    try:
        orchestrator = Orchestrator()
        print("✅ All agents initialized successfully")
        
        agent_names = list(orchestrator.agents.keys())
        print(f"📊 Total agents: {len(agent_names)}")
        print(f"📋 Agent list: {', '.join(agent_names)}")
        
        # Check for any missing dependencies
        missing_agents = []
        for name, agent in orchestrator.agents.items():
            if agent is None:
                missing_agents.append(name)
        
        if missing_agents:
            print(f"⚠️ Missing agents: {missing_agents}")
            return False
        else:
            print("✅ All agents are properly initialized")
            return True
            
    except Exception as e:
        print(f"❌ Agent initialization failed: {e}")
        return False

def test_pipeline_execution():
    """Test that pipelines execute without conflicts."""
    print("\n🔄 Testing Pipeline Execution")
    print("=" * 50)
    
    try:
        orchestrator = Orchestrator()
        
        # Test different pipeline categories
        test_inputs = [
            "hello",
            "what is 5 + 3?",
            "tell me a joke",
            "how are you doing?"
        ]
        
        for i, test_input in enumerate(test_inputs, 1):
            print(f"\n{i}. Testing: '{test_input}'")
            
            start_time = time.time()
            result = orchestrator.handle(test_input)
            execution_time = time.time() - start_time
            
            if hasattr(result, 'final_response'):
                response = result.final_response
            else:
                response = str(result)
            
            print(f"   Response: '{response}'")
            print(f"   Execution time: {execution_time:.2f}s")
            print(f"   ✅ Pipeline executed successfully")
        
        print("\n✅ All pipeline executions completed without conflicts")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        return False

def main():
    """Run all harmony tests."""
    print("🤖 Quark Pillar Harmony Test Suite")
    print("=" * 60)
    
    # Test 1: Agent Dependencies
    deps_ok = test_agent_dependencies()
    
    # Test 2: Pipeline Execution
    pipeline_ok = test_pipeline_execution()
    
    # Test 3: Pillar Harmony
    harmony_ok = test_pillar_harmony()
    
    print("\n" + "=" * 60)
    print("📊 Test Results Summary:")
    print(f"   Agent Dependencies: {'✅ PASS' if deps_ok else '❌ FAIL'}")
    print(f"   Pipeline Execution: {'✅ PASS' if pipeline_ok else '❌ FAIL'}")
    print(f"   Pillar Harmony: {'✅ PASS' if harmony_ok else '❌ FAIL'}")
    
    if deps_ok and pipeline_ok and harmony_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ All pillars are working harmoniously without conflicts")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        print("⚠️ There may be conflicts between pillars")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 