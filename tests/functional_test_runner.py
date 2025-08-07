#!/usr/bin/env python3
"""
Functional Test Runner for Quark AI System
Tests actual functionality of key components
"""

import asyncio
import json
import tempfile
import shutil
from typing import Dict, Any

from agents.reasoning_agent import ReasoningAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.memory_agent import MemoryAgent
from core.context_window_manager import ContextWindowManager
from core.memory_eviction import MemoryEvictionManager


async def test_reasoning_functionality():
    """Test reasoning agent functionality"""
    print("=" * 60)
    print("🧠 TESTING REASONING FUNCTIONALITY")
    print("=" * 60)
    
    print("\n1. Creating ReasoningAgent...")
    reasoning_agent = ReasoningAgent()
    
    print("\n2. Testing deductive reasoning...")
    result = await reasoning_agent.process_message({
        "type": "reasoning_request",
        "query": "If all A are B, and all B are C, then what can we conclude about A and C?",
        "reasoning_type": "deductive"
    })
    
    print(f"   Result: {json.dumps(result, indent=2)}")
    assert result["status"] == "success"
    assert "conclusion" in result
    print("   ✅ Deductive reasoning working")
    
    print("\n3. Testing causal reasoning...")
    result = await reasoning_agent.process_message({
        "type": "reasoning_request",
        "query": "What are the likely causes of increased system latency?",
        "reasoning_type": "causal"
    })
    
    print(f"   Result: {json.dumps(result, indent=2)}")
    assert result["status"] == "success"
    assert "causes" in result or "analysis" in result
    print("   ✅ Causal reasoning working")
    
    print("\n✅ REASONING FUNCTIONALITY: WORKING")


async def test_explainability_functionality():
    """Test explainability agent functionality"""
    print("\n" + "=" * 60)
    print("🔍 TESTING EXPLAINABILITY FUNCTIONALITY")
    print("=" * 60)
    
    print("\n1. Creating ExplainabilityAgent...")
    explainability_agent = ExplainabilityAgent()
    
    print("\n2. Testing explanation generation...")
    result = await explainability_agent.process_message({
        "type": "explanation_request",
        "decision_id": "test_decision_1",
        "explanation_type": "decision_rationale",
        "context": {"decision": "approved", "factors": ["safety", "efficiency"]}
    })
    
    print(f"   Result: {json.dumps(result, indent=2)}")
    assert result["status"] == "success"
    assert "explanation" in result
    print("   ✅ Explanation generation working")
    
    print("\n3. Testing transparency report...")
    result = await explainability_agent.process_message({
        "type": "transparency_report",
        "component": "reasoning_engine",
        "transparency_level": "detailed"
    })
    
    print(f"   Result: {json.dumps(result, indent=2)}")
    assert result["status"] == "success"
    assert "report" in result
    print("   ✅ Transparency report working")
    
    print("\n✅ EXPLAINABILITY FUNCTIONALITY: WORKING")


async def test_memory_functionality():
    """Test memory system functionality"""
    print("\n" + "=" * 60)
    print("🧠 TESTING MEMORY FUNCTIONALITY")
    print("=" * 60)
    
    print("\n1. Creating MemoryAgent...")
    temp_dir = tempfile.mkdtemp()
    memory_agent = MemoryAgent(memory_dir=temp_dir)
    
    print("\n2. Testing memory storage...")
    result = memory_agent.generate("User likes pizza", operation="store_memory", memory_type="episodic")
    print(f"   Result: {json.dumps(result, indent=2)}")
    assert result["status"] == "success"
    print("   ✅ Memory storage working")
    
    print("\n3. Testing memory retrieval...")
    result = memory_agent.generate("pizza", operation="retrieve_memories", max_results=5)
    print(f"   Result: {json.dumps(result, indent=2)}")
    assert result["status"] == "success"
    assert "memories" in result
    print("   ✅ Memory retrieval working")
    
    print("\n4. Testing memory stats...")
    result = memory_agent.generate("", operation="get_memory_stats")
    print(f"   Result: {json.dumps(result, indent=2)}")
    assert result["status"] == "success"
    assert "basic_stats" in result
    print("   ✅ Memory stats working")
    
    print("\n5. Cleanup...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n✅ MEMORY FUNCTIONALITY: WORKING")


async def test_context_window_functionality():
    """Test context window manager functionality"""
    print("\n" + "=" * 60)
    print("💬 TESTING CONTEXT WINDOW FUNCTIONALITY")
    print("=" * 60)
    
    print("\n1. Creating ContextWindowManager...")
    context_manager = ContextWindowManager(max_tokens=1000)
    
    print("\n2. Testing message addition...")
    context_manager.add_message("user", "Hello")
    context_manager.add_message("assistant", "Hi there!")
    context_manager.add_message("user", "How are you?")
    
    print("\n3. Testing context retrieval...")
    context = context_manager.get_context()
    print(f"   Context: {json.dumps(context, indent=2)}")
    assert len(context) == 3
    print("   ✅ Context retrieval working")
    
    print("\n4. Testing context clearing...")
    context_manager.clear_context()
    context_after_clear = context_manager.get_context()
    assert len(context_after_clear) == 0
    print("   ✅ Context clearing working")
    
    print("\n✅ CONTEXT WINDOW FUNCTIONALITY: WORKING")


async def test_memory_eviction_functionality():
    """Test memory eviction functionality"""
    print("\n" + "=" * 60)
    print("🗑️  TESTING MEMORY EVICTION FUNCTIONALITY")
    print("=" * 60)
    
    print("\n1. Creating MemoryAgent and EvictionManager...")
    temp_dir = tempfile.mkdtemp()
    memory_agent = MemoryAgent(memory_dir=temp_dir)
    eviction_manager = MemoryEvictionManager(memory_agent)
    
    print("\n2. Testing eviction policies...")
    policies = eviction_manager.eviction_policies
    print(f"   Policies: {json.dumps(policies, indent=2)}")
    assert "time_based" in policies
    assert "size_based" in policies
    assert "relevance_based" in policies
    print("   ✅ Eviction policies working")
    
    print("\n3. Testing cleanup execution...")
    cleanup_result = eviction_manager.run_cleanup()
    print(f"   Result: {json.dumps(cleanup_result, indent=2)}")
    assert "status" in cleanup_result
    print("   ✅ Cleanup execution working")
    
    print("\n4. Cleanup...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n✅ MEMORY EVICTION FUNCTIONALITY: WORKING")


async def test_integration_functionality():
    """Test integration between components"""
    print("\n" + "=" * 60)
    print("🔗 TESTING INTEGRATION FUNCTIONALITY")
    print("=" * 60)
    
    print("\n1. Creating multiple agents...")
    reasoning_agent = ReasoningAgent()
    explainability_agent = ExplainabilityAgent()
    temp_dir = tempfile.mkdtemp()
    memory_agent = MemoryAgent(memory_dir=temp_dir)
    
    print("\n2. Testing reasoning with explanation...")
    # First, do reasoning
    reasoning_result = await reasoning_agent.process_message({
        "type": "reasoning_request",
        "query": "If all A are B, and all B are C, then what can we conclude?",
        "reasoning_type": "deductive"
    })
    
    # Then, explain the reasoning
    explanation_result = await explainability_agent.process_message({
        "type": "explanation_request",
        "decision_id": "reasoning_decision_1",
        "explanation_type": "decision_rationale",
        "context": {"reasoning_result": reasoning_result}
    })
    
    print(f"   Reasoning: {json.dumps(reasoning_result, indent=2)}")
    print(f"   Explanation: {json.dumps(explanation_result, indent=2)}")
    
    assert reasoning_result["status"] == "success"
    assert explanation_result["status"] == "success"
    print("   ✅ Reasoning with explanation working")
    
    print("\n3. Testing memory with reasoning...")
    # Store reasoning result in memory
    memory_result = memory_agent.generate(
        json.dumps(reasoning_result), 
        operation="store_memory", 
        memory_type="episodic"
    )
    
    # Retrieve and use in reasoning
    retrieval_result = memory_agent.generate(
        "reasoning", 
        operation="retrieve_memories", 
        max_results=1
    )
    
    print(f"   Memory storage: {json.dumps(memory_result, indent=2)}")
    print(f"   Memory retrieval: {json.dumps(retrieval_result, indent=2)}")
    
    assert memory_result["status"] == "success"
    assert retrieval_result["status"] == "success"
    print("   ✅ Memory with reasoning working")
    
    print("\n4. Cleanup...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n✅ INTEGRATION FUNCTIONALITY: WORKING")


async def run_all_functional_tests():
    """Run all functional tests"""
    print("🚀 QUARK AI SYSTEM - FUNCTIONAL TEST RUNNER")
    print("=" * 60)
    
    tests = [
        test_reasoning_functionality,
        test_explainability_functionality,
        test_memory_functionality,
        test_context_window_functionality,
        test_memory_eviction_functionality,
        test_integration_functionality
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests, 1):
        try:
            print(f"\n📋 Running Functional Test {i}/{len(tests)}: {test.__name__}")
            await test()
            passed += 1
        except Exception as e:
            print(f"\n❌ FUNCTIONAL TEST FAILED: {test.__name__}")
            print(f"   Error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 FUNCTIONAL TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL FUNCTIONAL TESTS PASSED!")
        print("   The Quark AI System is fully functional and operational.")
    else:
        print(f"\n⚠️  {failed} functional test(s) failed")
        print("   Some functionality may not be working correctly.")


if __name__ == "__main__":
    asyncio.run(run_all_functional_tests()) 