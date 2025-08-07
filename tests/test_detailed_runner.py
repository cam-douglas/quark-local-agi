#!/usr/bin/env python3
"""
Detailed Test Runner for Quark AI System
Shows the process of each test with actual input/output
"""

import asyncio
import json
from typing import Dict, Any

from agents.reasoning_agent import ReasoningAgent
from agents.explainability_agent import ExplainabilityAgent


async def test_agent_initialization():
    """Test agent initialization process"""
    print("=" * 60)
    print("TEST: Agent Initialization")
    print("=" * 60)
    
    print("\n1. Creating ReasoningAgent...")
    reasoning_agent = ReasoningAgent()
    print(f"   ✓ ReasoningAgent created: {reasoning_agent}")
    print(f"   ✓ Name attribute: {reasoning_agent.name}")
    
    print("\n2. Creating ExplainabilityAgent...")
    explainability_agent = ExplainabilityAgent()
    print(f"   ✓ ExplainabilityAgent created: {explainability_agent}")
    print(f"   ✓ Name attribute: {explainability_agent.name}")
    
    print("\n3. Validation...")
    assert reasoning_agent is not None
    assert reasoning_agent.name == "reasoning"
    assert explainability_agent is not None
    assert explainability_agent.name == "explainability"
    print("   ✓ All assertions passed!")
    
    print("\n✅ TEST PASSED: Agent Initialization")


async def test_advanced_reasoning_capabilities():
    """Test advanced reasoning capabilities process"""
    print("=" * 60)
    print("TEST: Advanced Reasoning Capabilities")
    print("=" * 60)
    
    print("\n1. Creating ReasoningAgent...")
    reasoning_agent = ReasoningAgent()
    
    print("\n2. Preparing test message...")
    test_message = {
        "type": "reasoning_request",
        "query": "If all A are B, and all B are C, then what can we conclude about A and C?",
        "reasoning_type": "deductive"
    }
    print(f"   Input message: {json.dumps(test_message, indent=2)}")
    
    print("\n3. Processing message...")
    result = await reasoning_agent.process_message(test_message)
    print(f"   Output result: {json.dumps(result, indent=2)}")
    
    print("\n4. Validation...")
    assert result["status"] == "success"
    assert "conclusion" in result
    print("   ✓ Status is 'success'")
    print("   ✓ Contains 'conclusion' field")
    
    print("\n✅ TEST PASSED: Advanced Reasoning Capabilities")


async def test_explanation_generation():
    """Test explanation generation process"""
    print("=" * 60)
    print("TEST: Explanation Generation")
    print("=" * 60)
    
    print("\n1. Creating ExplainabilityAgent...")
    explainability_agent = ExplainabilityAgent()
    
    print("\n2. Preparing test message...")
    test_message = {
        "type": "explanation_request",
        "decision_id": "test_decision_1",
        "explanation_type": "decision_rationale",
        "context": {"decision": "approved", "factors": ["safety", "efficiency"]}
    }
    print(f"   Input message: {json.dumps(test_message, indent=2)}")
    
    print("\n3. Processing message...")
    result = await explainability_agent.process_message(test_message)
    print(f"   Output result: {json.dumps(result, indent=2)}")
    
    print("\n4. Validation...")
    assert result["status"] == "success"
    assert "explanation" in result
    print("   ✓ Status is 'success'")
    print("   ✓ Contains 'explanation' field")
    
    print("\n✅ TEST PASSED: Explanation Generation")


async def test_transparency_report():
    """Test transparency report generation process"""
    print("=" * 60)
    print("TEST: Transparency Report Generation")
    print("=" * 60)
    
    print("\n1. Creating ExplainabilityAgent...")
    explainability_agent = ExplainabilityAgent()
    
    print("\n2. Preparing test message...")
    test_message = {
        "type": "transparency_report",
        "component": "reasoning_engine",
        "transparency_level": "detailed"
    }
    print(f"   Input message: {json.dumps(test_message, indent=2)}")
    
    print("\n3. Processing message...")
    result = await explainability_agent.process_message(test_message)
    print(f"   Output result: {json.dumps(result, indent=2)}")
    
    print("\n4. Validation...")
    assert result["status"] == "success"
    assert "report" in result
    print("   ✓ Status is 'success'")
    print("   ✓ Contains 'report' field")
    
    print("\n✅ TEST PASSED: Transparency Report Generation")


async def test_causal_reasoning():
    """Test causal reasoning process"""
    print("=" * 60)
    print("TEST: Causal Reasoning")
    print("=" * 60)
    
    print("\n1. Creating ReasoningAgent...")
    reasoning_agent = ReasoningAgent()
    
    print("\n2. Preparing test message...")
    test_message = {
        "type": "reasoning_request",
        "query": "What are the likely causes of increased system latency?",
        "reasoning_type": "causal"
    }
    print(f"   Input message: {json.dumps(test_message, indent=2)}")
    
    print("\n3. Processing message...")
    result = await reasoning_agent.process_message(test_message)
    print(f"   Output result: {json.dumps(result, indent=2)}")
    
    print("\n4. Validation...")
    assert result["status"] == "success"
    assert "causes" in result or "analysis" in result
    print("   ✓ Status is 'success'")
    print("   ✓ Contains 'causes' or 'analysis' field")
    
    print("\n✅ TEST PASSED: Causal Reasoning")


async def test_abstract_reasoning():
    """Test abstract reasoning process"""
    print("=" * 60)
    print("TEST: Abstract Reasoning")
    print("=" * 60)
    
    print("\n1. Creating ReasoningAgent...")
    reasoning_agent = ReasoningAgent()
    
    print("\n2. Preparing test message...")
    test_message = {
        "type": "reasoning_request",
        "query": "How can we improve system efficiency through architectural changes?",
        "reasoning_type": "abstract"
    }
    print(f"   Input message: {json.dumps(test_message, indent=2)}")
    
    print("\n3. Processing message...")
    result = await reasoning_agent.process_message(test_message)
    print(f"   Output result: {json.dumps(result, indent=2)}")
    
    print("\n4. Validation...")
    assert result["status"] == "success"
    assert "suggestions" in result or "recommendations" in result
    print("   ✓ Status is 'success'")
    print("   ✓ Contains 'suggestions' or 'recommendations' field")
    
    print("\n✅ TEST PASSED: Abstract Reasoning")


async def run_all_tests():
    """Run all tests with detailed output"""
    print("🚀 QUARK AI SYSTEM - DETAILED TEST RUNNER")
    print("=" * 60)
    
    tests = [
        test_agent_initialization,
        test_advanced_reasoning_capabilities,
        test_explanation_generation,
        test_transparency_report,
        test_causal_reasoning,
        test_abstract_reasoning
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests, 1):
        try:
            print(f"\n📋 Running Test {i}/{len(tests)}: {test.__name__}")
            await test()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {test.__name__}")
            print(f"   Error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")


if __name__ == "__main__":
    asyncio.run(run_all_tests()) 