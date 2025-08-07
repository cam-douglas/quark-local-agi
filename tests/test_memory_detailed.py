#!/usr/bin/env python3
"""
Detailed Memory System Test Runner
Shows the process of memory system tests with actual input/output
"""

import asyncio
import json
import tempfile
import shutil
from typing import Dict, Any

from agents.memory_agent import MemoryAgent
from core.context_window_manager import ContextWindowManager
from core.memory_eviction import MemoryEvictionManager


async def test_memory_agent_initialization():
    """Test memory agent initialization process"""
    print("=" * 60)
    print("TEST: Memory Agent Initialization")
    print("=" * 60)
    
    print("\n1. Creating temporary directory...")
    temp_dir = tempfile.mkdtemp()
    print(f"   ✓ Temp directory created: {temp_dir}")
    
    print("\n2. Creating MemoryAgent...")
    memory_agent = MemoryAgent(memory_dir=temp_dir)
    print(f"   ✓ MemoryAgent created: {memory_agent}")
    print(f"   ✓ Long-term memory initialized: {memory_agent.long_term_memory}")
    
    print("\n3. Validation...")
    assert memory_agent is not None
    assert memory_agent.long_term_memory is not None
    print("   ✓ All assertions passed!")
    
    print("\n4. Cleanup...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("   ✓ Temporary directory cleaned up")
    
    print("\n✅ TEST PASSED: Memory Agent Initialization")


async def test_store_memory():
    """Test memory storage process"""
    print("=" * 60)
    print("TEST: Store Memory")
    print("=" * 60)
    
    print("\n1. Creating MemoryAgent...")
    temp_dir = tempfile.mkdtemp()
    memory_agent = MemoryAgent(memory_dir=temp_dir)
    
    print("\n2. Preparing test data...")
    content = "User likes pizza"
    print(f"   Content to store: '{content}'")
    
    print("\n3. Storing memory...")
    result = memory_agent.generate(content, operation="store_memory", memory_type="episodic")
    print(f"   Result: {json.dumps(result, indent=2)}")
    
    print("\n4. Validation...")
    assert isinstance(result, dict)
    assert "status" in result
    print("   ✓ Result is a dictionary")
    print("   ✓ Contains 'status' field")
    
    print("\n5. Cleanup...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n✅ TEST PASSED: Store Memory")


async def test_retrieve_memories():
    """Test memory retrieval process"""
    print("=" * 60)
    print("TEST: Retrieve Memories")
    print("=" * 60)
    
    print("\n1. Creating MemoryAgent...")
    temp_dir = tempfile.mkdtemp()
    memory_agent = MemoryAgent(memory_dir=temp_dir)
    
    print("\n2. Storing test memory...")
    content = "User prefers Italian food"
    store_result = memory_agent.generate(content, operation="store_memory", memory_type="episodic")
    print(f"   Stored: {json.dumps(store_result, indent=2)}")
    
    print("\n3. Retrieving memories...")
    query = "food preferences"
    result = memory_agent.generate(query, operation="retrieve_memories", max_results=5)
    print(f"   Query: '{query}'")
    print(f"   Result: {json.dumps(result, indent=2)}")
    
    print("\n4. Validation...")
    assert isinstance(result, dict)
    assert "memories" in result
    print("   ✓ Result is a dictionary")
    print("   ✓ Contains 'memories' field")
    
    print("\n5. Cleanup...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n✅ TEST PASSED: Retrieve Memories")


async def test_context_window_manager():
    """Test context window manager process"""
    print("=" * 60)
    print("TEST: Context Window Manager")
    print("=" * 60)
    
    print("\n1. Creating ContextWindowManager...")
    context_manager = ContextWindowManager(max_tokens=1000)
    print(f"   ✓ ContextWindowManager created: {context_manager}")
    print(f"   ✓ Max tokens: {context_manager.max_tokens}")
    
    print("\n2. Adding messages...")
    messages = [
        ("user", "Hello"),
        ("assistant", "Hi there!"),
        ("user", "How are you?")
    ]
    
    for role, content in messages:
        print(f"   Adding {role}: '{content}'")
        context_manager.add_message(role, content)
    
    print("\n3. Getting context...")
    context = context_manager.get_context()
    print(f"   Context: {json.dumps(context, indent=2)}")
    
    print("\n4. Validation...")
    assert isinstance(context, list)
    assert len(context) > 0
    print("   ✓ Context is a list")
    print("   ✓ Context has messages")
    
    print("\n5. Clearing context...")
    context_manager.clear_context()
    context_after_clear = context_manager.get_context()
    print(f"   Context after clear: {json.dumps(context_after_clear, indent=2)}")
    
    print("\n6. Validation after clear...")
    assert len(context_after_clear) == 0
    print("   ✓ Context is empty after clear")
    
    print("\n✅ TEST PASSED: Context Window Manager")


async def test_memory_eviction_manager():
    """Test memory eviction manager process"""
    print("=" * 60)
    print("TEST: Memory Eviction Manager")
    print("=" * 60)
    
    print("\n1. Creating MemoryAgent and EvictionManager...")
    temp_dir = tempfile.mkdtemp()
    memory_agent = MemoryAgent(memory_dir=temp_dir)
    eviction_manager = MemoryEvictionManager(memory_agent)
    print(f"   ✓ MemoryAgent created: {memory_agent}")
    print(f"   ✓ EvictionManager created: {eviction_manager}")
    
    print("\n2. Checking eviction policies...")
    policies = eviction_manager.eviction_policies
    print(f"   Policies: {json.dumps(policies, indent=2)}")
    
    print("\n3. Running cleanup...")
    cleanup_result = eviction_manager.run_cleanup()
    print(f"   Cleanup result: {json.dumps(cleanup_result, indent=2)}")
    
    print("\n4. Validation...")
    assert isinstance(cleanup_result, dict)
    assert "status" in cleanup_result
    print("   ✓ Cleanup result is a dictionary")
    print("   ✓ Contains 'status' field")
    
    print("\n5. Cleanup...")
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    print("\n✅ TEST PASSED: Memory Eviction Manager")


async def run_all_memory_tests():
    """Run all memory system tests with detailed output"""
    print("🧠 QUARK AI SYSTEM - MEMORY SYSTEM TEST RUNNER")
    print("=" * 60)
    
    tests = [
        test_memory_agent_initialization,
        test_store_memory,
        test_retrieve_memories,
        test_context_window_manager,
        test_memory_eviction_manager
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(tests, 1):
        try:
            print(f"\n📋 Running Memory Test {i}/{len(tests)}: {test.__name__}")
            await test()
            passed += 1
        except Exception as e:
            print(f"\n❌ MEMORY TEST FAILED: {test.__name__}")
            print(f"   Error: {str(e)}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("📊 MEMORY TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/(passed+failed)*100):.1f}%")
    
    if failed == 0:
        print("\n🎉 ALL MEMORY TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} memory test(s) failed")


if __name__ == "__main__":
    asyncio.run(run_all_memory_tests()) 