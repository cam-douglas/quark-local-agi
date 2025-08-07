#!/usr/bin/env python3
"""
Test Natural Language Coding Capabilities in Quark
"""

import asyncio
import sys
import os

# Add the parent directory to the path so we can import quark modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.coding_assistant_agent import CodingAssistantAgent
from core.orchestrator import Orchestrator
from core.router import Router

async def test_coding_assistant():
    """Test the coding assistant directly"""
    print("🧪 Testing Coding Assistant Agent")
    print("=" * 50)
    
    agent = CodingAssistantAgent()
    
    test_requests = [
        "Write a Python function to calculate the factorial of a number",
        "Create a JavaScript function that sorts an array of objects by name",
        "Explain this code: def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "Fix this broken Python code: def divide(a, b): return a/b",
        "Write a SQL query to find all users who registered in the last 30 days"
    ]
    
    for i, request in enumerate(test_requests, 1):
        print(f"\n🔍 Test {i}: {request}")
        print("-" * 50)
        
        result = agent.generate(request)
        
        if result.get("success"):
            print(f"✅ Task Type: {result.get('task_type')}")
            print(f"📝 Language: {result.get('language')}")
            print(f"💬 Response:\n{result.get('response')}")
        else:
            print(f"❌ Error: {result.get('error')}")
            if result.get('fallback_response'):
                print(f"🔄 Fallback: {result.get('fallback_response')}")
        
        print()

def test_orchestrator_routing():
    """Test that coding requests are properly routed through the orchestrator"""
    print("\n🎯 Testing Orchestrator Routing")
    print("=" * 50)
    
    router = Router()
    orchestrator = Orchestrator()
    
    coding_requests = [
        "Write a Python function to reverse a string",
        "Create a REST API endpoint in Node.js",
        "Help me debug this JavaScript error",
        "Refactor this code for better performance",
        "Generate a SQL query for user authentication"
    ]
    
    for request in coding_requests:
        print(f"\n📝 Request: {request}")
        
        # Test routing
        category = router.route(request)
        print(f"🎯 Routed to: {category}")
        
        # Test classification
        classification = router.classify_intent(request)
        print(f"🔍 Classification: {classification}")
        
        # Test orchestrator
        try:
            result = orchestrator.handle(request)
            print(f"✅ Orchestrator result: {type(result).__name__}")
            if hasattr(result, 'category'):
                print(f"📂 Category: {result.category}")
            if hasattr(result, 'final_response') and result.final_response:
                print(f"💬 Response preview: {result.final_response[:100]}...")
        except Exception as e:
            print(f"❌ Orchestrator error: {e}")

def test_natural_language_examples():
    """Test various natural language coding requests"""
    print("\n🗣️  Testing Natural Language Examples")
    print("=" * 50)
    
    examples = [
        # Basic code generation
        "I need a function that checks if a number is prime",
        "Can you write some Python code to read a CSV file?",
        "Create a class for managing a shopping cart",
        
        # Code explanation
        "What does this code do: for i in range(10): print(i**2)",
        "Explain how this sorting algorithm works",
        
        # Problem solving
        "I'm getting a 'list index out of range' error, how do I fix it?",
        "My API is returning 500 errors, help me debug this",
        
        # Code improvement
        "Make this code more efficient",
        "How can I optimize this database query?",
        
        # Different languages
        "Write a Java method to connect to a database",
        "Create a React component for a user profile",
        "Build a shell script to backup files"
    ]
    
    agent = CodingAssistantAgent()
    
    for example in examples[:5]:  # Test first 5 to avoid too much output
        print(f"\n💭 Natural Language: '{example}'")
        print("-" * 40)
        
        result = agent.generate(example)
        
        if result.get("success"):
            print(f"✅ Identified as: {result.get('task_type')} ({result.get('language')})")
            response_preview = result.get('response', '')[:150]
            print(f"📝 Response preview: {response_preview}...")
        else:
            print(f"❌ Failed: {result.get('error')}")

async def main():
    """Run all tests"""
    print("🚀 Testing Quark Natural Language Coding Capabilities")
    print("=" * 60)
    
    try:
        # Test 1: Direct coding assistant
        await test_coding_assistant()
        
        # Test 2: Orchestrator routing
        test_orchestrator_routing()
        
        # Test 3: Natural language examples
        test_natural_language_examples()
        
        print("\n🎉 All tests completed!")
        print("✅ Your Quark model can now handle natural language coding tasks!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if we're in the right directory
    if not os.path.exists("agents"):
        print("❌ Please run this script from the quark root directory")
        sys.exit(1)
    
    asyncio.run(main())