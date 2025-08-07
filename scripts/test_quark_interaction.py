#!/usr/bin/env python3
"""
Test Quark Interaction
Simple script to test Quark's response handling
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.orchestrator import Orchestrator

def test_quark_response():
    """Test Quark's response handling"""
    print("🧪 Testing Quark interaction...")
    
    try:
        # Initialize orchestrator
        orchestrator = Orchestrator()
        
        # Test questions
        test_questions = [
            "how are you?",
            "what is 2+2?",
            "tell me a joke",
            "what is the weather like?"
        ]
        
        for question in test_questions:
            print(f"\n🤖 User: {question}")
            
            try:
                result = orchestrator.handle(question)
                
                if hasattr(result, 'final_response'):
                    response = result.final_response
                elif hasattr(result, 'response'):
                    response = result.response
                else:
                    response = str(result)
                
                print(f"🤖 Quark: {response}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n✅ Test completed!")
        
    except Exception as e:
        print(f"❌ Failed to initialize orchestrator: {e}")

if __name__ == "__main__":
    test_quark_response() 