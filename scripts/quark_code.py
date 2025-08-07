#!/usr/bin/env python3
"""
Quark Code Assistant
Simple script for code generation, completion, and assistance

Usage:
  python3 scripts/quark_code.py "complete: def fibonacci(n):"
  python3 scripts/quark_code.py "generate: create a sorting function"
  python3 scripts/quark_code.py "explain: print('hello world')"
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def print_code_result(result: str, task_type: str = "Code"):
    """Print code result with simple formatting"""
    print(f"\n{'='*60}")
    print(f"📝 {task_type} Result:")
    print('='*60)
    print(result)
    print('='*60)

async def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("🤖 Quark Code Assistant")
        print("\nUsage:")
        print("  python3 scripts/quark_code.py 'complete: def fibonacci(n):'")
        print("  python3 scripts/quark_code.py 'generate: create a sorting function'")
        print("  python3 scripts/quark_code.py 'explain: print(\"hello world\")'")
        print("  python3 scripts/quark_code.py 'refactor: [your code here]'")
        print("\nShortcuts:")
        print("  python3 scripts/quark_code.py 'def fibonacci(n):'  (auto-complete)")
        print("  python3 scripts/quark_code.py 'create a function to sort a list'  (auto-generate)")
        return
    
    user_input = " ".join(sys.argv[1:])
    
    try:
        from agents.code_generation_agent import CodeGenerationAgent
        
        print("🤖 Initializing Quark Code Assistant...")
        agent = CodeGenerationAgent()
        
        # Parse input
        if user_input.startswith("complete:"):
            task = "complete"
            code = user_input[9:].strip()
        elif user_input.startswith("generate:"):
            task = "generate" 
            code = user_input[9:].strip()
        elif user_input.startswith("explain:"):
            task = "explain"
            code = user_input[8:].strip()
        elif user_input.startswith("refactor:"):
            task = "refactor"
            code = user_input[9:].strip()
        else:
            # Auto-detect based on content
            if user_input.strip().startswith(("def ", "class ", "import ", "from ", "if ", "for ", "while ")):
                task = "complete"
                code = user_input
            else:
                task = "generate"
                code = user_input
        
        print(f"🔄 Processing {task} request...")
        
        # Execute the task
        if task == "complete":
            result = await agent.complete_code(code, "python")
            print_code_result(result, "Code Completion")
        elif task == "generate":
            result = await agent.generate_from_description(code, "python")
            print_code_result(result, "Generated Code")
        elif task == "explain":
            result = await agent.get_code_explanation(code, "python")
            print_code_result(result, "Code Explanation")
        elif task == "refactor":
            result = await agent.refactor_code(code, "python")
            print_code_result(result, "Refactored Code")
        
        print("\n✅ Task completed successfully!")
        
    except ImportError as e:
        print(f"❌ Error: Missing dependencies. {e}")
        print("Please ensure all required packages are installed.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())