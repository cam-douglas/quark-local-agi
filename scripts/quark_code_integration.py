#!/usr/bin/env python3
"""
Integration script to add code generation to the global quark command
"""

import os
import sys
from pathlib import Path

def update_quark_shell():
    """Update the quark shell script to include code generation"""
    quark_shell_path = Path(__file__).parent / "quark_shell.sh"
    
    if not quark_shell_path.exists():
        print("❌ quark_shell.sh not found")
        return False
    
    # Read current content
    with open(quark_shell_path, 'r') as f:
        content = f.read()
    
    # Check if code generation is already integrated
    if "code generation" in content.lower():
        print("✅ Code generation already integrated")
        return True
    
    # Add code generation command handling
    integration_code = '''
# Handle code generation commands
if [ "$1" = "code" ]; then
    shift
    exec python3 scripts/quark_code.py "$@"
fi

if [ "$1" = "complete" ]; then
    shift
    exec python3 scripts/quark_code.py "complete: $*"
fi

if [ "$1" = "generate" ]; then
    shift
    exec python3 scripts/quark_code.py "generate: $*"
fi

if [ "$1" = "explain" ]; then
    shift
    exec python3 scripts/quark_code.py "explain: $*"
fi

if [ "$1" = "refactor" ]; then
    shift
    exec python3 scripts/quark_code.py "refactor: $*"
fi

'''
    
    # Find the right place to insert (before the main execution)
    lines = content.split('\n')
    insert_index = -1
    
    for i, line in enumerate(lines):
        if 'exec python3 scripts/quark_chat.py' in line:
            insert_index = i
            break
    
    if insert_index > 0:
        lines.insert(insert_index, integration_code)
        new_content = '\n'.join(lines)
        
        # Write back
        with open(quark_shell_path, 'w') as f:
            f.write(new_content)
        
        print("✅ Code generation integrated into quark shell")
        return True
    else:
        print("❌ Could not find insertion point in quark_shell.sh")
        return False

def create_help_text():
    """Create help text for code generation commands"""
    help_text = """
Code Generation Commands:
─────────────────────────────────────────────────────────
quark code                    - Interactive code assistant
quark complete "def func():"  - Complete partial code
quark generate "description"  - Generate code from description
quark explain "code"          - Explain existing code  
quark refactor "code"         - Refactor code for improvement

Examples:
─────────────────────────────────────────────────────────
quark complete "def fibonacci(n):"
quark generate "create a function to sort a list"
quark explain "print('hello world')"
quark refactor "for i in range(len(arr)): print(arr[i])"
"""
    
    help_file = Path(__file__).parent.parent / "docs" / "CODE_GENERATION_HELP.md"
    with open(help_file, 'w') as f:
        f.write(help_text)
    
    print(f"✅ Help documentation created: {help_file}")

def main():
    """Main integration function"""
    print("🔗 Integrating Code Generation with Quark")
    print("=" * 50)
    
    # Update shell script
    if update_quark_shell():
        print("✅ Shell integration successful")
    else:
        print("❌ Shell integration failed")
        return False
    
    # Create help documentation
    create_help_text()
    
    print("\n🎉 Code Generation Integration Complete!")
    print("\nYou can now use:")
    print("  quark code                     - Interactive assistant")  
    print("  quark complete 'def func():'   - Complete code")
    print("  quark generate 'description'   - Generate code")
    print("  quark explain 'code'          - Explain code")
    print("  quark refactor 'code'         - Refactor code")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)