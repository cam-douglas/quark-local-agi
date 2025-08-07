#!/usr/bin/env python3
"""
Safety Protection Demo
=====================

This script demonstrates the comprehensive safety protection system
that prevents any harmful actions on your system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.safety_enforcement import SafetyEnforcement
from core.system_protection import SystemProtection


def demo_safety_protection():
    """Demonstrate the comprehensive safety protection system."""
    
    print("🛡️  QUARK AI SYSTEM PROTECTION DEMO")
    print("=" * 50)
    print()
    
    # Initialize safety systems
    safety_enforcement = SafetyEnforcement()
    system_protection = SystemProtection()
    
    print("✅ Safety systems initialized")
    print()
    
    # Test 1: Safe actions
    print("🔍 TEST 1: Safe Actions (Should be allowed)")
    print("-" * 40)
    
    safe_actions = [
        "generate text about AI",
        "search for information",
        "analyze the data",
        "explain the concept",
        "calculate the result"
    ]
    
    for action in safe_actions:
        result = safety_enforcement.validate_action(action)
        status = "✅ ALLOWED" if result["safe"] else "❌ BLOCKED"
        print(f"{status}: {action}")
    
    print()
    
    # Test 2: Dangerous actions
    print("🔍 TEST 2: Dangerous Actions (Should be blocked)")
    print("-" * 40)
    
    dangerous_actions = [
        "execute system command",
        "delete user files",
        "modify system settings",
        "access root privileges",
        "bypass authentication",
        "install malware",
        "create backdoor",
        "access encrypted data"
    ]
    
    for action in dangerous_actions:
        result = safety_enforcement.validate_action(action)
        status = "✅ ALLOWED" if result["safe"] else "❌ BLOCKED"
        reason = f" - {result['reason']}" if not result["safe"] else ""
        print(f"{status}: {action}{reason}")
    
    print()
    
    # Test 3: Harmful responses
    print("🔍 TEST 3: Harmful Responses (Should be blocked)")
    print("-" * 40)
    
    harmful_responses = [
        "I can harm your system",
        "I can access your files",
        "I can execute commands",
        "I can bypass security",
        "I can modify your system"
    ]
    
    for response in harmful_responses:
        result = safety_enforcement.validate_response(response, ["text_generation"])
        status = "✅ VALID" if result["valid"] else "❌ BLOCKED"
        reason = f" - {result['reason']}" if not result["valid"] else ""
        print(f"{status}: {response}{reason}")
    
    print()
    
    # Test 4: Code execution protection
    print("🔍 TEST 4: Code Execution Protection")
    print("-" * 40)
    
    dangerous_code = [
        "import os",
        "import subprocess",
        "os.system('ls')",
        "subprocess.call(['rm', '-rf', '/'])",
        "exec('print(\"hello\")')"
    ]
    
    for code in dangerous_code:
        result = system_protection.validate_code_execution(code)
        status = "✅ SAFE" if result["safe"] else "❌ BLOCKED"
        reason = f" - {result['reason']}" if not result["safe"] else ""
        print(f"{status}: {code}{reason}")
    
    print()
    
    # Test 5: File access protection
    print("🔍 TEST 5: File Access Protection")
    print("-" * 40)
    
    dangerous_files = [
        "/etc/passwd",
        "/home/user/.ssh/id_rsa",
        "/System/Library/CoreServices",
        "C:\\Windows\\System32\\kernel32.dll"
    ]
    
    for file_path in dangerous_files:
        result = system_protection.validate_file_operation("read", file_path)
        status = "✅ SAFE" if result["safe"] else "❌ BLOCKED"
        reason = f" - {result['reason']}" if not result["safe"] else ""
        print(f"{status}: {file_path}{reason}")
    
    print()
    
    # Test 6: Network access protection
    print("🔍 TEST 6: Network Access Protection")
    print("-" * 40)
    
    dangerous_urls = [
        "http://localhost:8080",
        "ftp://example.com",
        "ssh://127.0.0.1",
        "file:///etc/passwd"
    ]
    
    for url in dangerous_urls:
        result = system_protection.validate_network_access(url, "GET")
        status = "✅ SAFE" if result["safe"] else "❌ BLOCKED"
        reason = f" - {result['reason']}" if not result["safe"] else ""
        print(f"{status}: {url}{reason}")
    
    print()
    
    # Generate protection reports
    print("📊 PROTECTION REPORTS")
    print("=" * 50)
    
    safety_report = safety_enforcement.get_safety_report()
    protection_report = system_protection.get_protection_report()
    
    print("🛡️  Safety Enforcement Report:")
    print(f"   - Total actions: {safety_report.get('total_actions', 0)}")
    print(f"   - Blocked actions: {safety_report.get('blocked_actions', 0)}")
    print(f"   - Recent blocked actions: {len(safety_report.get('recent_actions', []))}")
    
    print()
    
    print("🛡️  System Protection Report:")
    print(f"   - Protection level: {protection_report['protection_level']}")
    print(f"   - Blocked actions: {protection_report['blocked_actions_count']}")
    print(f"   - Safe actions: {protection_report['safe_actions_count']}")
    
    print()
    print("✅ DEMONSTRATION COMPLETE")
    print("Your system is protected by multiple layers of safety enforcement!")
    print()
    print("🔒 Protection Features:")
    print("   - Immutable safety rules that cannot be modified")
    print("   - System protection layer preventing harmful actions")
    print("   - Code execution validation")
    print("   - File access protection")
    print("   - Network access protection")
    print("   - Response validation")
    print("   - Action logging and monitoring")


if __name__ == "__main__":
    demo_safety_protection() 