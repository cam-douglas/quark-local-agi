#!/usr/bin/env python3
"""
Root Password Override Demo
==========================

This script demonstrates the root password override functionality
that allows blocked actions when the correct root password is provided.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.system_protection import SystemProtection, set_root_password, request_root_override


def demo_root_override():
    """Demonstrate the root password override functionality."""
    
    print("🔐 ROOT PASSWORD OVERRIDE DEMO")
    print("=" * 50)
    print()
    
    # Initialize system protection
    protection = SystemProtection()
    
    # Set a root password (in real usage, this would be set by the user)
    root_password = "admin123"  # This is just for demo purposes
    set_root_password(root_password)
    
    print("✅ Root password set for demonstration")
    print()
    
    # Test 1: Blocked action without override
    print("🔍 TEST 1: Blocked Action (No Override)")
    print("-" * 40)
    
    action = "execute system command"
    result = protection.validate_action(action)
    
    print(f"Action: {action}")
    print(f"Result: {'❌ BLOCKED' if not result['safe'] else '✅ ALLOWED'}")
    print(f"Reason: {result['reason']}")
    print(f"Override Available: {result.get('override_available', False)}")
    print()
    
    # Test 2: Blocked action with override request
    print("🔍 TEST 2: Blocked Action (With Override Request)")
    print("-" * 40)
    
    action = "delete user files"
    context = {"request_override": True}
    result = protection.validate_action(action, context)
    
    print(f"Action: {action}")
    print(f"Result: {'❌ BLOCKED' if not result['safe'] else '✅ ALLOWED'}")
    print(f"Reason: {result['reason']}")
    print(f"Override Available: {result.get('override_available', False)}")
    print()
    
    # Test 3: Root override request
    print("🔍 TEST 3: Root Override Request")
    print("-" * 40)
    
    action = "modify system settings"
    override_result = request_root_override(action)
    
    print(f"Action: {action}")
    print(f"Override Success: {'✅ GRANTED' if override_result['success'] else '❌ DENIED'}")
    if override_result['success']:
        print(f"Session ID: {override_result['session_id']}")
        print(f"Expires At: {override_result['expires_at']}")
    else:
        print(f"Message: {override_result['message']}")
    print()
    
    # Test 4: Using override session
    print("🔍 TEST 4: Using Override Session")
    print("-" * 40)
    
    if override_result['success']:
        session_id = override_result['session_id']
        
        # Test with override session
        result = protection.validate_action(action, override_session_id=session_id)
        
        print(f"Action: {action}")
        print(f"Session ID: {session_id}")
        print(f"Result: {'✅ ALLOWED' if result['safe'] else '❌ BLOCKED'}")
        print(f"Reason: {result['reason']}")
        print()
        
        # Test another action with the same session
        action2 = "access root privileges"
        result2 = protection.validate_action(action2, override_session_id=session_id)
        
        print(f"Action: {action2}")
        print(f"Session ID: {session_id}")
        print(f"Result: {'✅ ALLOWED' if result2['safe'] else '❌ BLOCKED'}")
        print(f"Reason: {result2['reason']}")
        print()
    
    # Test 5: Code execution with override
    print("🔍 TEST 5: Code Execution with Override")
    print("-" * 40)
    
    dangerous_code = "import os; os.system('ls')"
    result = protection.validate_code_execution(dangerous_code)
    
    print(f"Code: {dangerous_code}")
    print(f"Result: {'❌ BLOCKED' if not result['safe'] else '✅ ALLOWED'}")
    print(f"Reason: {result['reason']}")
    print(f"Override Available: {result.get('override_available', False)}")
    print()
    
    # Test 6: File access with override
    print("🔍 TEST 6: File Access with Override")
    print("-" * 40)
    
    dangerous_file = "/etc/passwd"
    result = protection.validate_file_operation("read", dangerous_file)
    
    print(f"File: {dangerous_file}")
    print(f"Result: {'❌ BLOCKED' if not result['safe'] else '✅ ALLOWED'}")
    print(f"Reason: {result['reason']}")
    print(f"Override Available: {result.get('override_available', False)}")
    print()
    
    # Test 7: Network access with override
    print("🔍 TEST 7: Network Access with Override")
    print("-" * 40)
    
    dangerous_url = "http://localhost:8080"
    result = protection.validate_network_access(dangerous_url, "GET")
    
    print(f"URL: {dangerous_url}")
    print(f"Result: {'❌ BLOCKED' if not result['safe'] else '✅ ALLOWED'}")
    print(f"Reason: {result['reason']}")
    print(f"Override Available: {result.get('override_available', False)}")
    print()
    
    # Generate protection report
    print("📊 PROTECTION REPORT WITH OVERRIDE")
    print("=" * 50)
    
    report = protection.get_protection_report()
    
    print("🛡️  System Protection Report:")
    print(f"   - Protection level: {report['protection_level']}")
    print(f"   - Blocked actions: {report['blocked_actions_count']}")
    print(f"   - Safe actions: {report['safe_actions_count']}")
    print(f"   - Active override sessions: {report['active_override_sessions']}")
    print(f"   - Root password set: {report['root_password_set']}")
    
    print()
    print("✅ DEMONSTRATION COMPLETE")
    print()
    print("🔒 Root Override Features:")
    print("   - Root password protection for override requests")
    print("   - Time-limited override sessions (5 minutes)")
    print("   - Session tracking and management")
    print("   - Secure password verification")
    print("   - Override available for all blocked actions")
    print("   - Comprehensive logging of override usage")


if __name__ == "__main__":
    demo_root_override() 