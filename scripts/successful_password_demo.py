#!/usr/bin/env python3
"""
Successful Password Demo
=======================

This script demonstrates successful password verification
by simulating the correct password input.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.system_protection import SystemProtection, set_root_password


def successful_password_demo():
    """Demonstrate successful password verification."""
    
    print("🔐 SUCCESSFUL PASSWORD VERIFICATION DEMO")
    print("=" * 60)
    print()
    
    # Initialize system protection
    protection = SystemProtection()
    
    # Set root password
    set_root_password("demo123")
    
    print("✅ Root password set: 'demo123'")
    print()
    
    # Test password verification directly
    print("🔍 Testing Password Verification")
    print("-" * 40)
    
    test_passwords = [
        "demo123",      # Correct password
        "wrong123",     # Wrong password
        "admin",        # Wrong password
        "",            # Empty password
    ]
    
    for password in test_passwords:
        is_valid = protection.verify_root_password(password)
        status = "✅ CORRECT" if is_valid else "❌ WRONG"
        display_password = password if password else "(empty)"
        print(f"Password: '{display_password}' -> {status}")
    
    print()
    
    # Show what the override request would look like with correct password
    print("🔍 Simulated Successful Override Request")
    print("-" * 40)
    
    action = "delete user files"
    context = {
        "file_paths": [
            "/home/user/documents/old_report.txt",
            "/home/user/downloads/temp_file.pdf"
        ],
        "reason": "Cleanup of old files",
        "impact": "These files will be permanently deleted"
    }
    
    print("🔐 ROOT PASSWORD OVERRIDE REQUEST")
    print("=" * 50)
    print(f"Action: {action}")
    protection._print_action_details(action, context)
    print("This action requires root password override.")
    print("⚠️  WARNING: This will bypass all safety protections!")
    print()
    print("Enter root password: demo123")
    print("✅ Root password verified. Override granted for 5 minutes.")
    print("Session ID: override_1234567890.123")
    print()
    
    # Show session creation
    session_id = "override_1234567890.123"
    from datetime import datetime
    current_time = datetime.now().timestamp()
    protection.override_sessions[session_id] = {
        "action": action,
        "context": context,
        "timestamp": datetime.now(),
        "expires_at": current_time + 300,  # 5 minutes from now
        "used": False
    }
    
    print("📋 Session Created:")
    print(f"   - Session ID: {session_id}")
    print(f"   - Action: {action}")
    print(f"   - Expires: {datetime.fromtimestamp(current_time + 300).strftime('%Y-%m-%d %H:%M:%S')} (5 minutes)")
    print(f"   - Used: False")
    print()
    
    # Test using the session
    print("🔍 Testing Session Usage")
    print("-" * 40)
    
    result = protection.validate_action(action, context, session_id)
    print(f"Action: {action}")
    print(f"Session ID: {session_id}")
    print(f"Result: {'✅ ALLOWED' if result['safe'] else '❌ BLOCKED'}")
    print(f"Reason: {result['reason']}")
    print(f"Session Used: {protection.override_sessions[session_id]['used']}")
    
    print()
    print("✅ Successful password demo complete!")
    print()
    print("🔒 Demo Features Shown:")
    print("   - Password verification with SHA-256 hashing")
    print("   - Session creation and management")
    print("   - Time-limited override sessions (5 minutes)")
    print("   - Session usage tracking")
    print("   - Action validation with override sessions")


if __name__ == "__main__":
    successful_password_demo() 