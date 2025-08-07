#!/usr/bin/env python3
"""
Live Password Demo
=================

This script demonstrates the actual password prompt interface
for root override requests.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.system_protection import SystemProtection, set_root_password, request_root_override


def live_password_demo():
    """Demonstrate the live password prompt interface."""
    
    print("🔐 LIVE PASSWORD OVERRIDE DEMO")
    print("=" * 50)
    print()
    print("This demo will show you the actual password prompt interface.")
    print("The password for this demo is: 'demo123'")
    print()
    
    # Initialize system protection
    protection = SystemProtection()
    
    # Set root password
    set_root_password("demo123")
    
    print("✅ Root password set: 'demo123'")
    print()
    
    # Demo 1: File deletion with specific files
    print("🔍 DEMO 1: File Deletion Override")
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
    
    print("Requesting override for file deletion...")
    print("(You'll see the password prompt below)")
    print()
    
    # This will show the actual password prompt
    result = request_root_override(action, context)
    
    print()
    print(f"Override result: {'✅ GRANTED' if result['success'] else '❌ DENIED'}")
    if result['success']:
        print(f"Session ID: {result['session_id']}")
        print(f"Expires at: {result['expires_at']}")
    else:
        print(f"Message: {result['message']}")
    
    print()
    print("-" * 50)
    print()
    
    # Demo 2: System command execution
    print("🔍 DEMO 2: System Command Override")
    print("-" * 40)
    
    action = "execute system commands"
    context = {
        "commands": [
            "sudo apt update",
            "sudo systemctl restart nginx"
        ],
        "reason": "System maintenance",
        "impact": "System packages will be updated and nginx restarted"
    }
    
    print("Requesting override for system commands...")
    print("(You'll see the password prompt below)")
    print()
    
    # This will show the actual password prompt
    result = request_root_override(action, context)
    
    print()
    print(f"Override result: {'✅ GRANTED' if result['success'] else '❌ DENIED'}")
    if result['success']:
        print(f"Session ID: {result['session_id']}")
        print(f"Expires at: {result['expires_at']}")
    else:
        print(f"Message: {result['message']}")
    
    print()
    print("✅ Live password demo complete!")
    print()
    print("🔒 Demo Features Shown:")
    print("   - Detailed action context display")
    print("   - Secure password prompt (getpass)")
    print("   - Password verification")
    print("   - Session creation and management")
    print("   - Success/failure feedback")


if __name__ == "__main__":
    live_password_demo() 