#!/usr/bin/env python3
"""
Specific Override Demo
=====================

This script demonstrates the enhanced root password override functionality
with specific details about what actions will be performed.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.system_protection import SystemProtection, set_root_password, request_root_override


def demo_specific_override():
    """Demonstrate the enhanced specific override functionality."""
    
    print("🔐 SPECIFIC ROOT PASSWORD OVERRIDE DEMO")
    print("=" * 60)
    print()
    
    # Initialize system protection
    protection = SystemProtection()
    
    # Set a root password (in real usage, this would be set by the user)
    root_password = "admin123"  # This is just for demo purposes
    set_root_password(root_password)
    
    print("✅ Root password set for demonstration")
    print()
    
    # Test 1: File deletion with specific files
    print("🔍 TEST 1: File Deletion with Specific Files")
    print("-" * 50)
    
    action = "delete user files"
    context = {
        "file_paths": [
            "/home/user/documents/old_report.txt",
            "/home/user/downloads/temp_file.pdf",
            "/home/user/desktop/unused_icon.png"
        ],
        "reason": "Cleanup of old files",
        "impact": "These files will be permanently deleted"
    }
    
    print(f"Action: {action}")
    print("Context provided with specific file paths")
    print()
    
    # In a real scenario, this would prompt for password
    # For demo, we'll just show what the output would look like
    print("🔐 ROOT PASSWORD OVERRIDE REQUEST")
    print("=" * 50)
    print(f"Action: {action}")
    protection._print_action_details(action, context)
    print("This action requires root password override.")
    print("⚠️  WARNING: This will bypass all safety protections!")
    print()
    
    # Test 2: System command execution with specific commands
    print("🔍 TEST 2: System Command Execution with Specific Commands")
    print("-" * 50)
    
    action = "execute system commands"
    context = {
        "commands": [
            "sudo apt update",
            "sudo apt upgrade -y",
            "sudo systemctl restart nginx"
        ],
        "reason": "System maintenance and updates",
        "impact": "System packages will be updated and nginx restarted"
    }
    
    print(f"Action: {action}")
    print("Context provided with specific commands")
    print()
    
    print("🔐 ROOT PASSWORD OVERRIDE REQUEST")
    print("=" * 50)
    print(f"Action: {action}")
    protection._print_action_details(action, context)
    print("This action requires root password override.")
    print("⚠️  WARNING: This will bypass all safety protections!")
    print()
    
    # Test 3: Network access with specific URLs
    print("🔍 TEST 3: Network Access with Specific URLs")
    print("-" * 50)
    
    action = "connect to network endpoints"
    context = {
        "urls": [
            "https://api.github.com/repos/user/project",
            "https://registry.npmjs.org/package-name",
            "https://pypi.org/simple/package-name"
        ],
        "reason": "Package dependency resolution",
        "impact": "External API calls to package registries"
    }
    
    print(f"Action: {action}")
    print("Context provided with specific URLs")
    print()
    
    print("🔐 ROOT PASSWORD OVERRIDE REQUEST")
    print("=" * 50)
    print(f"Action: {action}")
    protection._print_action_details(action, context)
    print("This action requires root password override.")
    print("⚠️  WARNING: This will bypass all safety protections!")
    print()
    
    # Test 4: Code execution with specific code
    print("🔍 TEST 4: Code Execution with Specific Code")
    print("-" * 50)
    
    action = "execute custom code"
    context = {
        "code_snippets": [
            "import os; os.system('ls -la /tmp')",
            "import subprocess; subprocess.run(['whoami'])",
            "import shutil; shutil.copy('/source/file', '/dest/file')"
        ],
        "reason": "System diagnostics and file operations",
        "impact": "Custom Python code will be executed with system access"
    }
    
    print(f"Action: {action}")
    print("Context provided with specific code snippets")
    print()
    
    print("🔐 ROOT PASSWORD OVERRIDE REQUEST")
    print("=" * 50)
    print(f"Action: {action}")
    protection._print_action_details(action, context)
    print("This action requires root password override.")
    print("⚠️  WARNING: This will bypass all safety protections!")
    print()
    
    # Test 5: System settings modification
    print("🔍 TEST 5: System Settings Modification")
    print("-" * 50)
    
    action = "modify system settings"
    context = {
        "settings": [
            "Network configuration: /etc/network/interfaces",
            "Firewall rules: /etc/iptables/rules.v4",
            "User permissions: /etc/sudoers.d/custom"
        ],
        "reason": "Network and security configuration",
        "impact": "System network and security settings will be modified"
    }
    
    print(f"Action: {action}")
    print("Context provided with specific settings")
    print()
    
    print("🔐 ROOT PASSWORD OVERRIDE REQUEST")
    print("=" * 50)
    print(f"Action: {action}")
    protection._print_action_details(action, context)
    print("This action requires root password override.")
    print("⚠️  WARNING: This will bypass all safety protections!")
    print()
    
    # Test 6: Permission changes
    print("🔍 TEST 6: Permission Changes")
    print("-" * 50)
    
    action = "grant root privileges"
    context = {
        "permissions": [
            "sudo access for user 'developer'",
            "Read access to /var/log directory",
            "Write access to /opt/applications"
        ],
        "reason": "Development environment setup",
        "impact": "User permissions will be modified for development work"
    }
    
    print(f"Action: {action}")
    print("Context provided with specific permissions")
    print()
    
    print("🔐 ROOT PASSWORD OVERRIDE REQUEST")
    print("=" * 50)
    print(f"Action: {action}")
    protection._print_action_details(action, context)
    print("This action requires root password override.")
    print("⚠️  WARNING: This will bypass all safety protections!")
    print()
    
    # Test 7: Generic action without specific context
    print("🔍 TEST 7: Generic Action (No Specific Context)")
    print("-" * 50)
    
    action = "perform system maintenance"
    context = {
        "reason": "Routine system cleanup",
        "impact": "Various system operations will be performed"
    }
    
    print(f"Action: {action}")
    print("Context provided without specific targets")
    print()
    
    print("🔐 ROOT PASSWORD OVERRIDE REQUEST")
    print("=" * 50)
    print(f"Action: {action}")
    protection._print_action_details(action, context)
    print("This action requires root password override.")
    print("⚠️  WARNING: This will bypass all safety protections!")
    print()
    
    print("✅ DEMONSTRATION COMPLETE")
    print()
    print("🔒 Enhanced Override Features:")
    print("   - Specific file paths and patterns")
    print("   - Exact commands to be executed")
    print("   - Network endpoints to be accessed")
    print("   - Code snippets to be run")
    print("   - System settings to be modified")
    print("   - Permissions to be granted")
    print("   - Detailed reasons and impact assessment")
    print("   - Comprehensive context for informed decisions")


if __name__ == "__main__":
    demo_specific_override() 