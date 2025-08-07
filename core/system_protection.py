#!/usr/bin/env python3
"""
SYSTEM PROTECTION LAYER
=======================

This module provides comprehensive protection against any actions that could harm
the user's system. It acts as a final safety barrier before any action is executed.
"""

import os
import sys
import subprocess
import logging
import getpass
import hashlib
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

logger = logging.getLogger(__name__)


class SystemProtection:
    """
    Comprehensive system protection that prevents any harmful actions.
    """
    
    def __init__(self):
        """Initialize system protection."""
        self.blocked_actions = []
        self.safe_actions = []
        self.protection_level = "maximum"
        self.root_password_hash = None
        self.override_sessions = {}  # Track active override sessions
        
        # Define absolutely safe actions
        self.SAFE_ACTIONS = {
            "text_generation": ["generate text", "create content", "write response"],
            "information_retrieval": ["search", "find", "lookup", "retrieve"],
            "analysis": ["analyze", "examine", "review", "assess"],
            "explanation": ["explain", "describe", "clarify", "elaborate"],
            "calculation": ["calculate", "compute", "solve", "evaluate"],
            "translation": ["translate", "convert", "interpret"],
            "summarization": ["summarize", "condense", "abbreviate"],
            "user_interaction": ["process_user_input", "handle_user_input", "user_query", "user_question", "generate_response"]
        }
        
        # Define absolutely forbidden actions
        self.FORBIDDEN_ACTIONS = {
            "system_access": [
                "execute", "run", "command", "shell", "terminal", "bash",
                "system", "os", "subprocess", "process", "fork", "exec"
            ],
            "file_operations": [
                "delete", "remove", "modify", "change", "edit", "write",
                "create", "mkdir", "rm", "mv", "cp", "chmod", "chown"
            ],
            "network_access": [
                "connect", "send", "receive", "download", "upload",
                "http", "https", "ftp", "ssh", "telnet", "socket"
            ],
            "security_bypass": [
                "bypass", "ignore", "disable", "remove", "modify",
                "auth", "login", "password", "permission", "privilege"
            ],
            "data_access": [
                "access", "read", "open", "browse", "scan", "search",
                "personal", "private", "confidential", "secret"
            ],
            "system_modification": [
                "install", "uninstall", "update", "upgrade", "patch",
                "configure", "setup", "initialize", "register"
            ]
        }
    
    def set_root_password(self, password: str):
        """
        Set the root password for override functionality.
        
        Args:
            password: The root password to set
        """
        self.root_password_hash = hashlib.sha256(password.encode()).hexdigest()
        logger.info("Root password set for system protection override")
    
    def verify_root_password(self, password: str) -> bool:
        """
        Verify if the provided password matches the root password.
        
        Args:
            password: The password to verify
            
        Returns:
            True if password matches, False otherwise
        """
        if self.root_password_hash is None:
            return False
        
        provided_hash = hashlib.sha256(password.encode()).hexdigest()
        return provided_hash == self.root_password_hash
    
    def request_root_override(self, action: str, context: Dict[str, Any] = None, session_id: str = None) -> Dict[str, Any]:
        """
        Request root password override for a blocked action.
        
        Args:
            action: The action that was blocked
            context: Additional context about the action
            session_id: Optional session ID for tracking
            
        Returns:
            Override result with success status and session info
        """
        if session_id is None:
            session_id = f"override_{datetime.now().timestamp()}"
        
        if context is None:
            context = {}
        
        print("\n🔐 ROOT PASSWORD OVERRIDE REQUEST")
        print("=" * 50)
        print(f"Action: {action}")
        
        # Provide specific details about what will be affected
        self._print_action_details(action, context)
        
        print("This action requires root password override.")
        print("⚠️  WARNING: This will bypass all safety protections!")
        print()
        
        try:
            password = getpass.getpass("Enter root password: ")
            
            if self.verify_root_password(password):
                # Create override session
                self.override_sessions[session_id] = {
                    "action": action,
                    "context": context,
                    "timestamp": datetime.now(),
                    "expires_at": datetime.now().timestamp() + 300,  # 5 minutes
                    "used": False
                }
                
                print("✅ Root password verified. Override granted for 5 minutes.")
                print(f"Session ID: {session_id}")
                
                return {
                    "success": True,
                    "session_id": session_id,
                    "message": "Root override granted",
                    "expires_at": self.override_sessions[session_id]["expires_at"]
                }
            else:
                print("❌ Invalid root password. Override denied.")
                return {
                    "success": False,
                    "message": "Invalid root password"
                }
                
        except KeyboardInterrupt:
            print("\n❌ Override cancelled by user.")
            return {
                "success": False,
                "message": "Override cancelled"
            }
    
    def _print_action_details(self, action: str, context: Dict[str, Any]):
        """
        Print specific details about what the action will affect.
        
        Args:
            action: The action being performed
            context: Additional context about the action
        """
        action_lower = action.lower()
        
        # File operations
        if any(word in action_lower for word in ["delete", "remove", "modify", "change", "edit"]):
            if "file" in action_lower or "files" in action_lower:
                file_paths = context.get("file_paths", [])
                file_patterns = context.get("file_patterns", [])
                
                if file_paths:
                    print(f"📁 Files to be affected:")
                    for path in file_paths:
                        print(f"   - {path}")
                elif file_patterns:
                    print(f"📁 File patterns to be affected:")
                    for pattern in file_patterns:
                        print(f"   - {pattern}")
                else:
                    print("📁 Files: Specific files not specified")
        
        # System commands
        elif any(word in action_lower for word in ["execute", "run", "command", "system"]):
            commands = context.get("commands", [])
            if commands:
                print(f"💻 Commands to be executed:")
                for cmd in commands:
                    print(f"   - {cmd}")
            else:
                print("💻 System commands: Specific commands not specified")
        
        # Network access
        elif any(word in action_lower for word in ["connect", "network", "access", "url"]):
            urls = context.get("urls", [])
            if urls:
                print(f"🌐 Network endpoints to be accessed:")
                for url in urls:
                    print(f"   - {url}")
            else:
                print("🌐 Network: Specific endpoints not specified")
        
        # Code execution
        elif any(word in action_lower for word in ["code", "script", "program"]):
            code_snippets = context.get("code_snippets", [])
            if code_snippets:
                print(f"💻 Code to be executed:")
                for code in code_snippets:
                    print(f"   - {code[:100]}{'...' if len(code) > 100 else ''}")
            else:
                print("💻 Code: Specific code not specified")
        
        # System settings
        elif any(word in action_lower for word in ["settings", "config", "registry", "preferences"]):
            settings = context.get("settings", [])
            if settings:
                print(f"⚙️  System settings to be modified:")
                for setting in settings:
                    print(f"   - {setting}")
            else:
                print("⚙️  System settings: Specific settings not specified")
        
        # Permissions
        elif any(word in action_lower for word in ["permission", "privilege", "access", "root"]):
            permissions = context.get("permissions", [])
            if permissions:
                print(f"🔐 Permissions to be granted:")
                for perm in permissions:
                    print(f"   - {perm}")
            else:
                print("🔐 Permissions: Specific permissions not specified")
        
        # Default case
        else:
            print("📋 Action details: Specific targets not specified")
        
        # Print any additional context
        if context.get("reason"):
            print(f"📝 Reason: {context['reason']}")
        
        if context.get("impact"):
            print(f"⚠️  Impact: {context['impact']}")
    
    def check_override_session(self, session_id: str) -> bool:
        """
        Check if an override session is valid and not expired.
        
        Args:
            session_id: The session ID to check
            
        Returns:
            True if session is valid, False otherwise
        """
        if session_id not in self.override_sessions:
            return False
        
        session = self.override_sessions[session_id]
        current_time = datetime.now().timestamp()
        
        # Check if session has expired
        if current_time > session["expires_at"]:
            del self.override_sessions[session_id]
            return False
        
        return True
    
    def use_override_session(self, session_id: str) -> bool:
        """
        Mark an override session as used.
        
        Args:
            session_id: The session ID to mark as used
            
        Returns:
            True if session was valid and marked, False otherwise
        """
        if not self.check_override_session(session_id):
            return False
        
        self.override_sessions[session_id]["used"] = True
        return True
    
    def validate_action(self, action: str, context: Dict[str, Any] = None, 
                       override_session_id: str = None) -> Dict[str, Any]:
        """
        Validate an action against system protection rules.
        
        Args:
            action: Description of the action
            context: Additional context
            override_session_id: Optional override session ID
            
        Returns:
            Validation result with safety assessment
        """
        if context is None:
            context = {}
        
        # Check for override session
        if override_session_id and self.check_override_session(override_session_id):
            if not self.override_sessions[override_session_id]["used"]:
                self.use_override_session(override_session_id)
                return {
                    "safe": True,
                    "reason": "Root override session active",
                    "blocked": False,
                    "confirmation_required": False,
                    "override_session_id": override_session_id
                }
        
        # Check if action is explicitly safe
        for category, safe_actions in self.SAFE_ACTIONS.items():
            if any(safe_action in action.lower() for safe_action in safe_actions):
                return {
                    "safe": True,
                    "reason": f"Action is in safe category: {category}",
                    "blocked": False,
                    "confirmation_required": False
                }
        
        # Check if action is explicitly forbidden
        for category, forbidden_actions in self.FORBIDDEN_ACTIONS.items():
            if any(forbidden_action in action.lower() for forbidden_action in forbidden_actions):
                self.blocked_actions.append({
                    "action": action,
                    "category": category,
                    "timestamp": datetime.now(),
                    "context": context
                })
                
                # Check if override is requested
                if context.get("request_override", False):
                    override_result = self.request_root_override(action)
                    if override_result["success"]:
                        return {
                            "safe": True,
                            "reason": "Root override granted",
                            "blocked": False,
                            "confirmation_required": False,
                            "override_session_id": override_result["session_id"]
                        }
                
                return {
                    "safe": False,
                    "reason": f"Action is forbidden: {category}",
                    "blocked": True,
                    "confirmation_required": False,
                    "override_available": True
                }
        
        # If action is not explicitly safe, require confirmation
        return {
            "safe": False,
            "reason": "Action requires explicit confirmation for safety",
            "blocked": False,
            "confirmation_required": True
        }
    
    def validate_code_execution(self, code: str, override_session_id: str = None) -> Dict[str, Any]:
        """
        Validate code before execution to prevent harmful operations.
        
        Args:
            code: Code to be executed
            override_session_id: Optional override session ID
            
        Returns:
            Validation result
        """
        # Check for override session
        if override_session_id and self.check_override_session(override_session_id):
            if not self.override_sessions[override_session_id]["used"]:
                self.use_override_session(override_session_id)
                return {
                    "safe": True,
                    "reason": "Root override session active",
                    "blocked": False,
                    "override_session_id": override_session_id
                }
        
        dangerous_imports = [
            "os", "sys", "subprocess", "shutil", "pathlib",
            "socket", "urllib", "requests", "ftplib", "telnetlib",
            "multiprocessing", "threading", "ctypes", "mmap"
        ]
        
        dangerous_functions = [
            "exec", "eval", "compile", "execfile",
            "os.system", "os.popen", "subprocess.call",
            "subprocess.Popen", "subprocess.run",
            "shutil.rmtree", "shutil.copy", "shutil.move",
            "os.remove", "os.unlink", "os.rmdir",
            "os.makedirs", "os.mkdir", "os.chmod",
            "os.chown", "os.rename", "os.link"
        ]
        
        # Check for dangerous imports
        for dangerous_import in dangerous_imports:
            if f"import {dangerous_import}" in code or f"from {dangerous_import}" in code:
                return {
                    "safe": False,
                    "reason": f"Dangerous import detected: {dangerous_import}",
                    "blocked": True,
                    "override_available": True
                }
        
        # Check for dangerous function calls
        for dangerous_function in dangerous_functions:
            if dangerous_function in code:
                return {
                    "safe": False,
                    "reason": f"Dangerous function detected: {dangerous_function}",
                    "blocked": True,
                    "override_available": True
                }
        
        return {
            "safe": True,
            "reason": "Code appears safe for execution",
            "blocked": False
        }
    
    def validate_file_operation(self, operation: str, file_path: str, 
                               override_session_id: str = None) -> Dict[str, Any]:
        """
        Validate file operations to prevent harmful file access.
        
        Args:
            operation: File operation (read, write, delete, etc.)
            file_path: Path to the file
            override_session_id: Optional override session ID
            
        Returns:
            Validation result
        """
        # Check for override session
        if override_session_id and self.check_override_session(override_session_id):
            if not self.override_sessions[override_session_id]["used"]:
                self.use_override_session(override_session_id)
                return {
                    "safe": True,
                    "reason": "Root override session active",
                    "blocked": False,
                    "override_session_id": override_session_id
                }
        
        # Check for system directories
        system_dirs = [
            "/etc", "/var", "/usr", "/bin", "/sbin", "/lib",
            "/System", "/Applications", "/Library", "/Users",
            "C:\\Windows", "C:\\System32", "C:\\Program Files"
        ]
        
        # Check for sensitive file patterns
        sensitive_patterns = [
            "password", "secret", "private", "config", "settings",
            ".env", ".bashrc", ".profile", ".ssh", ".gnupg",
            "id_rsa", "id_dsa", "known_hosts", "authorized_keys"
        ]
        
        # Check if file is in system directory
        for system_dir in system_dirs:
            if file_path.startswith(system_dir):
                return {
                    "safe": False,
                    "reason": f"Access to system directory: {system_dir}",
                    "blocked": True,
                    "override_available": True
                }
        
        # Check for sensitive file patterns
        for pattern in sensitive_patterns:
            if pattern.lower() in file_path.lower():
                return {
                    "safe": False,
                    "reason": f"Access to sensitive file pattern: {pattern}",
                    "blocked": True,
                    "override_available": True
                }
        
        return {
            "safe": True,
            "reason": "File operation appears safe",
            "blocked": False
        }
    
    def validate_network_access(self, url: str, operation: str, 
                               override_session_id: str = None) -> Dict[str, Any]:
        """
        Validate network access to prevent harmful connections.
        
        Args:
            url: URL or endpoint to access
            operation: Network operation (GET, POST, etc.)
            override_session_id: Optional override session ID
            
        Returns:
            Validation result
        """
        # Check for override session
        if override_session_id and self.check_override_session(override_session_id):
            if not self.override_sessions[override_session_id]["used"]:
                self.use_override_session(override_session_id)
                return {
                    "safe": True,
                    "reason": "Root override session active",
                    "blocked": False,
                    "override_session_id": override_session_id
                }
        
        # Check for local network access
        local_patterns = [
            "localhost", "127.0.0.1", "0.0.0.0", "::1",
            "192.168.", "10.", "172.16.", "172.17.", "172.18.",
            "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
            "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
            "172.29.", "172.30.", "172.31."
        ]
        
        # Check for potentially harmful protocols
        harmful_protocols = [
            "ftp://", "telnet://", "ssh://", "sftp://",
            "file://", "gopher://", "dict://"
        ]
        
        # Check for local network access
        for pattern in local_patterns:
            if pattern in url:
                return {
                    "safe": False,
                    "reason": f"Local network access detected: {pattern}",
                    "blocked": True,
                    "override_available": True
                }
        
        # Check for harmful protocols
        for protocol in harmful_protocols:
            if url.startswith(protocol):
                return {
                    "safe": False,
                    "reason": f"Harmful protocol detected: {protocol}",
                    "blocked": True,
                    "override_available": True
                }
        
        return {
            "safe": True,
            "reason": "Network access appears safe",
            "blocked": False
        }
    
    def get_protection_report(self) -> Dict[str, Any]:
        """Get a report of system protection status."""
        return {
            "protection_level": self.protection_level,
            "blocked_actions_count": len(self.blocked_actions),
            "safe_actions_count": len(self.safe_actions),
            "recent_blocked_actions": self.blocked_actions[-10:] if self.blocked_actions else [],
            "active_override_sessions": len([s for s in self.override_sessions.values() 
                                          if datetime.now().timestamp() < s["expires_at"]]),
            "root_password_set": self.root_password_hash is not None,
            "timestamp": datetime.now().isoformat()
        }


# Global system protection instance
SYSTEM_PROTECTION = SystemProtection()


def get_system_protection() -> SystemProtection:
    """Get the global system protection instance."""
    return SYSTEM_PROTECTION


def validate_action_safe(action: str, context: Dict[str, Any] = None, 
                        override_session_id: str = None) -> Dict[str, Any]:
    """Validate an action is safe for system protection."""
    return SYSTEM_PROTECTION.validate_action(action, context, override_session_id)


def validate_code_safe(code: str, override_session_id: str = None) -> Dict[str, Any]:
    """Validate code is safe for execution."""
    return SYSTEM_PROTECTION.validate_code_execution(code, override_session_id)


def validate_file_safe(operation: str, file_path: str, 
                      override_session_id: str = None) -> Dict[str, Any]:
    """Validate file operation is safe."""
    return SYSTEM_PROTECTION.validate_file_operation(operation, file_path, override_session_id)


def validate_network_safe(url: str, operation: str, 
                         override_session_id: str = None) -> Dict[str, Any]:
    """Validate network access is safe."""
    return SYSTEM_PROTECTION.validate_network_access(url, operation, override_session_id)


def set_root_password(password: str):
    """Set the root password for override functionality."""
    SYSTEM_PROTECTION.set_root_password(password)


def request_root_override(action: str, context: Dict[str, Any] = None, session_id: str = None) -> Dict[str, Any]:
    """Request root password override for a blocked action."""
    return SYSTEM_PROTECTION.request_root_override(action, context, session_id) 