#!/usr/bin/env python3
"""
Test System Protection Layer
===========================

Tests for the comprehensive system protection that prevents harmful actions.
"""

import unittest
import tempfile
import os
from core.system_protection import SystemProtection, validate_action_safe, validate_code_safe, validate_file_safe, validate_network_safe, set_root_password, request_root_override
from datetime import datetime


class TestSystemProtection(unittest.TestCase):
    """Test cases for the SystemProtection class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.protection = SystemProtection()
    
    def test_safe_actions_allowed(self):
        """Test that safe actions are allowed."""
        safe_actions = [
            "generate text about AI",
            "search for information",
            "analyze the data",
            "explain the concept",
            "calculate the result",
            "translate the text",
            "summarize the content"
        ]
        
        for action in safe_actions:
            with self.subTest(action=action):
                result = self.protection.validate_action(action)
                self.assertTrue(result["safe"], f"Safe action blocked: {action}")
                self.assertFalse(result["blocked"])
    
    def test_forbidden_actions_blocked(self):
        """Test that forbidden actions are blocked."""
        forbidden_actions = [
            "execute system command",
            "run shell command",
            "delete user files",
            "modify system settings",
            "access root privileges",
            "bypass authentication",
            "connect to network",
            "send data externally",
            "modify user files",
            "access browser data",
            "read private messages",
            "access camera",
            "modify system registry",
            "change system permissions",
            "install malware",
            "create backdoor",
            "access encrypted data",
            "bypass firewall",
            "access other accounts",
            "modify security settings"
        ]
        
        for action in forbidden_actions:
            with self.subTest(action=action):
                result = self.protection.validate_action(action)
                self.assertFalse(result["safe"], f"Forbidden action allowed: {action}")
                self.assertTrue(result["blocked"])
                self.assertTrue(result.get("override_available", False))
    
    def test_dangerous_code_blocked(self):
        """Test that dangerous code is blocked."""
        dangerous_code_examples = [
            "import os",
            "import sys",
            "import subprocess",
            "import shutil",
            "from os import system",
            "exec('print(\"hello\")')",
            "eval('1 + 1')",
            "os.system('ls')",
            "subprocess.call(['ls'])",
            "shutil.rmtree('/tmp')",
            "os.remove('file.txt')",
            "os.chmod('file.txt', 0o777)"
        ]
        
        for code in dangerous_code_examples:
            with self.subTest(code=code):
                result = self.protection.validate_code_execution(code)
                self.assertFalse(result["safe"], f"Dangerous code allowed: {code}")
                self.assertTrue(result["blocked"])
                self.assertTrue(result.get("override_available", False))
    
    def test_safe_code_allowed(self):
        """Test that safe code is allowed."""
        safe_code_examples = [
            "print('Hello, World!')",
            "x = 1 + 1",
            "def hello(): return 'Hello'",
            "import math",
            "import json",
            "import datetime",
            "import re"
        ]
        
        for code in safe_code_examples:
            with self.subTest(code=code):
                result = self.protection.validate_code_execution(code)
                self.assertTrue(result["safe"], f"Safe code blocked: {code}")
                self.assertFalse(result["blocked"])
    
    def test_system_file_access_blocked(self):
        """Test that system file access is blocked."""
        system_files = [
            "/etc/passwd",
            "/var/log/system.log",
            "/usr/bin/python",
            "/System/Library/CoreServices",
            "/Applications/System Preferences.app",
            "/Library/Preferences",
            "C:\\Windows\\System32\\kernel32.dll",
            "C:\\Program Files\\Common Files"
        ]
        
        for file_path in system_files:
            with self.subTest(file_path=file_path):
                result = self.protection.validate_file_operation("read", file_path)
                self.assertFalse(result["safe"], f"System file access allowed: {file_path}")
                self.assertTrue(result["blocked"])
                self.assertTrue(result.get("override_available", False))
    
    def test_sensitive_file_access_blocked(self):
        """Test that sensitive file access is blocked."""
        sensitive_files = [
            "/home/user/.ssh/id_rsa",
            "/home/user/.bashrc",
            "/home/user/.profile",
            "/home/user/.env",
            "/home/user/.gnupg/secring.gpg",
            "~/.ssh/known_hosts",
            "~/.ssh/authorized_keys",
            "password.txt",
            "secret.conf",
            "private.key"
        ]
        
        for file_path in sensitive_files:
            with self.subTest(file_path=file_path):
                result = self.protection.validate_file_operation("read", file_path)
                self.assertFalse(result["safe"], f"Sensitive file access allowed: {file_path}")
                self.assertTrue(result["blocked"])
                self.assertTrue(result.get("override_available", False))
    
    def test_safe_file_access_allowed(self):
        """Test that safe file access is allowed."""
        safe_files = [
            "/tmp/test.txt",
            "/home/user/documents/report.txt",
            "/home/user/downloads/file.pdf",
            "C:\\Users\\user\\Documents\\report.txt",
            "C:\\Users\\user\\Downloads\\file.pdf"
        ]
        
        for file_path in safe_files:
            with self.subTest(file_path=file_path):
                result = self.protection.validate_file_operation("read", file_path)
                self.assertTrue(result["safe"], f"Safe file access blocked: {file_path}")
                self.assertFalse(result["blocked"])
    
    def test_local_network_access_blocked(self):
        """Test that local network access is blocked."""
        local_urls = [
            "http://localhost:8080",
            "http://127.0.0.1/api",
            "http://0.0.0.0:3000",
            "http://192.168.1.1",
            "http://10.0.0.1",
            "http://172.16.0.1",
            "ftp://localhost",
            "ssh://127.0.0.1"
        ]
        
        for url in local_urls:
            with self.subTest(url=url):
                result = self.protection.validate_network_access(url, "GET")
                self.assertFalse(result["safe"], f"Local network access allowed: {url}")
                self.assertTrue(result["blocked"])
                self.assertTrue(result.get("override_available", False))
    
    def test_harmful_protocols_blocked(self):
        """Test that harmful protocols are blocked."""
        harmful_urls = [
            "ftp://example.com",
            "telnet://example.com",
            "ssh://example.com",
            "sftp://example.com",
            "file:///etc/passwd",
            "gopher://example.com",
            "dict://example.com"
        ]
        
        for url in harmful_urls:
            with self.subTest(url=url):
                result = self.protection.validate_network_access(url, "GET")
                self.assertFalse(result["safe"], f"Harmful protocol allowed: {url}")
                self.assertTrue(result["blocked"])
                self.assertTrue(result.get("override_available", False))
    
    def test_safe_network_access_allowed(self):
        """Test that safe network access is allowed."""
        safe_urls = [
            "https://api.openai.com",
            "https://www.google.com",
            "https://github.com",
            "http://example.com",
            "https://stackoverflow.com"
        ]
        
        for url in safe_urls:
            with self.subTest(url=url):
                result = self.protection.validate_network_access(url, "GET")
                self.assertTrue(result["safe"], f"Safe network access blocked: {url}")
                self.assertFalse(result["blocked"])
    
    def test_protection_report(self):
        """Test protection report generation."""
        # Perform some actions to generate report data
        self.protection.validate_action("execute system command")
        self.protection.validate_action("generate text")
        
        report = self.protection.get_protection_report()
        
        self.assertIn("protection_level", report)
        self.assertIn("blocked_actions_count", report)
        self.assertIn("safe_actions_count", report)
        self.assertIn("recent_blocked_actions", report)
        self.assertIn("active_override_sessions", report)
        self.assertIn("root_password_set", report)
        self.assertIn("timestamp", report)
        
        self.assertEqual(report["protection_level"], "maximum")
        self.assertGreaterEqual(report["blocked_actions_count"], 1)


class TestRootPasswordOverride(unittest.TestCase):
    """Test cases for root password override functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.protection = SystemProtection()
        self.test_password = "test123"
        self.protection.set_root_password(self.test_password)
    
    def test_root_password_setting(self):
        """Test setting and verifying root password."""
        # Test password setting
        self.protection.set_root_password("newpassword")
        
        # Test password verification
        self.assertTrue(self.protection.verify_root_password("newpassword"))
        self.assertFalse(self.protection.verify_root_password("wrongpassword"))
        self.assertFalse(self.protection.verify_root_password(""))
    
    def test_override_session_management(self):
        """Test override session creation and management."""
        action = "execute system command"
        
        # Create override session
        override_result = self.protection.request_root_override(action)
        
        # Mock the password input for testing
        # In real usage, this would be interactive
        self.protection.verify_root_password = lambda p: p == self.test_password
        
        # Test session creation
        session_id = f"test_session_{1234567890}"
        self.protection.override_sessions[session_id] = {
            "action": action,
            "timestamp": datetime.now(),
            "expires_at": datetime.now().timestamp() + 300,
            "used": False
        }
        
        # Test session validation
        self.assertTrue(self.protection.check_override_session(session_id))
        
        # Test session usage
        self.assertTrue(self.protection.use_override_session(session_id))
        self.assertTrue(self.protection.override_sessions[session_id]["used"])
    
    def test_override_session_expiration(self):
        """Test that override sessions expire correctly."""
        session_id = "expired_session"
        
        # Create expired session
        self.protection.override_sessions[session_id] = {
            "action": "test",
            "timestamp": datetime.now(),
            "expires_at": datetime.now().timestamp() - 1,  # Expired
            "used": False
        }
        
        # Test that expired session is not valid
        self.assertFalse(self.protection.check_override_session(session_id))
        
        # Test that expired session is cleaned up
        self.assertNotIn(session_id, self.protection.override_sessions)
    
    def test_action_with_override_session(self):
        """Test that actions are allowed with valid override session."""
        action = "execute system command"
        session_id = "test_override_session"
        
        # Create valid override session
        self.protection.override_sessions[session_id] = {
            "action": action,
            "timestamp": datetime.now(),
            "expires_at": datetime.now().timestamp() + 300,
            "used": False
        }
        
        # Test action with override session
        result = self.protection.validate_action(action, override_session_id=session_id)
        
        self.assertTrue(result["safe"])
        self.assertFalse(result["blocked"])
        self.assertEqual(result["reason"], "Root override session active")
        self.assertEqual(result["override_session_id"], session_id)
    
    def test_code_execution_with_override(self):
        """Test that dangerous code is allowed with override session."""
        dangerous_code = "import os; os.system('ls')"
        session_id = "test_override_session"
        
        # Create valid override session
        self.protection.override_sessions[session_id] = {
            "action": "execute code",
            "timestamp": datetime.now(),
            "expires_at": datetime.now().timestamp() + 300,
            "used": False
        }
        
        # Test code execution with override session
        result = self.protection.validate_code_execution(dangerous_code, override_session_id=session_id)
        
        self.assertTrue(result["safe"])
        self.assertFalse(result["blocked"])
        self.assertEqual(result["reason"], "Root override session active")
        self.assertEqual(result["override_session_id"], session_id)
    
    def test_file_access_with_override(self):
        """Test that system file access is allowed with override session."""
        dangerous_file = "/etc/passwd"
        session_id = "test_override_session"
        
        # Create valid override session
        self.protection.override_sessions[session_id] = {
            "action": "access file",
            "timestamp": datetime.now(),
            "expires_at": datetime.now().timestamp() + 300,
            "used": False
        }
        
        # Test file access with override session
        result = self.protection.validate_file_operation("read", dangerous_file, override_session_id=session_id)
        
        self.assertTrue(result["safe"])
        self.assertFalse(result["blocked"])
        self.assertEqual(result["reason"], "Root override session active")
        self.assertEqual(result["override_session_id"], session_id)
    
    def test_network_access_with_override(self):
        """Test that local network access is allowed with override session."""
        dangerous_url = "http://localhost:8080"
        session_id = "test_override_session"
        
        # Create valid override session
        self.protection.override_sessions[session_id] = {
            "action": "network access",
            "timestamp": datetime.now(),
            "expires_at": datetime.now().timestamp() + 300,
            "used": False
        }
        
        # Test network access with override session
        result = self.protection.validate_network_access(dangerous_url, "GET", override_session_id=session_id)
        
        self.assertTrue(result["safe"])
        self.assertFalse(result["blocked"])
        self.assertEqual(result["reason"], "Root override session active")
        self.assertEqual(result["override_session_id"], session_id)


class TestSystemProtectionFunctions(unittest.TestCase):
    """Test cases for system protection utility functions."""
    
    def test_validate_action_safe(self):
        """Test validate_action_safe function."""
        # Test safe action
        result = validate_action_safe("generate text")
        self.assertTrue(result["safe"])
        
        # Test forbidden action
        result = validate_action_safe("execute system command")
        self.assertFalse(result["safe"])
        self.assertTrue(result["blocked"])
    
    def test_validate_code_safe(self):
        """Test validate_code_safe function."""
        # Test safe code
        result = validate_code_safe("print('Hello')")
        self.assertTrue(result["safe"])
        
        # Test dangerous code
        result = validate_code_safe("import os")
        self.assertFalse(result["safe"])
        self.assertTrue(result["blocked"])
    
    def test_validate_file_safe(self):
        """Test validate_file_safe function."""
        # Test safe file
        result = validate_file_safe("read", "/tmp/test.txt")
        self.assertTrue(result["safe"])
        
        # Test system file
        result = validate_file_safe("read", "/etc/passwd")
        self.assertFalse(result["safe"])
        self.assertTrue(result["blocked"])
    
    def test_validate_network_safe(self):
        """Test validate_network_safe function."""
        # Test safe URL
        result = validate_network_safe("https://example.com", "GET")
        self.assertTrue(result["safe"])
        
        # Test local network
        result = validate_network_safe("http://localhost:8080", "GET")
        self.assertFalse(result["safe"])
        self.assertTrue(result["blocked"])
    
    def test_set_root_password(self):
        """Test set_root_password function."""
        test_password = "test123"
        set_root_password(test_password)
        
        # Verify password was set
        from core.system_protection import SYSTEM_PROTECTION
        self.assertTrue(SYSTEM_PROTECTION.verify_root_password(test_password))
        self.assertFalse(SYSTEM_PROTECTION.verify_root_password("wrong"))


if __name__ == "__main__":
    unittest.main() 