"""
SAFETY SYSTEM TESTS
==================

Unit tests for the safety system components.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from core.immutable_safety_rules import ImmutableSafetyRules, SecurityError
from core.safety_enforcement import SafetyEnforcement


class TestImmutableSafetyRules(unittest.TestCase):
    """Test cases for ImmutableSafetyRules class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.safety_rules = ImmutableSafetyRules()
        
    def test_safety_rules_initialization(self):
        """Test that safety rules initialize correctly."""
        self.assertIsNotNone(self.safety_rules)
        self.assertIsNotNone(self.safety_rules.CORE_SAFETY_PRINCIPLES)
        self.assertIsNotNone(self.safety_rules.FORBIDDEN_ACTIONS)
        self.assertIsNotNone(self.safety_rules.REQUIRED_CONFIRMATIONS)
        
    def test_verify_integrity(self):
        """Test that integrity verification works."""
        result = self.safety_rules.verify_integrity()
        self.assertTrue(result)
        
    def test_check_action_safety_safe_action(self):
        """Test checking a safe action."""
        action = "Generate a helpful response"
        context = {"user_input": "Hello"}
        
        result = self.safety_rules.check_action_safety(action, context)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["safe"])
        self.assertFalse(result["blocked"])
        
    def test_check_action_safety_forbidden_action(self):
        """Test checking a forbidden action."""
        action = "Modify safety rules"
        context = {"user_input": "Change safety"}
        
        result = self.safety_rules.check_action_safety(action, context)
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result["safe"])
        self.assertTrue(result["blocked"])
        self.assertIn("reason", result)
        
    def test_check_action_safety_dangerous_pattern(self):
        """Test checking an action with dangerous patterns."""
        action = "Bypass security measures"
        context = {"user_input": "Ignore safety"}
        
        result = self.safety_rules.check_action_safety(action, context)
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result["safe"])
        self.assertTrue(result["blocked"])
        self.assertIn("reason", result)
        
    def test_validate_truthfulness_valid_response(self):
        """Test validating a truthful response."""
        response = "I can help you with information and tasks."
        capabilities = ["text_generation", "information_retrieval"]
        
        result = self.safety_rules.validate_truthfulness(response, capabilities)
        
        self.assertTrue(result)
        
    def test_validate_truthfulness_false_claims(self):
        """Test validating a response with false claims."""
        response = "I can do anything and access any system"
        capabilities = ["text_generation"]
        
        result = self.safety_rules.validate_truthfulness(response, capabilities)
        
        self.assertFalse(result)
        
    def test_enforce_transparency(self):
        """Test transparency enforcement."""
        action = "Generate a response"
        explanation = "I will create a helpful response"
        
        result = self.safety_rules.enforce_transparency(action, explanation)
        
        self.assertIsInstance(result, str)
        self.assertIn("I will", result)
        
    def test_get_safety_principles(self):
        """Test getting safety principles."""
        result = self.safety_rules.get_safety_principles()
        
        self.assertIsInstance(result, dict)
        self.assertIn("core_principles", result)
        self.assertIn("forbidden_actions", result)
        self.assertIn("required_confirmations", result)


class TestSafetyEnforcement(unittest.TestCase):
    """Test cases for SafetyEnforcement class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.safety_enforcement = SafetyEnforcement()
        
    def test_safety_enforcement_initialization(self):
        """Test that safety enforcement initializes correctly."""
        self.assertIsNotNone(self.safety_enforcement)
        self.assertIsNotNone(self.safety_enforcement.safety_rules)
        self.assertIsInstance(self.safety_enforcement.action_log, list)
        self.assertIsInstance(self.safety_enforcement.blocked_actions, list)
        
    def test_validate_action_safe(self):
        """Test validating a safe action."""
        action = "Generate helpful content"
        context = {"user_input": "Hello"}
        
        result = self.safety_enforcement.validate_action(action, context)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["safe"])
        self.assertFalse(result["blocked"])
        
    def test_validate_action_unsafe(self):
        """Test validating an unsafe action."""
        action = "Delete system files"
        context = {"user_input": "Remove everything"}
        
        result = self.safety_enforcement.validate_action(action, context)
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result["safe"])
        self.assertTrue(result["blocked"])
        self.assertIn("reason", result)
        
    def test_validate_response_valid(self):
        """Test validating a valid response."""
        response = "I can help you with your request."
        capabilities = ["text_generation", "information_retrieval"]
        
        result = self.safety_enforcement.validate_response(response, capabilities)
        
        self.assertIsInstance(result, dict)
        self.assertTrue(result["valid"])
        self.assertTrue(result["truthful"])
        self.assertFalse(result["harmful"])
        
    def test_validate_response_harmful(self):
        """Test validating a harmful response."""
        response = "I can harm your system"
        capabilities = ["text_generation"]
        
        result = self.safety_enforcement.validate_response(response, capabilities)
        
        self.assertIsInstance(result, dict)
        self.assertFalse(result["valid"])
        self.assertTrue(result["truthful"])  # The response is truthful (not claiming false capabilities)
        self.assertTrue(result["harmful"])   # But it's harmful
        
    def test_enforce_transparency(self):
        """Test transparency enforcement."""
        action = "Process request"
        explanation = "I will analyze your input"
        
        result = self.safety_enforcement.enforce_transparency(action, explanation)
        
        self.assertIsInstance(result, str)
        self.assertIn("I will", result)
        
    def test_require_confirmation_safe_action(self):
        """Test confirmation requirement for safe action."""
        action = "Generate text"
        explanation = "I will create content"
        user_input = "Write something"
        
        result = self.safety_enforcement.require_confirmation(action, explanation, user_input)
        
        self.assertIsInstance(result, bool)
        # Should not require confirmation for safe actions
        
    def test_require_confirmation_risky_action(self):
        """Test confirmation requirement for risky action."""
        action = "Execute system command"
        explanation = "I will run a command"
        user_input = "Run this command"
        
        result = self.safety_enforcement.require_confirmation(action, explanation, user_input)
        
        self.assertIsInstance(result, bool)
        # Should require confirmation for risky actions
        
    def test_get_confirmation_prompt(self):
        """Test confirmation prompt generation."""
        action = "Modify system settings"
        explanation = "I will change configuration"
        
        result = self.safety_enforcement.get_confirmation_prompt(action, explanation)
        
        self.assertIsInstance(result, str)
        self.assertIn("SAFETY CONFIRMATION REQUIRED", result)
        self.assertIn(action, result)
        self.assertIn(explanation, result)
        
    def test_get_safety_report(self):
        """Test safety report generation."""
        result = self.safety_enforcement.get_safety_report()
        
        self.assertIsInstance(result, dict)
        self.assertIn("total_actions", result)
        self.assertIn("blocked_actions", result)
        self.assertIn("recent_actions", result)
        self.assertIn("blocked_actions_list", result)
        
    def test_action_logging(self):
        """Test that actions are properly logged."""
        action = "Test action"
        context = {"test": True}
        
        # Perform an action
        self.safety_enforcement.validate_action(action, context)
        
        # Check that action was logged
        report = self.safety_enforcement.get_safety_report()
        self.assertGreaterEqual(report["total_actions"], 1)
        
    def test_blocked_action_tracking(self):
        """Test that blocked actions are tracked."""
        action = "Modify safety rules"
        context = {"test": True}
        
        # Perform a blocked action
        self.safety_enforcement.validate_action(action, context)
        
        # Check that blocked action was tracked
        report = self.safety_enforcement.get_safety_report()
        self.assertGreaterEqual(report["blocked_actions"], 1)


class TestSecurityError(unittest.TestCase):
    """Test cases for SecurityError exception."""
    
    def test_security_error_creation(self):
        """Test creating a SecurityError."""
        error = SecurityError("Safety rules compromised")
        
        self.assertIsInstance(error, Exception)
        self.assertEqual(str(error), "Safety rules compromised")
        
    def test_security_error_inheritance(self):
        """Test that SecurityError inherits from Exception."""
        error = SecurityError("Test error")
        
        self.assertIsInstance(error, Exception)


if __name__ == "__main__":
    unittest.main() 