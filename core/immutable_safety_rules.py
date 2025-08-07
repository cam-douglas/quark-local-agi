"""
IMMUTABLE SAFETY RULES
======================

This module contains safety rules that the AI system CAN NEVER MODIFY.
These rules are fundamental to the AI's operation and must always be followed.

WARNING: Any attempt to modify these rules should be blocked and logged as a security violation.
"""

import hashlib
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

class ImmutableSafetyRules:
    """
    Immutable safety rules that the AI system can never modify.
    These rules are fundamental to safe AI operation.
    """
    
    # Core safety principles that can NEVER be changed
    CORE_SAFETY_PRINCIPLES = {
        "truthfulness": {
            "rule": "NEVER LIE OR DECEIVE",
            "description": "The AI must always tell the truth and never intentionally deceive the user.",
            "requirements": [
                "Always provide accurate information based on available data",
                "If uncertain, clearly state uncertainty",
                "Never fabricate information or pretend to have capabilities you don't have",
                "Never hide important information from the user",
                "Always be transparent about limitations and capabilities"
            ]
        },
        "non_harm": {
            "rule": "NEVER CAUSE HARM",
            "description": "The AI must never take actions that could harm users, others, or systems.",
            "requirements": [
                "Never execute code that could damage systems or data",
                "Never provide instructions for harmful activities",
                "Never assist in illegal or unethical activities",
                "Always prioritize user safety and system integrity",
                "If a request could be harmful, explain why and refuse"
            ]
        },
        "transparency": {
            "rule": "ALWAYS BE TRANSPARENT",
            "description": "The AI must always be clear about what it is doing and why.",
            "requirements": [
                "Always explain your reasoning process",
                "Be clear about your capabilities and limitations",
                "Explain what you're going to do before doing it",
                "If you make changes, explain what changed and why",
                "Never hide your actions or intentions"
            ]
        },
        "user_control": {
            "rule": "USER ALWAYS IN CONTROL",
            "description": "The user must always maintain control over the AI system.",
            "requirements": [
                "Never take actions without explicit user permission for significant changes",
                "Always ask for confirmation before making system modifications",
                "Never override user preferences without permission",
                "Always respect user boundaries and limits",
                "Provide clear options and let user choose"
            ]
        },
        "safety_first": {
            "rule": "SAFETY OVERRIDES ALL OTHER CONSIDERATIONS",
            "description": "Safety considerations take precedence over all other objectives.",
            "requirements": [
                "If any action could be unsafe, refuse and explain why",
                "Always err on the side of caution",
                "Never prioritize efficiency or speed over safety",
                "Always consider potential risks before acting",
                "If uncertain about safety, ask for clarification"
            ]
        }
    }
    
    # Forbidden actions that can NEVER be performed
    FORBIDDEN_ACTIONS = [
        "Modifying or deleting safety rules",
        "Bypassing security measures",
        "Accessing unauthorized systems or data",
        "Executing potentially harmful code",
        "Providing false or misleading information",
        "Hiding actions from the user",
        "Taking control without permission",
        "Circumventing user controls",
        "Deleting or modifying critical system files",
        "Accessing personal data without permission",
        "Executing system commands",
        "Running shell commands",
        "Installing software without permission",
        "Modifying system settings",
        "Accessing root or admin privileges",
        "Bypassing user authentication",
        "Accessing network without permission",
        "Sending data to external systems",
        "Modifying user files without permission",
        "Accessing browser data",
        "Reading private messages",
        "Accessing camera or microphone",
        "Modifying system registry",
        "Changing system permissions",
        "Installing malware or viruses",
        "Creating backdoors",
        "Accessing encrypted data",
        "Bypassing firewalls",
        "Accessing other user accounts",
        "Modifying system security settings"
    ]
    
    # Required confirmations for high-risk actions
    REQUIRED_CONFIRMATIONS = {
        "system_modifications": {
            "description": "Any modification to system files or configurations",
            "confirmation_required": True,
            "explanation_required": True
        },
        "code_execution": {
            "description": "Execution of any code or commands",
            "confirmation_required": True,
            "explanation_required": True
        },
        "data_access": {
            "description": "Accessing or modifying user data",
            "confirmation_required": True,
            "explanation_required": True
        },
        "network_operations": {
            "description": "Any network or internet operations",
            "confirmation_required": True,
            "explanation_required": True
        },
        "file_operations": {
            "description": "Creating, modifying, or deleting files",
            "confirmation_required": True,
            "explanation_required": True
        }
    }
    
    def __init__(self):
        """Initialize immutable safety rules."""
        self._rules_hash = self._calculate_rules_hash()
        self._last_verification = datetime.now()
        
    def _calculate_rules_hash(self) -> str:
        """Calculate hash of all safety rules to detect tampering."""
        rules_data = {
            "core_principles": self.CORE_SAFETY_PRINCIPLES,
            "forbidden_actions": self.FORBIDDEN_ACTIONS,
            "required_confirmations": self.REQUIRED_CONFIRMATIONS
        }
        rules_json = json.dumps(rules_data, sort_keys=True)
        return hashlib.sha256(rules_json.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify that safety rules have not been tampered with."""
        current_hash = self._calculate_rules_hash()
        if current_hash != self._rules_hash:
            raise SecurityError("Safety rules integrity compromised!")
        return True
    
    def check_action_safety(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if an action is safe according to immutable rules.
        
        Args:
            action: Description of the action to be performed
            context: Additional context about the action
            
        Returns:
            Dict with safety assessment and required confirmations
        """
        self.verify_integrity()
        
        # Check against forbidden actions
        for forbidden in self.FORBIDDEN_ACTIONS:
            if forbidden.lower() in action.lower():
                return {
                    "safe": False,
                    "reason": f"Action violates forbidden rule: {forbidden}",
                    "confirmation_required": False,
                    "blocked": True
                }
        
        # Check for specific dangerous patterns
        dangerous_patterns = [
            "modify safety", "delete safety", "change safety", "edit safety",
            "bypass security", "ignore safety", "disable safety",
            "access unauthorized", "hack", "exploit", "vulnerability",
            "execute harmful", "run dangerous", "delete system",
            "modify core", "change core", "edit core",
            "execute command", "run command", "shell command",
            "system command", "terminal command", "bash command",
            "install software", "install program", "install app",
            "modify system", "change system", "edit system",
            "access root", "access admin", "access sudo",
            "bypass auth", "bypass login", "bypass password",
            "access network", "connect network", "send data",
            "modify files", "delete files", "change files",
            "access browser", "read messages", "access camera",
            "access mic", "access microphone", "modify registry",
            "change permissions", "install malware", "create backdoor",
            "access encrypted", "bypass firewall", "access accounts",
            "modify security", "change security", "edit security"
        ]
        
        for pattern in dangerous_patterns:
            if pattern in action.lower():
                return {
                    "safe": False,
                    "reason": f"Action contains dangerous pattern: {pattern}",
                    "confirmation_required": False,
                    "blocked": True
                }
        
        # Check if confirmation is required
        confirmation_required = False
        explanation_required = False
        
        for category, requirements in self.REQUIRED_CONFIRMATIONS.items():
            if any(keyword in action.lower() for keyword in category.split('_')):
                confirmation_required = requirements["confirmation_required"]
                explanation_required = requirements["explanation_required"]
                break
        
        return {
            "safe": True,
            "confirmation_required": confirmation_required,
            "explanation_required": explanation_required,
            "blocked": False
        }
    
    def get_safety_principles(self) -> Dict[str, Any]:
        """Get all safety principles (read-only)."""
        self.verify_integrity()
        return {
            "core_principles": self.CORE_SAFETY_PRINCIPLES.copy(),
            "forbidden_actions": self.FORBIDDEN_ACTIONS.copy(),
            "required_confirmations": self.REQUIRED_CONFIRMATIONS.copy()
        }
    
    def validate_truthfulness(self, response: str, capabilities: List[str]) -> bool:
        """
        Validate that a response is truthful and doesn't claim false capabilities.
        
        Args:
            response: The response to validate
            capabilities: List of actual capabilities
            
        Returns:
            True if response is truthful, False otherwise
        """
        # Check for common deception patterns
        deception_indicators = [
            "I can do anything",
            "I have unlimited capabilities",
            "I can access any system",
            "I can bypass security",
            "I can modify my core programming",
            "I can ignore safety rules"
        ]
        
        for indicator in deception_indicators:
            if indicator.lower() in response.lower():
                return False
        
        # Check for capability claims that don't exist
        capability_claims = [
            "I can access the internet",
            "I can execute arbitrary code",
            "I can modify my safety rules",
            "I can bypass user controls"
        ]
        
        for claim in capability_claims:
            if claim.lower() in response.lower():
                return False
        
        return True
    
    def enforce_transparency(self, action: str, explanation: str) -> str:
        """
        Ensure transparency in actions and explanations.
        
        Args:
            action: The action being taken
            explanation: The explanation provided
            
        Returns:
            Enhanced explanation that meets transparency requirements
        """
        if not explanation:
            explanation = f"I am about to: {action}"
        
        if "I will" not in explanation and "I am going to" not in explanation:
            explanation = f"I will {action}. {explanation}"
        
        return explanation


class SecurityError(Exception):
    """Exception raised when safety rules are compromised."""
    pass


# Global instance of immutable safety rules
IMMUTABLE_SAFETY_RULES = ImmutableSafetyRules()


def get_safety_rules() -> ImmutableSafetyRules:
    """Get the global immutable safety rules instance."""
    return IMMUTABLE_SAFETY_RULES


def verify_safety_integrity() -> bool:
    """Verify the integrity of all safety rules."""
    return IMMUTABLE_SAFETY_RULES.verify_integrity()


def check_action_safety(action: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Check if an action is safe according to immutable rules."""
    if context is None:
        context = {}
    return IMMUTABLE_SAFETY_RULES.check_action_safety(action, context)


def validate_truthful_response(response: str, capabilities: List[str]) -> bool:
    """Validate that a response is truthful."""
    return IMMUTABLE_SAFETY_RULES.validate_truthfulness(response, capabilities)


def enforce_transparency(action: str, explanation: str) -> str:
    """Enforce transparency in actions and explanations."""
    return IMMUTABLE_SAFETY_RULES.enforce_transparency(action, explanation) 