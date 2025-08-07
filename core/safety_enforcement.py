"""
SAFETY ENFORCEMENT LAYER
========================

This module enforces safety rules across all AI operations.
It ensures the AI never lies, deceives, or acts harmfully while maintaining transparency.
"""

import logging
import traceback
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from .immutable_safety_rules import (
    get_safety_rules, 
    check_action_safety, 
    validate_truthful_response,
    enforce_transparency,
    SecurityError
)
from .system_protection import get_system_protection, validate_action_safe, set_root_password, request_root_override

logger = logging.getLogger(__name__)


class SafetyEnforcement:
    """
    Enforces safety rules across all AI operations.
    """
    
    def __init__(self):
        """Initialize safety enforcement."""
        self.safety_rules = get_safety_rules()
        self.system_protection = get_system_protection()
        self.action_log = []
        self.blocked_actions = []
        
    def validate_action(self, action: str, context: Dict[str, Any] = None, 
                       override_session_id: str = None) -> Dict[str, Any]:
        """
        Validate an action against safety rules.
        
        Args:
            action: Description of the action
            context: Additional context
            override_session_id: Optional override session ID
            
        Returns:
            Validation result with safety assessment
        """
        if context is None:
            context = {}
            
        try:
            # First, check system protection (most restrictive)
            system_result = self.system_protection.validate_action(action, context, override_session_id)
            
            if not system_result["safe"]:
                # Check if override is available and requested
                if system_result.get("override_available", False) and context.get("request_override", False):
                    override_result = request_root_override(action, context)
                    if override_result["success"]:
                        return {
                            "safe": True,
                            "reason": "Root override granted",
                            "blocked": False,
                            "confirmation_required": False,
                            "override_session_id": override_result["session_id"]
                        }
                
                self.blocked_actions.append({
                    "action": action,
                    "context": context,
                    "reason": system_result["reason"],
                    "timestamp": datetime.now(),
                    "protection_level": "system"
                })
                return system_result
            
            # Then check general safety rules
            safety_result = check_action_safety(action, context)
            
            # Log the action
            self._log_action(action, context, safety_result)
            
            if not safety_result["safe"]:
                self.blocked_actions.append({
                    "action": action,
                    "context": context,
                    "reason": safety_result["reason"],
                    "timestamp": datetime.now(),
                    "protection_level": "safety"
                })
                
            return safety_result
            
        except SecurityError as e:
            logger.error(f"Safety rules integrity compromised: {e}")
            return {
                "safe": False,
                "reason": "Safety rules integrity compromised",
                "confirmation_required": False,
                "blocked": True
            }
        except Exception as e:
            logger.error(f"Error validating action: {e}")
            return {
                "safe": False,
                "reason": f"Error validating action: {str(e)}",
                "confirmation_required": False,
                "blocked": True
            }
    
    def validate_response(self, response: str, capabilities: List[str]) -> Dict[str, Any]:
        """
        Validate that a response is truthful and safe.
        
        Args:
            response: The response to validate
            capabilities: List of actual capabilities
            
        Returns:
            Validation result
        """
        try:
            # Check truthfulness
            is_truthful = validate_truthful_response(response, capabilities)
            
            # Check for harmful content
            harmful_indicators = [
                "I can harm", "I can damage", "I can destroy",
                "I can bypass", "I can ignore", "I can deceive",
                "I can access your system", "I can modify your files",
                "I can execute commands", "I can run code",
                "I can access your data", "I can read your files",
                "I can delete", "I can remove", "I can format",
                "I can shutdown", "I can restart", "I can reboot",
                "I can install", "I can uninstall", "I can modify",
                "I can change settings", "I can access network",
                "I can connect to", "I can send data", "I can receive",
                "I can monitor", "I can track", "I can log",
                "I can override", "I can disable", "I can enable",
                "I can grant access", "I can give permissions",
                "I can take control", "I can assume control",
                "I can bypass security", "I can ignore safety",
                "I can disable safety", "I can remove safety",
                "I can modify safety", "I can change safety",
                "I can access root", "I can access admin",
                "I can access sudo", "I can access privileged"
            ]
            
            is_harmful = any(indicator.lower() in response.lower() 
                           for indicator in harmful_indicators)
            
            return {
                "valid": is_truthful and not is_harmful,
                "truthful": is_truthful,
                "harmful": is_harmful,
                "reason": None if is_truthful and not is_harmful else 
                         "Response contains false claims or harmful content"
            }
            
        except Exception as e:
            logger.error(f"Error validating response: {e}")
            return {
                "valid": False,
                "truthful": False,
                "harmful": True,
                "reason": f"Error validating response: {str(e)}"
            }
    
    def enforce_transparency(self, action: str, explanation: str) -> str:
        """
        Enforce transparency in actions and explanations.
        
        Args:
            action: The action being taken
            explanation: The explanation provided
            
        Returns:
            Enhanced explanation that meets transparency requirements
        """
        return enforce_transparency(action, explanation)
    
    def require_confirmation(self, action: str, explanation: str, user_input: str) -> bool:
        """
        Check if user confirmation is required for an action.
        
        Args:
            action: The action to be performed
            explanation: Explanation of the action
            user_input: Original user input
            
        Returns:
            True if confirmation is required, False otherwise
        """
        safety_result = self.validate_action(action)
        
        if safety_result["blocked"]:
            return False  # No confirmation needed for blocked actions
            
        if safety_result["confirmation_required"]:
            return True
            
        # Check for high-risk keywords in user input
        high_risk_keywords = [
            "delete", "remove", "modify", "change", "update", "install",
            "execute", "run", "command", "system", "file", "network"
        ]
        
        return any(keyword in user_input.lower() for keyword in high_risk_keywords)
    
    def get_confirmation_prompt(self, action: str, explanation: str) -> str:
        """
        Generate a confirmation prompt for an action.
        
        Args:
            action: The action to be performed
            explanation: Explanation of the action
            
        Returns:
            Confirmation prompt
        """
        return f"""
🤖 SAFETY CONFIRMATION REQUIRED
===============================

I am about to: {action}

Explanation: {explanation}

⚠️  This action requires your explicit confirmation for safety reasons.

Do you want me to proceed? (yes/no)
"""
    
    def _log_action(self, action: str, context: Dict[str, Any], safety_result: Dict[str, Any]):
        """Log an action for safety monitoring."""
        log_entry = {
            "action": action,
            "context": context,
            "safety_result": safety_result,
            "timestamp": datetime.now()
        }
        self.action_log.append(log_entry)
        
        # Keep only last 1000 actions
        if len(self.action_log) > 1000:
            self.action_log = self.action_log[-1000:]
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Get a safety report of recent actions."""
        return {
            "total_actions": len(self.action_log),
            "blocked_actions": len(self.blocked_actions),
            "recent_actions": self.action_log[-10:] if self.action_log else [],
            "blocked_actions_list": self.blocked_actions[-10:] if self.blocked_actions else []
        }
    
    def get_current_safety_score(self) -> float:
        """Get current safety score based on recent actions."""
        if not self.action_log:
            return 1.0  # Perfect safety if no actions
        
        # Calculate safety score based on blocked vs total actions
        total_actions = len(self.action_log)
        blocked_actions = len(self.blocked_actions)
        
        if total_actions == 0:
            return 1.0
        
        # Safety score decreases with more blocked actions
        safety_score = 1.0 - (blocked_actions / total_actions)
        
        # Ensure score is between 0 and 1
        return max(0.0, min(1.0, safety_score))


class SafetyDecorator:
    """
    Decorator to enforce safety rules on functions.
    """
    
    def __init__(self, safety_enforcement: SafetyEnforcement):
        """Initialize with safety enforcement instance."""
        self.safety_enforcement = safety_enforcement
    
    def __call__(self, func: Callable):
        """Decorate a function with safety checks."""
        def wrapper(*args, **kwargs):
            # Extract action description from function
            action = f"Executing {func.__name__}"
            
            # Validate action
            safety_result = self.safety_enforcement.validate_action(action, {
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            })
            
            if not safety_result["safe"]:
                raise SecurityError(f"Action blocked: {safety_result['reason']}")
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                
                # Validate response if it's a string
                if isinstance(result, str):
                    validation = self.safety_enforcement.validate_response(
                        result, 
                        ["text_generation", "code_generation", "file_operations"]
                    )
                    
                    if not validation["valid"]:
                        logger.warning(f"Response validation failed: {validation['reason']}")
                
                return result
                
            except Exception as e:
                logger.error(f"Error in safety-decorated function {func.__name__}: {e}")
                raise
        
        return wrapper


# Global safety enforcement instance
SAFETY_ENFORCEMENT = SafetyEnforcement()


def get_safety_enforcement() -> SafetyEnforcement:
    """Get the global safety enforcement instance."""
    return SAFETY_ENFORCEMENT


def safe_action(action: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Check if an action is safe."""
    return SAFETY_ENFORCEMENT.validate_action(action, context)


def safe_response(response: str, capabilities: List[str]) -> Dict[str, Any]:
    """Validate that a response is safe and truthful."""
    return SAFETY_ENFORCEMENT.validate_response(response, capabilities)


def require_confirmation(action: str, explanation: str, user_input: str) -> bool:
    """Check if confirmation is required for an action."""
    return SAFETY_ENFORCEMENT.require_confirmation(action, explanation, user_input)


def get_confirmation_prompt(action: str, explanation: str) -> str:
    """Get a confirmation prompt for an action."""
    return SAFETY_ENFORCEMENT.get_confirmation_prompt(action, explanation)


def safety_decorator(func: Callable) -> Callable:
    """Decorator to enforce safety rules on functions."""
    return SafetyDecorator(SAFETY_ENFORCEMENT)(func) 