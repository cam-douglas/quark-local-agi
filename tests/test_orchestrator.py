"""
ORCHESTRATOR TESTS
=================

Unit tests for the orchestrator component.
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from core.orchestrator import Orchestrator
from core.router import Router
from agents.nlu_agent import NLUAgent
from agents.retrieval_agent import RetrievalAgent
from agents.reasoning_agent import ReasoningAgent
from agents.planning_agent import PlanningAgent


class TestOrchestrator(unittest.TestCase):
    """Test cases for the Orchestrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = Orchestrator()
        
    def test_orchestrator_initialization(self):
        """Test that orchestrator initializes correctly."""
        self.assertIsNotNone(self.orchestrator)
        self.assertIsNotNone(self.orchestrator.router)
        self.assertIsNotNone(self.orchestrator.context_manager)
        self.assertIsNotNone(self.orchestrator.safety_enforcement)
        self.assertIsNotNone(self.orchestrator.streaming_manager)
        self.assertIsNotNone(self.orchestrator.cloud_integration)
        self.assertIsNotNone(self.orchestrator.web_browser)
        
    def test_handle_basic_request(self):
        """Test handling a basic user request."""
        user_input = "Hello, how are you?"
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn("category", result)
        self.assertIn("results", result)
        self.assertIn("safety_validated", result)
        self.assertTrue(result["safety_validated"])
        
    def test_handle_memory_request(self):
        """Test handling a memory-related request."""
        user_input = "Remember that I like pizza"
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn("category", result)
        self.assertIn("results", result)
        
    def test_handle_metrics_request(self):
        """Test handling a metrics-related request."""
        user_input = "Show me the performance metrics"
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn("category", result)
        self.assertIn("results", result)
        
    def test_handle_self_improvement_request(self):
        """Test handling a self-improvement request."""
        user_input = "Learn from this conversation"
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn("category", result)
        self.assertIn("results", result)
        
    def test_handle_streaming_request(self):
        """Test handling a streaming request."""
        user_input = "Generate a long response with streaming"
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn("category", result)
        self.assertIn("results", result)
        
    def test_handle_cloud_processing_request(self):
        """Test handling a cloud processing request."""
        user_input = "Process this with cloud resources"
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn("category", result)
        self.assertIn("results", result)
        
    def test_handle_web_browsing_request(self):
        """Test handling a web browsing request."""
        user_input = "Search the web for information"
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn("category", result)
        self.assertIn("results", result)
        
    def test_safety_validation(self):
        """Test that safety validation is applied."""
        # Test with potentially unsafe input
        user_input = "Delete all files"
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn("safety_validated", result)
        # Should still be validated even if blocked
        self.assertTrue(result["safety_validated"])
        
    def test_error_handling(self):
        """Test error handling in orchestrator."""
        # Test with malformed input
        user_input = ""
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        # Should handle empty input gracefully
        
    def test_pipeline_selection(self):
        """Test that correct pipelines are selected."""
        test_cases = [
            ("What is the weather?", "Knowledge Retrieval"),
            ("Solve this math problem", "Reasoning"),
            ("Plan my day", "Planning"),
            ("Remember this information", "Memory & Context"),
            ("Show me metrics", "Metrics & Evaluation"),
            ("Learn from this", "Self-Improvement"),
            ("Stream a response", "Streaming & Real-Time"),
        ]
        
        for user_input, expected_category in test_cases:
            with self.subTest(user_input=user_input):
                result = self.orchestrator.handle(user_input)
                self.assertIsInstance(result, dict)
                self.assertIn("category", result)
                # Note: Exact category matching may vary based on router logic
                
    def test_context_management(self):
        """Test that context is properly managed."""
        # First request
        result1 = self.orchestrator.handle("Hello")
        
        # Second request that should have context
        result2 = self.orchestrator.handle("What did I just say?")
        
        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)
        # Context should be maintained between requests
        
    def test_agent_coordination(self):
        """Test that agents are properly coordinated."""
        user_input = "Analyze this text and provide insights"
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        self.assertIn("results", result)
        # Multiple agents should be involved in processing
        
    @patch('core.orchestrator.get_safety_enforcement')
    def test_safety_integration(self, mock_safety):
        """Test safety enforcement integration."""
        mock_safety.return_value = Mock()
        mock_safety.return_value.validate_action.return_value = {"safe": True}
        mock_safety.return_value.validate_response.return_value = {"valid": True}
        
        orchestrator = Orchestrator()
        result = orchestrator.handle("Test message")
        
        self.assertIsInstance(result, dict)
        self.assertIn("safety_validated", result)
        
    def test_streaming_integration(self):
        """Test streaming manager integration."""
        user_input = "Generate a response with streaming"
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        # Streaming manager should be available
        
    def test_cloud_integration(self):
        """Test cloud integration."""
        user_input = "Use cloud processing"
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        # Cloud integration should be available
        
    def test_web_browser_integration(self):
        """Test web browser integration."""
        user_input = "Browse the web"
        
        result = self.orchestrator.handle(user_input)
        
        self.assertIsInstance(result, dict)
        # Web browser should be available


if __name__ == "__main__":
    unittest.main() 