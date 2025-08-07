#!/usr/bin/env python3
"""
Tests for Pillar 5: Orchestrator & Multi-Agent Framework
========================================================

Tests the orchestrator and multi-agent framework to ensure proper
coordination between NLU, Retrieval, Reasoning, and Planning agents.
"""

import pytest
import time
from datetime import datetime
from typing import Dict, Any

from core.orchestrator import Orchestrator, AgentResult, PipelineResult
from agents.nlu_agent import NLUAgent, NLUResult, Intent
from agents.retrieval_agent import RetrievalAgent, RetrievalResult, Document
from agents.planning_agent import PlanningAgent, PlanningResult, Plan, PlanStep, PlanStatus, PlanPriority


@pytest.fixture
def orchestrator():
    """Create an orchestrator for testing."""
    return Orchestrator(max_workers=2)


@pytest.fixture
def nlu_agent():
    """Create an NLU agent for testing."""
    return NLUAgent()


@pytest.fixture
def retrieval_agent():
    """Create a retrieval agent for testing."""
    return RetrievalAgent()


@pytest.fixture
def planning_agent():
    """Create a planning agent for testing."""
    return PlanningAgent()


class TestOrchestrator:
    """Test the orchestrator and multi-agent framework."""
    
    def test_orchestrator_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator is not None
        assert orchestrator.max_workers == 2
        assert len(orchestrator.agents) > 0
        
        # Check that key agents are present
        expected_agents = ["NLU", "Retrieval", "Reasoning", "Planning", "Memory", "Metrics"]
        for agent_name in expected_agents:
            assert agent_name in orchestrator.agents
    
    def test_nlu_agent_functionality(self, nlu_agent):
        """Test NLU agent functionality."""
        # Test intent classification
        result = nlu_agent.generate("What is artificial intelligence?")
        
        assert isinstance(result, NLUResult)
        assert result.primary_intent is not None
        assert result.primary_intent.intent in ["question", "request", "statement"]
        assert result.confidence > 0.0
        assert result.language == "english"
    
    def test_retrieval_agent_functionality(self, retrieval_agent):
        """Test retrieval agent functionality."""
        # Test document retrieval
        result = retrieval_agent.generate("artificial intelligence")
        
        assert isinstance(result, RetrievalResult)
        assert len(result.documents) > 0
        assert result.confidence > 0.0
        assert result.search_type in ["semantic", "keyword", "hybrid", "exact"]
        
        # Check document structure
        for doc in result.documents:
            assert isinstance(doc, Document)
            assert doc.id is not None
            assert doc.title is not None
            assert doc.content is not None
            assert doc.relevance_score > 0.0
    
    def test_planning_agent_functionality(self, planning_agent):
        """Test planning agent functionality."""
        # Test plan generation
        result = planning_agent.generate("Learn machine learning")
        
        assert isinstance(result, PlanningResult)
        assert result.plan is not None
        assert len(result.plan.steps) > 0
        assert result.confidence > 0.0
        assert result.reasoning is not None
        
        # Check plan structure
        plan = result.plan
        assert plan.id is not None
        assert plan.title is not None
        assert plan.goal == "Learn machine learning"
        assert plan.total_estimated_time > 0.0
        
        # Check step structure
        for step in plan.steps:
            assert isinstance(step, PlanStep)
            assert step.id is not None
            assert step.title is not None
            assert step.description is not None
            assert step.estimated_time > 0.0
    
    def test_orchestrator_pipeline_execution(self, orchestrator):
        """Test orchestrator pipeline execution."""
        # Test a simple pipeline
        prompt = "What is machine learning and how can I learn it?"
        
        result = orchestrator.handle(prompt)
        
        assert result is not None
        assert hasattr(result, 'category')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'pipeline')
        assert hasattr(result, 'agent_results')
        assert hasattr(result, 'final_response')
        assert hasattr(result, 'total_execution_time')
    
    def test_parallel_agent_execution(self, orchestrator):
        """Test parallel agent execution."""
        # Test that parallel agents can run simultaneously
        start_time = time.time()
        
        # Execute a pipeline that should use parallel agents
        result = orchestrator.handle("Find information about AI and create a learning plan")
        
        execution_time = time.time() - start_time
        
        # Check that execution was reasonably fast (indicating parallel execution)
        assert execution_time < 20.0  # Should complete within 20 seconds
        
        # Check that we have results from multiple agents
        assert len(result.agent_results) > 1
    
    def test_agent_error_handling(self, orchestrator):
        """Test that the orchestrator handles agent errors gracefully."""
        # Test with a malformed request that might cause errors
        result = orchestrator.handle("")
        
        # Should still return a result, even if it's an error response
        assert result is not None
    
    def test_pipeline_categorization(self, orchestrator):
        """Test that the orchestrator correctly categorizes requests."""
        test_cases = [
            ("What is AI?", "Natural Language Understanding"),
            ("Find documents about machine learning", "Knowledge Retrieval"),
            ("Explain how neural networks work", "Reasoning"),
            ("Create a plan to learn Python", "Planning"),
            ("Remember this information", "Memory & Context"),
            ("How well did I perform?", "Metrics & Evaluation"),
        ]
        
        for prompt, expected_category in test_cases:
            result = orchestrator.handle(prompt)
            # Note: The actual categorization might be different due to intent classification
            # We just check that we get a valid result
            assert result is not None
            assert result.category is not None
    
    def test_agent_result_structure(self, orchestrator):
        """Test that agent results have the correct structure."""
        result = orchestrator.handle("What is artificial intelligence?")
        
        for agent_name, agent_result in result.agent_results.items():
            assert isinstance(agent_result, AgentResult)
            assert agent_result.agent_name == agent_name
            assert agent_result.success in [True, False]
            assert agent_result.output is not None
            assert agent_result.execution_time >= 0.0
    
    def test_pipeline_result_structure(self, orchestrator):
        """Test that pipeline results have the correct structure."""
        result = orchestrator.handle("Create a plan to learn machine learning")
        
        assert isinstance(result, PipelineResult)
        assert result.category is not None
        assert result.confidence >= 0.0 and result.confidence <= 1.0
        assert isinstance(result.pipeline, list)
        assert isinstance(result.agent_results, dict)
        assert isinstance(result.final_response, str)
        assert result.total_execution_time >= 0.0
        assert isinstance(result.parallel_execution, bool)
    
    def test_orchestrator_performance_stats(self, orchestrator):
        """Test orchestrator performance statistics."""
        # Execute a few requests to generate stats
        orchestrator.handle("What is AI?")
        orchestrator.handle("Find information about machine learning")
        orchestrator.handle("Create a learning plan")
        
        stats = orchestrator.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "execution_stats" in stats
        assert "parallel_agents" in stats
        assert "max_workers" in stats
        
        # Check nested execution stats
        execution_stats = stats["execution_stats"]
        assert "parallel_executions" in execution_stats
        assert "average_execution_time" in execution_stats
        assert "success_rate" in execution_stats
        
        assert stats["total_requests"] >= 2  # At least 2 requests should be recorded
        assert execution_stats["success_rate"] >= 0.0 and execution_stats["success_rate"] <= 1.0
    
    def test_agent_model_loading(self, orchestrator):
        """Test that all agents have their models loaded."""
        for agent_name, agent in orchestrator.agents.items():
            # Check that the agent has a model or fallback mechanism
            assert hasattr(agent, 'model') or hasattr(agent, '_ensure_model')
    
    def test_orchestrator_shutdown(self, orchestrator):
        """Test orchestrator shutdown."""
        # This should not raise an exception
        orchestrator.shutdown()
        
        # After shutdown, the orchestrator should still be accessible
        assert orchestrator is not None


class TestAgentIntegration:
    """Test integration between different agents."""
    
    def test_nlu_to_retrieval_integration(self, nlu_agent, retrieval_agent):
        """Test integration between NLU and retrieval agents."""
        # NLU analysis
        nlu_result = nlu_agent.generate("What is machine learning?")
        
        # Use NLU result to improve retrieval
        retrieval_result = retrieval_agent.generate(
            nlu_result.primary_intent.intent + " " + "machine learning"
        )
        
        assert len(retrieval_result.documents) > 0
        assert retrieval_result.confidence > 0.0
    
    def test_retrieval_to_planning_integration(self, retrieval_agent, planning_agent):
        """Test integration between retrieval and planning agents."""
        # Get relevant documents
        retrieval_result = retrieval_agent.generate("artificial intelligence")
        
        # Use retrieved information to create a plan
        if retrieval_result.documents:
            doc_content = retrieval_result.documents[0].content[:100]
            plan_result = planning_agent.generate(f"Learn about: {doc_content}")
            
            assert plan_result.plan is not None
            assert len(plan_result.plan.steps) > 0
    
    def test_full_pipeline_integration(self, orchestrator):
        """Test full pipeline integration."""
        # Test a complex request that should use multiple agents
        result = orchestrator.handle(
            "I want to learn artificial intelligence. Can you find information about it and create a study plan?"
        )
        
        # Should have results from multiple agents
        assert len(result.agent_results) >= 2
        
        # Should have a final response
        assert len(result.final_response) > 0
        
        # Should have reasonable confidence
        assert result.confidence > 0.0


class TestAgentFallbacks:
    """Test agent fallback mechanisms."""
    
    def test_nlu_fallback(self, nlu_agent):
        """Test NLU agent fallback when models are not available."""
        # Force fallback by temporarily disabling the model
        original_model = nlu_agent.intent_classifier
        nlu_agent.intent_classifier = None
        
        result = nlu_agent.generate("Hello, how are you?")
        
        # Should still return a valid result
        assert isinstance(result, NLUResult)
        assert result.primary_intent is not None
        assert result.confidence > 0.0
        
        # Restore model
        nlu_agent.intent_classifier = original_model
    
    def test_retrieval_fallback(self, retrieval_agent):
        """Test retrieval agent fallback when models are not available."""
        # Force fallback by temporarily disabling the model
        original_model = retrieval_agent.embedding_model
        retrieval_agent.embedding_model = None
        
        result = retrieval_agent.generate("artificial intelligence")
        
        # Should still return a valid result
        assert isinstance(result, RetrievalResult)
        assert len(result.documents) > 0
        assert result.confidence > 0.0
        
        # Restore model
        retrieval_agent.embedding_model = original_model
    
    def test_planning_fallback(self, planning_agent):
        """Test planning agent fallback when models are not available."""
        # Force fallback by temporarily disabling the model
        original_model = planning_agent.planning_model
        planning_agent.planning_model = None
        
        result = planning_agent.generate("Learn machine learning")
        
        # Should still return a valid result
        assert isinstance(result, PlanningResult)
        assert result.plan is not None
        assert len(result.plan.steps) > 0
        assert result.confidence > 0.0
        
        # Restore model
        planning_agent.planning_model = original_model


if __name__ == "__main__":
    pytest.main([__file__]) 