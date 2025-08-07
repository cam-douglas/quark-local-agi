#!/usr/bin/env python3
"""
Simple Tests for Pillar 5: Orchestrator & Multi-Agent Framework
===============================================================

Basic tests that don't require heavy dependencies to verify
the core functionality of the multi-agent framework.
"""

import pytest
import time
from datetime import datetime
from typing import Dict, Any

# Test the individual agent classes without heavy dependencies
from agents.base import Agent


class TestAgentBase:
    """Test the base agent functionality."""
    
    def test_agent_base_initialization(self):
        """Test that the base agent can be initialized."""
        # Create a simple test agent
        class TestAgent(Agent):
            def __init__(self):
                super().__init__("test")
            
            def load_model(self):
                self.model = "test_model"
            
            def generate(self, prompt: str, **kwargs):
                return f"Test response to: {prompt}"
        
        agent = TestAgent()
        assert agent is not None
        assert agent.name == "test"
        assert agent.model == "test_model"
    
    def test_agent_generate_method(self):
        """Test that agents can generate responses."""
        class TestAgent(Agent):
            def __init__(self):
                super().__init__("test")
            
            def load_model(self):
                self.model = "test_model"
            
            def generate(self, prompt: str, **kwargs):
                return f"Test response to: {prompt}"
        
        agent = TestAgent()
        result = agent.generate("Hello, world!")
        assert result == "Test response to: Hello, world!"


class TestAgentFallbacks:
    """Test agent fallback mechanisms."""
    
    def test_nlu_fallback_mechanism(self):
        """Test NLU agent fallback when models are not available."""
        # Import with potential fallback
        try:
            from agents.nlu_agent import NLUAgent
            agent = NLUAgent()
            
            # Test basic functionality
            result = agent.generate("What is AI?")
            assert result is not None
            assert hasattr(result, 'primary_intent')
            assert hasattr(result, 'confidence')
            
        except Exception as e:
            # If import fails, that's okay - we're testing fallbacks
            assert "transformers" in str(e) or "torch" in str(e)
    
    def test_retrieval_fallback_mechanism(self):
        """Test retrieval agent fallback when models are not available."""
        try:
            from agents.retrieval_agent import RetrievalAgent
            agent = RetrievalAgent()
            
            # Test basic functionality
            result = agent.generate("artificial intelligence")
            assert result is not None
            assert hasattr(result, 'documents')
            assert hasattr(result, 'confidence')
            
        except Exception as e:
            # If import fails, that's okay - we're testing fallbacks
            assert "transformers" in str(e) or "torch" in str(e)
    
    def test_planning_fallback_mechanism(self):
        """Test planning agent fallback when models are not available."""
        try:
            from agents.planning_agent import PlanningAgent
            agent = PlanningAgent()
            
            # Test basic functionality
            result = agent.generate("Learn machine learning")
            assert result is not None
            assert hasattr(result, 'plan')
            assert hasattr(result, 'confidence')
            
        except Exception as e:
            # If import fails, that's okay - we're testing fallbacks
            assert "transformers" in str(e) or "torch" in str(e)


class TestDataStructures:
    """Test the data structures used in the orchestrator."""
    
    def test_agent_result_structure(self):
        """Test AgentResult data structure."""
        from core.orchestrator import AgentResult
        
        result = AgentResult(
            agent_name="test_agent",
            success=True,
            output="test output",
            execution_time=1.5,
            error=None,
            metadata={"test": "data"}
        )
        
        assert result.agent_name == "test_agent"
        assert result.success is True
        assert result.output == "test output"
        assert result.execution_time == 1.5
        assert result.error is None
        assert result.metadata["test"] == "data"
    
    def test_pipeline_result_structure(self):
        """Test PipelineResult data structure."""
        from core.orchestrator import PipelineResult, AgentResult
        
        agent_results = {
            "test_agent": AgentResult(
                agent_name="test_agent",
                success=True,
                output="test output",
                execution_time=1.5
            )
        }
        
        result = PipelineResult(
            category="test_category",
            confidence=0.8,
            pipeline=["test_agent"],
            agent_results=agent_results,
            final_response="Final test response",
            total_execution_time=2.0,
            parallel_execution=False
        )
        
        assert result.category == "test_category"
        assert result.confidence == 0.8
        assert result.pipeline == ["test_agent"]
        assert len(result.agent_results) == 1
        assert result.final_response == "Final test response"
        assert result.total_execution_time == 2.0
        assert result.parallel_execution is False


class TestPlanningDataStructures:
    """Test planning-related data structures."""
    
    def test_plan_step_structure(self):
        """Test PlanStep data structure."""
        from agents.planning_agent import PlanStep, PlanStatus, PlanPriority
        
        step = PlanStep(
            id="step_1",
            title="Test Step",
            description="This is a test step",
            status=PlanStatus.PENDING,
            priority=PlanPriority.MEDIUM,
            estimated_time=30.0,
            dependencies=[],
            resources=[],
            notes="Test notes",
            timestamp=datetime.now()
        )
        
        assert step.id == "step_1"
        assert step.title == "Test Step"
        assert step.description == "This is a test step"
        assert step.status == PlanStatus.PENDING
        assert step.priority == PlanPriority.MEDIUM
        assert step.estimated_time == 30.0
        assert step.dependencies == []
        assert step.resources == []
        assert step.notes == "Test notes"
    
    def test_plan_structure(self):
        """Test Plan data structure."""
        from agents.planning_agent import Plan, PlanStep, PlanStatus, PlanPriority
        
        steps = [
            PlanStep(
                id="step_1",
                title="Step 1",
                description="First step",
                status=PlanStatus.PENDING,
                priority=PlanPriority.MEDIUM,
                estimated_time=30.0,
                dependencies=[],
                resources=[],
                notes="",
                timestamp=datetime.now()
            )
        ]
        
        plan = Plan(
            id="plan_001",
            title="Test Plan",
            description="A test plan",
            goal="Test goal",
            steps=steps,
            total_estimated_time=30.0,
            priority=PlanPriority.MEDIUM,
            status=PlanStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        assert plan.id == "plan_001"
        assert plan.title == "Test Plan"
        assert plan.description == "A test plan"
        assert plan.goal == "Test goal"
        assert len(plan.steps) == 1
        assert plan.total_estimated_time == 30.0
        assert plan.priority == PlanPriority.MEDIUM
        assert plan.status == PlanStatus.PENDING


class TestOrchestratorConfiguration:
    """Test orchestrator configuration and constants."""
    
    def test_pipeline_definitions(self):
        """Test that pipeline definitions are properly structured."""
        from core.orchestrator import PIPELINES
        
        # Check that pipelines are defined
        assert isinstance(PIPELINES, dict)
        assert len(PIPELINES) > 0
        
        # Check that each pipeline is a list of strings
        for pipeline_name, pipeline in PIPELINES.items():
            assert isinstance(pipeline, list)
            assert all(isinstance(agent, str) for agent in pipeline)
    
    def test_parallel_agent_definitions(self):
        """Test that parallel agent definitions are properly structured."""
        from core.orchestrator import PARALLEL_AGENTS
        
        # Check that parallel agents are defined
        assert isinstance(PARALLEL_AGENTS, dict)
        assert len(PARALLEL_AGENTS) > 0
        
        # Check that each value is a boolean
        for agent_name, can_parallel in PARALLEL_AGENTS.items():
            assert isinstance(agent_name, str)
            assert isinstance(can_parallel, bool)


if __name__ == "__main__":
    pytest.main([__file__]) 