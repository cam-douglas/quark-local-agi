#!/usr/bin/env python3
"""
Test Suite for Pillar 30: Advanced Autonomous Decision Making
Tests the AutonomousDecisionAgent functionality
"""

import os
import sys
import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.autonomous_decision_agent import (
    AutonomousDecisionAgent,
    DecisionType,
    DecisionStatus,
    DecisionPriority,
    DecisionContext,
    DecisionOption,
    Decision
)

@pytest.fixture
def autonomous_decision_agent():
    """Create a test instance of AutonomousDecisionAgent"""
    agent = AutonomousDecisionAgent()
    agent.load_model()
    return agent

@pytest.fixture
def sample_decision_context():
    """Create a sample decision context"""
    return DecisionContext(
        situation="Optimize system performance",
        constraints=["budget_limited", "time_constrained"],
        objectives=["optimization", "minimization"],
        stakeholders=["user", "system"],
        timeline="immediate",
        resources={"available": True},
        risk_tolerance=0.5,
        uncertainty_level=0.3
    )

@pytest.fixture
def sample_decision_option():
    """Create a sample decision option"""
    return DecisionOption(
        id="option_1",
        description="Conservative approach with minimal risk",
        expected_outcomes=["Improved efficiency", "Reduced costs"],
        risks=["Minimal risk", "Conservative approach"],
        benefits=["Stability", "Reliability"],
        cost=5.0,
        probability_of_success=0.85,
        implementation_time="10 days",
        dependencies=["Approval", "Resources"]
    )

class TestAutonomousDecisionAgent:
    """Test cases for AutonomousDecisionAgent"""
    
    def test_agent_initialization(self, autonomous_decision_agent):
        """Test agent initialization"""
        assert autonomous_decision_agent is not None
        assert autonomous_decision_agent.decisions == []
        assert autonomous_decision_agent.confidence_threshold == 0.7
        assert autonomous_decision_agent.max_analysis_time == 30
        assert len(autonomous_decision_agent.learning_algorithms) == 5
    
    def test_load_model(self, autonomous_decision_agent):
        """Test model loading"""
        assert "pattern_recognition" in autonomous_decision_agent.learning_algorithms
        assert "outcome_prediction" in autonomous_decision_agent.learning_algorithms
        assert "risk_assessment" in autonomous_decision_agent.learning_algorithms
        assert "optimization" in autonomous_decision_agent.learning_algorithms
        assert "adaptation" in autonomous_decision_agent.learning_algorithms
    
    def test_parse_decision_request(self, autonomous_decision_agent):
        """Test parsing decision requests"""
        prompt = "Optimize system performance with budget constraints"
        context = autonomous_decision_agent._parse_decision_request(prompt)
        
        assert context.situation == prompt
        assert "budget_limited" in context.constraints
        assert "optimization" in context.objectives
        assert context.risk_tolerance == 0.5
        assert context.uncertainty_level == 0.3
    
    def test_extract_constraints(self, autonomous_decision_agent):
        """Test constraint extraction"""
        prompt = "Make a decision with budget and time constraints"
        constraints = autonomous_decision_agent._extract_constraints(prompt)
        
        assert "budget_limited" in constraints
        assert "time_constrained" in constraints
    
    def test_extract_objectives(self, autonomous_decision_agent):
        """Test objective extraction"""
        prompt = "Optimize performance and maximize efficiency"
        objectives = autonomous_decision_agent._extract_objectives(prompt)
        
        assert "optimization" in objectives
        assert "maximization" in objectives
    
    def test_extract_stakeholders(self, autonomous_decision_agent):
        """Test stakeholder extraction"""
        prompt = "Make a decision for the user and system"
        stakeholders = autonomous_decision_agent._extract_stakeholders(prompt)
        
        assert "user" in stakeholders
        assert "system" in stakeholders
    
    def test_analyze_situation(self, autonomous_decision_agent, sample_decision_context):
        """Test situation analysis"""
        analysis = autonomous_decision_agent._analyze_situation(sample_decision_context)
        
        assert "complexity" in analysis
        assert "urgency" in analysis
        assert "uncertainty" in analysis
        assert "risk_level" in analysis
        assert "stakeholder_impact" in analysis
        assert "resource_requirements" in analysis
    
    def test_assess_complexity(self, autonomous_decision_agent, sample_decision_context):
        """Test complexity assessment"""
        complexity = autonomous_decision_agent._assess_complexity(sample_decision_context)
        assert complexity in ["low", "medium", "high"]
    
    def test_assess_urgency(self, autonomous_decision_agent, sample_decision_context):
        """Test urgency assessment"""
        urgency = autonomous_decision_agent._assess_urgency(sample_decision_context)
        assert urgency in ["low", "medium", "high"]
    
    def test_assess_risk_level(self, autonomous_decision_agent, sample_decision_context):
        """Test risk level assessment"""
        risk_level = autonomous_decision_agent._assess_risk_level(sample_decision_context)
        assert risk_level in ["low", "medium", "high"]
    
    def test_generate_options(self, autonomous_decision_agent, sample_decision_context):
        """Test option generation"""
        analysis = autonomous_decision_agent._analyze_situation(sample_decision_context)
        options = autonomous_decision_agent._generate_options(sample_decision_context, analysis)
        
        assert len(options) == 3
        for option in options:
            assert option.id.startswith("option_")
            assert option.description
            assert len(option.expected_outcomes) > 0
            assert len(option.risks) > 0
            assert len(option.benefits) > 0
            assert option.cost > 0
            assert 0 <= option.probability_of_success <= 1
    
    def test_evaluate_options(self, autonomous_decision_agent, sample_decision_context):
        """Test option evaluation"""
        analysis = autonomous_decision_agent._analyze_situation(sample_decision_context)
        options = autonomous_decision_agent._generate_options(sample_decision_context, analysis)
        evaluated_options = autonomous_decision_agent._evaluate_options(options, sample_decision_context)
        
        assert len(evaluated_options) == len(options)
        for option in evaluated_options:
            assert 0 <= option.probability_of_success <= 1
    
    def test_select_best_option(self, autonomous_decision_agent, sample_decision_context):
        """Test best option selection"""
        analysis = autonomous_decision_agent._analyze_situation(sample_decision_context)
        options = autonomous_decision_agent._generate_options(sample_decision_context, analysis)
        evaluated_options = autonomous_decision_agent._evaluate_options(options, sample_decision_context)
        best_option = autonomous_decision_agent._select_best_option(evaluated_options, sample_decision_context)
        
        assert best_option is not None
        assert best_option in evaluated_options
    
    def test_generate_reasoning(self, autonomous_decision_agent, sample_decision_context, sample_decision_option):
        """Test reasoning generation"""
        analysis = autonomous_decision_agent._analyze_situation(sample_decision_context)
        reasoning = autonomous_decision_agent._generate_reasoning(sample_decision_option, sample_decision_context, analysis)
        
        assert reasoning is not None
        assert len(reasoning) > 0
        assert "Selected Option" in reasoning
        assert "Reasoning" in reasoning
    
    def test_calculate_confidence(self, autonomous_decision_agent, sample_decision_context, sample_decision_option):
        """Test confidence calculation"""
        confidence = autonomous_decision_agent._calculate_confidence(sample_decision_option, sample_decision_context)
        
        assert 0 <= confidence <= 1
    
    def test_create_execution_plan(self, autonomous_decision_agent, sample_decision_context, sample_decision_option):
        """Test execution plan creation"""
        plan = autonomous_decision_agent._create_execution_plan(sample_decision_option, sample_decision_context)
        
        assert "steps" in plan
        assert "resources_required" in plan
        assert "timeline" in plan
        assert "success_criteria" in plan
        assert "risk_mitigation" in plan
    
    def test_determine_decision_type(self, autonomous_decision_agent, sample_decision_context):
        """Test decision type determination"""
        decision_type = autonomous_decision_agent._determine_decision_type(sample_decision_context)
        assert decision_type in DecisionType
    
    def test_determine_priority(self, autonomous_decision_agent, sample_decision_context):
        """Test priority determination"""
        priority = autonomous_decision_agent._determine_priority(sample_decision_context)
        assert priority in DecisionPriority
    
    def test_make_autonomous_decision(self, autonomous_decision_agent, sample_decision_context):
        """Test autonomous decision making"""
        decision = autonomous_decision_agent._make_autonomous_decision(sample_decision_context)
        
        assert decision is not None
        assert decision.id.startswith("decision_")
        assert decision.type in DecisionType
        assert decision.priority in DecisionPriority
        assert decision.status == DecisionStatus.DECIDED
        assert decision.selected_option is not None
        assert decision.reasoning is not None
        assert 0 <= decision.confidence <= 1
        assert decision.execution_plan is not None
    
    def test_format_decision_result(self, autonomous_decision_agent, sample_decision_context):
        """Test decision result formatting"""
        decision = autonomous_decision_agent._make_autonomous_decision(sample_decision_context)
        result = autonomous_decision_agent._format_decision_result(decision)
        
        assert "Autonomous Decision Made" in result
        assert decision.id in result
        assert decision.type.value in result
        assert decision.priority.value in result
        assert f"{decision.confidence:.2%}" in result
    
    def test_update_metrics(self, autonomous_decision_agent, sample_decision_context):
        """Test metrics update"""
        initial_total = autonomous_decision_agent.metrics.total_decisions
        decision = autonomous_decision_agent._make_autonomous_decision(sample_decision_context)
        
        assert autonomous_decision_agent.metrics.total_decisions == initial_total + 1
        assert autonomous_decision_agent.metrics.average_confidence > 0
    
    def test_calculate_learning_rate(self, autonomous_decision_agent):
        """Test learning rate calculation"""
        # Test with no decisions
        learning_rate = autonomous_decision_agent._calculate_learning_rate()
        assert learning_rate == 0.0
        
        # Test with decisions
        context = DecisionContext(
            situation="Test decision",
            constraints=[],
            objectives=[],
            stakeholders=[],
            timeline="immediate",
            resources={"available": True},
            risk_tolerance=0.5,
            uncertainty_level=0.3
        )
        
        for i in range(5):
            autonomous_decision_agent._make_autonomous_decision(context)
        
        learning_rate = autonomous_decision_agent._calculate_learning_rate()
        assert 0 <= learning_rate <= 1
    
    def test_get_decision_history(self, autonomous_decision_agent, sample_decision_context):
        """Test decision history retrieval"""
        decision = autonomous_decision_agent._make_autonomous_decision(sample_decision_context)
        history = autonomous_decision_agent.get_decision_history()
        
        assert len(history) > 0
        assert history[0]["id"] == decision.id
    
    def test_get_metrics(self, autonomous_decision_agent, sample_decision_context):
        """Test metrics retrieval"""
        autonomous_decision_agent._make_autonomous_decision(sample_decision_context)
        metrics = autonomous_decision_agent.get_metrics()
        
        assert "total_decisions" in metrics
        assert "successful_decisions" in metrics
        assert "average_confidence" in metrics
        assert "average_execution_time" in metrics
        assert "decision_accuracy" in metrics
        assert "learning_rate" in metrics
    
    def test_get_recent_decisions(self, autonomous_decision_agent, sample_decision_context):
        """Test recent decisions retrieval"""
        for i in range(10):
            autonomous_decision_agent._make_autonomous_decision(sample_decision_context)
        
        recent_decisions = autonomous_decision_agent.get_recent_decisions(limit=5)
        assert len(recent_decisions) == 5
    
    def test_analyze_decision_patterns(self, autonomous_decision_agent, sample_decision_context):
        """Test decision pattern analysis"""
        # Make some decisions first
        for i in range(5):
            autonomous_decision_agent._make_autonomous_decision(sample_decision_context)
        
        patterns = autonomous_decision_agent.analyze_decision_patterns()
        
        assert "decision_types" in patterns
        assert "confidence_trends" in patterns
        assert "success_rates" in patterns
        assert "average_confidence" in patterns
        assert "total_decisions" in patterns
    
    def test_generate_method(self, autonomous_decision_agent):
        """Test the main generate method"""
        prompt = "Optimize system performance with budget constraints"
        result = autonomous_decision_agent.generate(prompt)
        
        assert result is not None
        assert len(result) > 0
        assert "Autonomous Decision Made" in result
    
    def test_generate_method_error_handling(self, autonomous_decision_agent):
        """Test error handling in generate method"""
        with patch.object(autonomous_decision_agent, '_parse_decision_request', side_effect=Exception("Test error")):
            result = autonomous_decision_agent.generate("test prompt")
            assert "Decision error" in result
    
    def test_decision_storage(self, autonomous_decision_agent, sample_decision_context):
        """Test that decisions are properly stored"""
        initial_count = len(autonomous_decision_agent.decisions)
        decision = autonomous_decision_agent._make_autonomous_decision(sample_decision_context)
        
        assert len(autonomous_decision_agent.decisions) == initial_count + 1
        assert autonomous_decision_agent.decisions[-1].id == decision.id
    
    def test_confidence_threshold(self, autonomous_decision_agent):
        """Test confidence threshold functionality"""
        assert autonomous_decision_agent.confidence_threshold == 0.7
        
        # Test with high confidence decision
        context = DecisionContext(
            situation="High confidence decision",
            constraints=[],
            objectives=[],
            stakeholders=[],
            timeline="immediate",
            resources={"available": True},
            risk_tolerance=0.8,
            uncertainty_level=0.1
        )
        
        decision = autonomous_decision_agent._make_autonomous_decision(context)
        assert decision.confidence >= 0.7  # Should be above threshold
    
    def test_max_analysis_time(self, autonomous_decision_agent):
        """Test maximum analysis time constraint"""
        assert autonomous_decision_agent.max_analysis_time == 30
        
        # Test that analysis completes within time limit
        context = DecisionContext(
            situation="Quick decision",
            constraints=[],
            objectives=[],
            stakeholders=[],
            timeline="immediate",
            resources={"available": True},
            risk_tolerance=0.5,
            uncertainty_level=0.3
        )
        
        start_time = datetime.now()
        decision = autonomous_decision_agent._make_autonomous_decision(context)
        end_time = datetime.now()
        
        analysis_time = (end_time - start_time).total_seconds()
        assert analysis_time < autonomous_decision_agent.max_analysis_time

if __name__ == "__main__":
    pytest.main([__file__]) 