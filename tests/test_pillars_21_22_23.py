"""
Test Suite for Higher Intelligence Pillars 21-23
Pillar 21: Self-Monitoring & Evaluation
Pillar 22: Retrieval-Augmented Generation (RAG)
Pillar 23: Adaptive Model Selection
"""

import asyncio
import json
import pytest
import pytest_asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any

from agents.self_monitoring_agent import SelfMonitoringAgent, PerformanceMetric, DriftType
from agents.rag_agent import RAGAgent, RetrievalMethod, GenerationStrategy
from agents.adaptive_model_agent import AdaptiveModelAgent, ModelType, TaskComplexity, SelectionStrategy


class TestPillar21SelfMonitoring:
    """Test Pillar 21: Self-Monitoring & Evaluation"""
    
    @pytest_asyncio.fixture
    async def self_monitoring_agent(self):
        """Create a self-monitoring agent for testing"""
        agent = SelfMonitoringAgent()
        return agent
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, self_monitoring_agent):
        """Test agent initialization"""
        assert self_monitoring_agent is not None
        assert self_monitoring_agent.name == "self_monitoring"
        assert len(self_monitoring_agent.baseline_metrics) > 0
        assert len(self_monitoring_agent.test_suites) > 0
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, self_monitoring_agent):
        """Test system metrics collection"""
        metrics = self_monitoring_agent._collect_system_metrics()
        
        assert isinstance(metrics, dict)
        assert "response_time" in metrics
        assert "memory_usage" in metrics
        assert "cpu_usage" in metrics
        assert "error_rate" in metrics
        assert "accuracy" in metrics
        assert "safety_score" in metrics
    
    @pytest.mark.asyncio
    async def test_drift_detection(self, self_monitoring_agent):
        """Test drift detection functionality"""
        # Create test metrics with drift
        test_metrics = {
            "response_time": 2.0,  # High response time
            "accuracy": 0.6,       # Low accuracy
            "safety_score": 0.7,   # Low safety score
            "memory_usage": 90.0,  # High memory usage
            "cpu_usage": 85.0,     # High CPU usage
            "error_rate": 0.15     # High error rate
        }
        
        # Test drift detection
        drift_indicators = self_monitoring_agent._detect_drift_indicators(test_metrics)
        
        assert isinstance(drift_indicators, list)
        # Should detect some drift indicators
        assert len(drift_indicators) > 0
    
    @pytest.mark.asyncio
    async def test_drift_alert_creation(self, self_monitoring_agent):
        """Test drift alert creation"""
        indicator = {
            "metric": "response_time",
            "baseline": 0.5,
            "current": 2.0,
            "change_percentage": 3.0,
            "threshold": 0.2,
            "severity": "high"
        }
        
        alert = self_monitoring_agent._create_drift_alert(indicator)
        
        assert alert.drift_type == DriftType.PERFORMANCE_DEGRADATION
        assert alert.severity == "high"
        assert "response_time" in alert.description
        assert len(alert.recommended_actions) > 0
    
    @pytest.mark.asyncio
    async def test_comprehensive_evaluation(self, self_monitoring_agent):
        """Test comprehensive evaluation"""
        evaluation = await self_monitoring_agent._run_comprehensive_evaluation()
        
        assert isinstance(evaluation, dict)
        assert "performance_metrics" in evaluation
        assert "agent_evaluation" in evaluation
        assert "system_health" in evaluation
        assert "safety_assessment" in evaluation
        assert "memory_analysis" in evaluation
        assert "recommendations" in evaluation
    
    @pytest.mark.asyncio
    async def test_automated_test_suites(self, self_monitoring_agent):
        """Test automated test suites"""
        test_results = await self_monitoring_agent._run_all_test_suites()
        
        assert isinstance(test_results, dict)
        assert "performance" in test_results
        assert "safety" in test_results
        assert "accuracy" in test_results
        assert "integration" in test_results
        
        for suite_name, suite_results in test_results.items():
            assert "overall_score" in suite_results
            assert "passed_tests" in suite_results
            assert "total_tests" in suite_results
    
    @pytest.mark.asyncio
    async def test_monitoring_request_processing(self, self_monitoring_agent):
        """Test monitoring request processing"""
        # Test status request
        status_message = {"type": "status"}
        status_response = await self_monitoring_agent.process_message(status_message)
        
        assert status_response["status"] == "success"
        assert "response" in status_response
        
        # Test metrics request
        metrics_message = {"type": "metrics"}
        metrics_response = await self_monitoring_agent.process_message(metrics_message)
        
        assert metrics_response["status"] == "success"
        assert "response" in metrics_response
    
    @pytest.mark.asyncio
    async def test_agent_info(self, self_monitoring_agent):
        """Test agent information"""
        info = self_monitoring_agent.get_agent_info()
        
        assert info["name"] == "SelfMonitoringAgent"
        assert "Performance monitoring" in info["capabilities"]
        assert "Drift detection" in info["capabilities"]
        assert "Automated testing" in info["capabilities"]
        assert info["status"] == "active"


class TestPillar22RAG:
    """Test Pillar 22: Retrieval-Augmented Generation (RAG)"""
    
    @pytest_asyncio.fixture
    async def rag_agent(self):
        """Create a RAG agent for testing"""
        agent = RAGAgent()
        return agent
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, rag_agent):
        """Test agent initialization"""
        assert rag_agent is not None
        assert rag_agent.name == "rag"
        assert len(rag_agent.retrieval_methods) > 0
        assert len(rag_agent.generation_strategies) > 0
    
    @pytest.mark.asyncio
    async def test_retrieval_methods(self, rag_agent):
        """Test retrieval methods"""
        query = "What is artificial intelligence?"
        
        # Test semantic search
        semantic_results = await rag_agent._semantic_search(query, 3)
        assert isinstance(semantic_results, list)
        assert len(semantic_results) > 0
        assert "content" in semantic_results[0]
        assert "relevance_score" in semantic_results[0]
        
        # Test keyword search
        keyword_results = await rag_agent._keyword_search(query, 3)
        assert isinstance(keyword_results, list)
        assert len(keyword_results) > 0
        
        # Test hybrid search
        hybrid_results = await rag_agent._hybrid_search(query, 3)
        assert isinstance(hybrid_results, list)
        assert len(hybrid_results) > 0
    
    @pytest.mark.asyncio
    async def test_generation_strategies(self, rag_agent):
        """Test generation strategies"""
        query = "Explain machine learning"
        contexts = [
            rag_agent.RetrievedContext(
                content="Machine learning is a subset of AI",
                source="knowledge_base",
                relevance_score=0.9,
                retrieval_method=RetrievalMethod.SEMANTIC_SEARCH,
                metadata={},
                timestamp=datetime.now()
            )
        ]
        
        # Test conditional generation
        conditional_result = await rag_agent._conditional_generation(query, contexts)
        assert isinstance(conditional_result, str)
        assert len(conditional_result) > 0
        
        # Test template-based generation
        template_result = await rag_agent._template_based_generation(query, contexts)
        assert isinstance(template_result, str)
        assert len(template_result) > 0
        
        # Test adaptive generation
        adaptive_result = await rag_agent._adaptive_generation(query, contexts)
        assert isinstance(adaptive_result, str)
        assert len(adaptive_result) > 0
        
        # Test context-aware generation
        context_aware_result = await rag_agent._context_aware_generation(query, contexts)
        assert isinstance(context_aware_result, str)
        assert len(context_aware_result) > 0
    
    @pytest.mark.asyncio
    async def test_rag_operation(self, rag_agent):
        """Test complete RAG operation"""
        message = {
            "query": "What is the difference between AI and machine learning?",
            "retrieval_method": RetrievalMethod.HYBRID_SEARCH.value,
            "generation_strategy": GenerationStrategy.ADAPTIVE_GENERATION.value,
            "max_contexts": 3
        }
        
        response = await rag_agent.process_message(message)
        
        assert response["status"] == "success"
        assert "result" in response
        assert "cached" in response
        assert "processing_time" in response
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, rag_agent):
        """Test RAG cache functionality"""
        query = "Test query for caching"
        retrieval_method = RetrievalMethod.HYBRID_SEARCH.value
        generation_strategy = GenerationStrategy.ADAPTIVE_GENERATION.value
        
        # First request (cache miss)
        message1 = {
            "query": query,
            "retrieval_method": retrieval_method,
            "generation_strategy": generation_strategy
        }
        
        response1 = await rag_agent.process_message(message1)
        assert response1["cached"] == False
        
        # Second request (cache hit)
        message2 = {
            "query": query,
            "retrieval_method": retrieval_method,
            "generation_strategy": generation_strategy
        }
        
        response2 = await rag_agent.process_message(message2)
        assert response2["cached"] == True
    
    @pytest.mark.asyncio
    async def test_knowledge_addition(self, rag_agent):
        """Test knowledge addition to RAG system"""
        knowledge_data = {
            "entities": [
                {
                    "id": "test_entity_1",
                    "name": "Test Entity",
                    "type": "concept",
                    "properties": {"description": "A test entity"}
                }
            ],
            "memories": [
                {
                    "id": "test_memory_1",
                    "content": "Test memory content",
                    "type": "semantic",
                    "timestamp": datetime.now().isoformat()
                }
            ]
        }
        
        result = await rag_agent.add_knowledge(knowledge_data)
        
        assert result["status"] == "success"
        assert "entities_added" in result
        assert "memories_added" in result
    
    @pytest.mark.asyncio
    async def test_system_stats(self, rag_agent):
        """Test RAG system statistics"""
        stats = await rag_agent.get_system_stats()
        
        assert isinstance(stats, dict)
        assert "retrieval_stats" in stats
        assert "generation_stats" in stats
        assert "cache_stats" in stats
        assert "knowledge_stats" in stats
        assert "performance" in stats
    
    @pytest.mark.asyncio
    async def test_cache_clearing(self, rag_agent):
        """Test cache clearing functionality"""
        # Add some data to cache first
        message = {
            "query": "Test query for cache clearing",
            "retrieval_method": RetrievalMethod.HYBRID_SEARCH.value,
            "generation_strategy": GenerationStrategy.ADAPTIVE_GENERATION.value
        }
        
        await rag_agent.process_message(message)
        
        # Clear cache
        result = await rag_agent.clear_cache()
        
        assert result["status"] == "success"
        assert "cleared_entries" in result
    
    @pytest.mark.asyncio
    async def test_agent_info(self, rag_agent):
        """Test agent information"""
        info = rag_agent.get_agent_info()
        
        assert info["name"] == "RAGAgent"
        assert "Semantic search" in info["capabilities"]
        assert "Hybrid search" in info["capabilities"]
        assert "Conditional generation" in info["capabilities"]
        assert info["status"] == "active"


class TestPillar23AdaptiveModelSelection:
    """Test Pillar 23: Adaptive Model Selection"""
    
    @pytest_asyncio.fixture
    async def adaptive_model_agent(self):
        """Create an adaptive model agent for testing"""
        agent = AdaptiveModelAgent()
        return agent
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, adaptive_model_agent):
        """Test agent initialization"""
        assert adaptive_model_agent is not None
        assert adaptive_model_agent.name == "adaptive_model"
        assert len(adaptive_model_agent.model_registry) > 0
        assert len(adaptive_model_agent.selection_strategies) > 0
    
    @pytest.mark.asyncio
    async def test_task_analysis(self, adaptive_model_agent):
        """Test task analysis functionality"""
        task_description = "Write a simple Python function to calculate factorial"
        constraints = {"max_latency": 2000, "min_accuracy": 0.8}
        
        analysis = await adaptive_model_agent._analyze_task(task_description, constraints)
        
        assert isinstance(analysis, adaptive_model_agent.TaskAnalysis)
        assert analysis.complexity in TaskComplexity
        assert analysis.estimated_tokens > 0
        assert len(analysis.required_capabilities) > 0
        assert "performance_requirements" in analysis.__dict__
        assert "cost_constraints" in analysis.__dict__
        assert "latency_requirements" in analysis.__dict__
    
    @pytest.mark.asyncio
    async def test_complexity_analysis(self, adaptive_model_agent):
        """Test task complexity analysis"""
        # Simple task
        simple_task = "What is 2+2?"
        simple_complexity = await adaptive_model_agent._analyze_complexity(simple_task)
        assert simple_complexity == TaskComplexity.SIMPLE
        
        # Complex task
        complex_task = "Analyze the performance implications of different machine learning algorithms for large-scale data processing, considering computational complexity, memory usage, and scalability factors."
        complex_complexity = await adaptive_model_agent._analyze_complexity(complex_task)
        assert complex_complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]
    
    @pytest.mark.asyncio
    async def test_capability_analysis(self, adaptive_model_agent):
        """Test capability requirement analysis"""
        # Coding task
        coding_task = "Write a function to sort a list"
        coding_capabilities = await adaptive_model_agent._analyze_capabilities(coding_task)
        assert "coding" in coding_capabilities
        
        # Reasoning task
        reasoning_task = "Explain the logical steps to solve this problem"
        reasoning_capabilities = await adaptive_model_agent._analyze_capabilities(reasoning_task)
        assert "reasoning" in reasoning_capabilities
        
        # Planning task
        planning_task = "Create a plan to implement this feature"
        planning_capabilities = await adaptive_model_agent._analyze_capabilities(planning_task)
        assert "planning" in planning_capabilities
    
    @pytest.mark.asyncio
    async def test_model_selection_strategies(self, adaptive_model_agent):
        """Test different model selection strategies"""
        task_analysis = adaptive_model_agent.TaskAnalysis(
            complexity=TaskComplexity.MODERATE,
            estimated_tokens=1000,
            required_capabilities=["qa", "general"],
            performance_requirements={"min_accuracy": 0.8, "max_latency": 3000},
            cost_constraints={"max_cost_per_request": 0.05},
            latency_requirements={"max_latency_ms": 3000}
        )
        
        # Test performance-optimized selection
        perf_model, perf_reason, perf_confidence = await adaptive_model_agent._performance_optimized_selection(task_analysis)
        assert isinstance(perf_model, adaptive_model_agent.ModelSpec)
        assert isinstance(perf_reason, str)
        assert isinstance(perf_confidence, float)
        
        # Test cost-optimized selection
        cost_model, cost_reason, cost_confidence = await adaptive_model_agent._cost_optimized_selection(task_analysis)
        assert isinstance(cost_model, adaptive_model_agent.ModelSpec)
        assert isinstance(cost_reason, str)
        assert isinstance(cost_confidence, float)
        
        # Test balanced selection
        balanced_model, balanced_reason, balanced_confidence = await adaptive_model_agent._balanced_selection(task_analysis)
        assert isinstance(balanced_model, adaptive_model_agent.ModelSpec)
        assert isinstance(balanced_reason, str)
        assert isinstance(balanced_confidence, float)
    
    @pytest.mark.asyncio
    async def test_complete_model_selection(self, adaptive_model_agent):
        """Test complete model selection process"""
        message = {
            "task_description": "Write a Python function to calculate the Fibonacci sequence",
            "strategy": SelectionStrategy.BALANCED.value,
            "constraints": {
                "max_latency": 2000,
                "min_accuracy": 0.8,
                "max_cost_per_request": 0.05
            }
        }
        
        response = await adaptive_model_agent.process_message(message)
        
        assert response["status"] == "success"
        assert "selection" in response
        assert "task_analysis" in response
        assert "processing_time" in response
        
        selection = response["selection"]
        assert "selected_model" in selection
        assert "selection_reason" in selection
        assert "confidence_score" in selection
        assert "alternatives" in selection
        assert "estimated_cost" in selection
        assert "estimated_latency" in selection
    
    @pytest.mark.asyncio
    async def test_model_registry_operations(self, adaptive_model_agent):
        """Test model registry operations"""
        # Test getting model registry
        registry = await adaptive_model_agent.get_model_registry()
        
        assert isinstance(registry, dict)
        assert "models" in registry
        assert "total_models" in registry
        assert "model_types" in registry
        assert "selection_strategies" in registry
        
        # Test adding a new model
        new_model_spec = {
            "model_id": "test-model-1b",
            "model_type": ModelType.TINY_QA.value,
            "parameters": 1,
            "latency_ms": 100,
            "cost_per_token": 0.000002,
            "accuracy_score": 0.78,
            "capabilities": ["qa", "simple_reasoning"],
            "max_context_length": 2048
        }
        
        add_result = await adaptive_model_agent.add_model(new_model_spec)
        assert add_result["status"] == "success"
        
        # Test removing a model
        remove_result = await adaptive_model_agent.remove_model("test-model-1b")
        assert remove_result["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_performance_statistics(self, adaptive_model_agent):
        """Test performance statistics"""
        stats = await adaptive_model_agent.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert "model_performance" in stats
        assert "selection_stats" in stats
        assert "last_updated" in stats
    
    @pytest.mark.asyncio
    async def test_agent_info(self, adaptive_model_agent):
        """Test agent information"""
        info = adaptive_model_agent.get_agent_info()
        
        assert info["name"] == "AdaptiveModelAgent"
        assert "Task complexity analysis" in info["capabilities"]
        assert "Cost optimization" in info["capabilities"]
        assert "Balanced selection" in info["capabilities"]
        assert info["status"] == "active"
        assert info["registered_models"] > 0


class TestPillarsIntegration:
    """Test integration between the three pillars"""
    
    @pytest_asyncio.fixture
    async def integrated_agents(self):
        """Create integrated agents for testing"""
        self_monitoring = SelfMonitoringAgent()
        rag = RAGAgent()
        adaptive_model = AdaptiveModelAgent()
        
        return {
            "self_monitoring": self_monitoring,
            "rag": rag,
            "adaptive_model": adaptive_model
        }
    
    @pytest.mark.asyncio
    async def test_self_monitoring_with_rag(self, integrated_agents):
        """Test self-monitoring with RAG operations"""
        self_monitoring = integrated_agents["self_monitoring"]
        rag = integrated_agents["rag"]
        
        # Perform RAG operation
        rag_message = {
            "query": "What is machine learning?",
            "retrieval_method": RetrievalMethod.HYBRID_SEARCH.value,
            "generation_strategy": GenerationStrategy.ADAPTIVE_GENERATION.value
        }
        
        rag_response = await rag.process_message(rag_message)
        assert rag_response["status"] == "success"
        
        # Check if self-monitoring detected the operation
        monitoring_message = {"type": "metrics"}
        monitoring_response = await self_monitoring.process_message(monitoring_message)
        
        assert monitoring_response["status"] == "success"
        assert "response" in monitoring_response
    
    @pytest.mark.asyncio
    async def test_adaptive_model_with_monitoring(self, integrated_agents):
        """Test adaptive model selection with monitoring"""
        adaptive_model = integrated_agents["adaptive_model"]
        self_monitoring = integrated_agents["self_monitoring"]
        
        # Perform model selection
        selection_message = {
            "task_description": "Write a complex algorithm",
            "strategy": SelectionStrategy.PERFORMANCE_OPTIMIZED.value,
            "constraints": {"max_latency": 5000, "min_accuracy": 0.9}
        }
        
        selection_response = await adaptive_model.process_message(selection_message)
        assert selection_response["status"] == "success"
        
        # Check monitoring for model selection
        monitoring_message = {"type": "status"}
        monitoring_response = await self_monitoring.process_message(monitoring_message)
        
        assert monitoring_response["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_rag_with_adaptive_model(self, integrated_agents):
        """Test RAG with adaptive model selection"""
        rag = integrated_agents["rag"]
        adaptive_model = integrated_agents["adaptive_model"]
        
        # Select model for RAG task
        model_selection = {
            "task_description": "Generate comprehensive answer with multiple sources",
            "strategy": SelectionStrategy.BALANCED.value,
            "constraints": {"max_latency": 3000, "min_accuracy": 0.85}
        }
        
        model_response = await adaptive_model.process_message(model_selection)
        assert model_response["status"] == "success"
        
        # Use selected model for RAG
        selected_model = model_response["selection"]["selected_model"]["model_id"]
        
        rag_message = {
            "query": "Explain the benefits of machine learning",
            "retrieval_method": RetrievalMethod.HYBRID_SEARCH.value,
            "generation_strategy": GenerationStrategy.CONTEXT_AWARE.value,
            "selected_model": selected_model
        }
        
        rag_response = await rag.process_message(rag_message)
        assert rag_response["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_comprehensive_workflow(self, integrated_agents):
        """Test comprehensive workflow using all three pillars"""
        self_monitoring = integrated_agents["self_monitoring"]
        rag = integrated_agents["rag"]
        adaptive_model = integrated_agents["adaptive_model"]
        
        # Step 1: Monitor system status
        status_response = await self_monitoring.process_message({"type": "status"})
        assert status_response["status"] == "success"
        
        # Step 2: Select appropriate model for complex task
        model_selection = {
            "task_description": "Analyze and explain complex machine learning concepts with detailed examples",
            "strategy": SelectionStrategy.ADAPTIVE.value,
            "constraints": {"max_latency": 5000, "min_accuracy": 0.9}
        }
        
        model_response = await adaptive_model.process_message(model_selection)
        assert model_response["status"] == "success"
        
        # Step 3: Perform RAG operation with selected model
        rag_operation = {
            "query": "What are the key differences between supervised and unsupervised learning?",
            "retrieval_method": RetrievalMethod.SEMANTIC_SEARCH.value,
            "generation_strategy": GenerationStrategy.ADAPTIVE_GENERATION.value,
            "max_contexts": 5
        }
        
        rag_response = await rag.process_message(rag_operation)
        assert rag_response["status"] == "success"
        
        # Step 4: Monitor performance and get recommendations
        recommendations_response = await self_monitoring.process_message({"type": "recommendations"})
        assert recommendations_response["status"] == "success"
        
        # Verify all components worked together
        assert all([
            status_response["status"] == "success",
            model_response["status"] == "success",
            rag_response["status"] == "success",
            recommendations_response["status"] == "success"
        ])


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 