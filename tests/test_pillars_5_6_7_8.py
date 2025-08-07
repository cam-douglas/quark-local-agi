#!/usr/bin/env python3
"""
Test file for Pillars 5, 6, 7, and 8
Tests the enhanced orchestrator, memory system, metrics, and self-improvement capabilities
"""

import unittest
import time
import json
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
from datetime import datetime

# Import the components we're testing
from core.orchestrator import Orchestrator, AgentResult, PipelineResult, PARALLEL_AGENTS
from agents.memory_agent import MemoryAgent
from agents.metrics_agent import MetricsAgent, SystemMonitor
from agents.self_improvement_agent import SelfImprovementAgent, LearningExample, ModelUpgrade


class TestPillar5Orchestrator(unittest.TestCase):
    """Test Pillar 5: Enhanced Orchestrator & Multi-Agent Framework"""
    
    def setUp(self):
        """Set up test environment."""
        self.orchestrator = Orchestrator(max_workers=2)
        self.test_prompt = "What is artificial intelligence?"
    
    def test_parallel_agent_configuration(self):
        """Test that parallel agent configuration is correct."""
        # Test that parallel agents are properly configured
        self.assertTrue(PARALLEL_AGENTS["Retrieval"])
        self.assertTrue(PARALLEL_AGENTS["Memory"])
        self.assertTrue(PARALLEL_AGENTS["Metrics"])
        self.assertTrue(PARALLEL_AGENTS["Safety"])
        
        # Test that sequential agents are properly configured
        self.assertFalse(PARALLEL_AGENTS["NLU"])
        self.assertFalse(PARALLEL_AGENTS["Reasoning"])
        self.assertFalse(PARALLEL_AGENTS["Planning"])
    
    def test_can_run_parallel(self):
        """Test parallel execution logic."""
        # Test that Retrieval can run in parallel
        self.assertTrue(self.orchestrator._can_run_parallel("Retrieval", 0, ["Retrieval", "Reasoning"]))
        
        # Test that Reasoning cannot run in parallel when Retrieval is in pipeline
        self.assertFalse(self.orchestrator._can_run_parallel("Reasoning", 1, ["Retrieval", "Reasoning"]))
        
        # Test that Planning cannot run in parallel when dependencies exist
        self.assertFalse(self.orchestrator._can_run_parallel("Planning", 2, ["Retrieval", "Reasoning", "Planning"]))
    
    def test_execute_agent_parallel(self):
        """Test parallel agent execution."""
        # Test successful execution
        result = self.orchestrator._execute_agent_parallel("Retrieval", "test input")
        self.assertIsInstance(result, AgentResult)
        self.assertEqual(result.agent_name, "Retrieval")
        self.assertTrue(result.success)
        self.assertGreater(result.execution_time, 0)
    
    def test_execute_pipeline_parallel(self):
        """Test parallel pipeline execution."""
        pipeline = ["Retrieval", "Reasoning"]
        result = self.orchestrator._execute_pipeline_parallel("test prompt", pipeline, "Reasoning")
        
        self.assertIsInstance(result, PipelineResult)
        self.assertEqual(result.category, "Reasoning")
        self.assertEqual(result.pipeline, pipeline)
        self.assertGreater(result.total_execution_time, 0)
        self.assertIn("Retrieval", result.agent_results)
        self.assertIn("Reasoning", result.agent_results)
    
    def test_orchestrator_handle(self):
        """Test the main orchestrator handle method."""
        result = self.orchestrator.handle(self.test_prompt)
        
        # Check that we get a valid response
        self.assertIn("response", result)
        self.assertIn("category", result)
        self.assertIn("pipeline", result)
        self.assertIn("execution_stats", result)
        
        # Check execution stats
        stats = result["execution_stats"]
        self.assertGreaterEqual(stats["total_requests"], 1)
        self.assertGreaterEqual(stats["success_rate"], 0)
    
    def test_performance_stats(self):
        """Test performance statistics."""
        stats = self.orchestrator.get_performance_stats()
        
        self.assertIn("execution_stats", stats)
        self.assertIn("parallel_agents", stats)
        self.assertIn("max_workers", stats)
        self.assertEqual(stats["max_workers"], 2)
    
    def test_shutdown(self):
        """Test orchestrator shutdown."""
        self.orchestrator.shutdown()
        # Should not raise any exceptions


class TestPillar6Memory(unittest.TestCase):
    """Test Pillar 6: Memory & Context Management"""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory_agent = MemoryAgent(memory_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_memory_initialization(self):
        """Test memory system initialization."""
        self.assertTrue(self.memory_agent.load_model())
        self.assertIsNotNone(self.memory_agent.long_term_memory)
    
    def test_store_memory(self):
        """Test memory storage with enhanced features."""
        content = "User likes pizza and prefers Italian restaurants"
        result = self.memory_agent.generate(
            content,
            operation="store_memory",
            memory_type="episodic",
            priority="medium"
        )
        
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")
        self.assertIn("memory_id", result)
        
        # Test duplicate detection
        duplicate_result = self.memory_agent.generate(
            content,
            operation="store_memory",
            memory_type="episodic",
            priority="medium"
        )
        
        # Should return success for both operations
        self.assertEqual(duplicate_result["status"], "success")
    
    def test_retrieve_memories(self):
        """Test memory retrieval with enhanced filtering."""
        # Store some test memories
        self.memory_agent.generate("I love pizza", operation="store_memory", memory_type="episodic")
        self.memory_agent.generate("I prefer Italian food", operation="store_memory", memory_type="episodic")
        self.memory_agent.generate("I work at a tech company", operation="store_memory", memory_type="semantic")
        
        # Test basic retrieval
        result = self.memory_agent.generate("pizza", operation="retrieve_memories", max_results=5)
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")
        self.assertIn("memories", result)
        
        # Test filtering by memory type
        preference_result = self.memory_agent.generate(
            "food", operation="retrieve_memories", memory_types=["episodic"]
        )
        self.assertEqual(preference_result["status"], "success")
    
    def test_context_aware_memories(self):
        """Test context-aware memory retrieval."""
        # Store memories with context
        self.memory_agent.generate("User works at Google", operation="store_memory", memory_type="semantic")
        self.memory_agent.generate("User lives in San Francisco", operation="store_memory", memory_type="semantic")
        self.memory_agent.generate("User likes coffee", operation="store_memory", memory_type="episodic")
        
        # Test retrieval with context
        result = self.memory_agent.generate(
            "Tell me about my preferences", 
            operation="retrieve_memories",
            max_results=10
        )
        
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")
    
    def test_memory_importance_scoring(self):
        """Test memory importance scoring."""
        content = "Important user preference"
        
        # Test storing with high priority
        result = self.memory_agent.generate(
            content,
            operation="store_memory",
            memory_type="episodic",
            priority="high"
        )
        
        self.assertIn("status", result)
        self.assertEqual(result["status"], "success")
    
    def test_memory_stats(self):
        """Test comprehensive memory statistics."""
        # Store some memories
        self.memory_agent.generate("Test memory 1", operation="store_memory", memory_type="episodic")
        self.memory_agent.generate("Test memory 2", operation="store_memory", memory_type="semantic")
        self.memory_agent.generate("Test memory 3", operation="store_memory", memory_type="procedural")
        
        stats = self.memory_agent.generate("", operation="get_memory_stats")
        
        self.assertIn("status", stats)
        self.assertEqual(stats["status"], "success")
        self.assertIn("basic_stats", stats)
        self.assertIn("total_memories", stats["basic_stats"])
    
    def test_memory_operations(self):
        """Test memory operations through generate method."""
        # Test store operation
        store_result = self.memory_agent.generate(
            "Important information",
            operation="store_memory",
            memory_type="episodic"
        )
        
        self.assertIn("status", store_result)
        self.assertEqual(store_result["status"], "success")
        
        # Test retrieve operation
        retrieve_result = self.memory_agent.generate(
            "Important information",
            operation="retrieve_memories"
        )
        
        self.assertIn("status", retrieve_result)
        self.assertEqual(retrieve_result["status"], "success")


class TestPillar7Metrics(unittest.TestCase):
    """Test Pillar 7: Metrics & Evaluation"""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.metrics_agent = MetricsAgent(metrics_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_system_monitor(self):
        """Test system monitoring capabilities."""
        monitor = SystemMonitor()
        
        memory_usage = monitor.get_memory_usage()
        self.assertGreaterEqual(memory_usage, 0)
        
        cpu_usage = monitor.get_cpu_usage()
        self.assertGreaterEqual(cpu_usage, 0)
        self.assertLessEqual(cpu_usage, 100)
        
        gpu_usage = monitor.get_gpu_usage()
        self.assertGreaterEqual(gpu_usage, 0)
    
    def test_operation_tracking(self):
        """Test operation tracking and timing."""
        operation_id = self.metrics_agent.start_operation("test_operation", "test input")
        self.assertNotEqual(operation_id, "disabled")
        
        # Simulate some processing time
        time.sleep(0.1)
        
        self.metrics_agent.end_operation(
            operation_id=operation_id,
            success=True,
            output_data="test output",
            tokens_used=100
        )
        
        # Check that metrics were recorded
        self.assertGreater(self.metrics_agent.current_metrics["total_requests"], 0)
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        # Run some test operations
        for i in range(5):
            op_id = self.metrics_agent.start_operation(f"test_op_{i}", f"input_{i}")
            time.sleep(0.01)
            self.metrics_agent.end_operation(op_id, success=True, output_data=f"output_{i}")
        
        summary = self.metrics_agent.get_performance_summary(time_window=3600)
        
        self.assertIn("total_operations", summary)
        self.assertIn("success_rate", summary)
        self.assertIn("average_latency", summary)
        self.assertIn("total_tokens", summary)
        self.assertEqual(summary["total_operations"], 5)
        self.assertEqual(summary["success_rate"], 1.0)
    
    def test_error_tracking(self):
        """Test error tracking and analysis."""
        # Record some errors
        for i in range(3):
            op_id = self.metrics_agent.start_operation(f"error_op_{i}")
            self.metrics_agent.end_operation(
                op_id, 
                success=False, 
                error_message=f"Test error {i}"
            )
        
        error_summary = self.metrics_agent.get_error_summary(time_window=3600)
        
        self.assertIn("total_errors", error_summary)
        self.assertIn("error_rate", error_summary)
        self.assertIn("common_errors", error_summary)
        self.assertEqual(error_summary["total_errors"], 3)
    
    def test_latency_analysis(self):
        """Test latency analysis and percentiles."""
        # Generate some latency data
        latencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for latency in latencies:
            op_id = self.metrics_agent.start_operation("latency_test")
            time.sleep(latency)
            self.metrics_agent.end_operation(op_id, success=True)
        
        analysis = self.metrics_agent.get_latency_analysis(time_window=3600)
        
        self.assertIn("average_latency", analysis)
        self.assertIn("p95_latency", analysis)
        self.assertIn("p99_latency", analysis)
        self.assertIn("latency_distribution", analysis)
        
        # Check that percentiles are calculated correctly
        self.assertGreater(analysis["p95_latency"], analysis["average_latency"])
        self.assertGreater(analysis["p99_latency"], analysis["p95_latency"])
    
    def test_model_performance_tracking(self):
        """Test model-specific performance tracking."""
        model_name = "test_model"
        
        # Record some model operations
        for i in range(3):
            op_id = self.metrics_agent.start_operation("model_test", model_name=model_name)
            time.sleep(0.1)
            self.metrics_agent.end_operation(
                op_id, 
                success=True, 
                tokens_used=50,
                model_name=model_name
            )
        
        model_perf = self.metrics_agent.get_model_performance(model_name)
        
        self.assertIn("total_requests", model_perf)
        self.assertIn("successful_requests", model_perf)
        self.assertIn("average_latency", model_perf)
        self.assertIn("error_rate", model_perf)
        self.assertEqual(model_perf["total_requests"], 3)
        self.assertEqual(model_perf["successful_requests"], 3)
    
    def test_metrics_export(self):
        """Test metrics export functionality."""
        # Generate some metrics data
        op_id = self.metrics_agent.start_operation("export_test")
        self.metrics_agent.end_operation(op_id, success=True)
        
        export_path = self.metrics_agent.export_metrics("test_export.json")
        
        self.assertIsNotNone(export_path)
        self.assertTrue(os.path.exists(export_path))
        
        # Check that export file contains valid JSON
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        self.assertIn("export_timestamp", export_data)
        self.assertIn("current_metrics", export_data)
        self.assertIn("model_performance", export_data)
    
    def test_metrics_operations(self):
        """Test metrics operations through generate method."""
        # Test summary operation
        summary_result = self.metrics_agent.generate("", operation="summary")
        self.assertEqual(summary_result["operation"], "summary")
        self.assertIn("current_metrics", summary_result)
        
        # Test export operation
        export_result = self.metrics_agent.generate("", operation="export")
        self.assertEqual(export_result["operation"], "export")
        self.assertIn("success", export_result)
        
        # Test clear operation
        clear_result = self.metrics_agent.generate("", operation="clear")
        self.assertEqual(clear_result["operation"], "clear")
        self.assertTrue(clear_result["success"])


class TestPillar8SelfImprovement(unittest.TestCase):
    """Test Pillar 8: Self-Improvement & Learning"""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.self_improvement_agent = SelfImprovementAgent(learning_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_learning_example_creation(self):
        """Test learning example creation and storage."""
        example_id = self.self_improvement_agent.add_learning_example(
            input_text="What is AI?",
            expected_output="AI is artificial intelligence",
            actual_output="AI stands for artificial intelligence",
            feedback_score=0.8,
            category="knowledge"
        )
        
        self.assertIsNotNone(example_id)
        self.assertEqual(len(self.self_improvement_agent.learning_examples), 1)
        
        example = self.self_improvement_agent.learning_examples[0]
        self.assertEqual(example.input_text, "What is AI?")
        self.assertEqual(example.feedback_score, 0.8)
        self.assertEqual(example.category, "knowledge")
    
    def test_feedback_score_calculation(self):
        """Test automatic feedback score calculation."""
        example_id = self.self_improvement_agent.add_learning_example(
            input_text="What is AI?",
            expected_output="AI is artificial intelligence",
            actual_output="AI is artificial intelligence",
            category="knowledge"
        )
        
        self.assertIsNotNone(example_id)
        example = self.self_improvement_agent.learning_examples[0]
        
        # Should calculate high similarity for identical outputs
        self.assertGreater(example.feedback_score, 0.9)
    
    def test_performance_gap_analysis(self):
        """Test performance gap analysis."""
        # Add examples with varying performance
        for i in range(10):
            feedback_score = 0.3 if i < 5 else 0.8  # Mix of poor and good performance
            self.self_improvement_agent.add_learning_example(
                input_text=f"Question {i}",
                expected_output=f"Expected {i}",
                actual_output=f"Actual {i}",
                feedback_score=feedback_score,
                category="test"
            )
        
        analysis = self.self_improvement_agent.analyze_performance_gaps()
        
        self.assertIn("total_examples", analysis)
        self.assertIn("average_feedback", analysis)
        self.assertIn("performance_gaps", analysis)
        self.assertIn("improvement_opportunities", analysis)
        
        self.assertEqual(analysis["total_examples"], 10)
        self.assertGreater(len(analysis["improvement_opportunities"]), 0)
    
    def test_self_reflection(self):
        """Test comprehensive self-reflection."""
        # Add some learning examples
        for i in range(15):
            self.self_improvement_agent.add_learning_example(
                input_text=f"Question {i}",
                expected_output=f"Expected {i}",
                actual_output=f"Actual {i}",
                feedback_score=0.6 + (i * 0.02),  # Gradually improving
                category="test"
            )
        
        reflection = self.self_improvement_agent.run_self_reflection()
        
        self.assertEqual(reflection["status"], "completed")
        self.assertIn("performance_analysis", reflection)
        self.assertIn("feedback_patterns", reflection)
        self.assertIn("recommendations", reflection)
        self.assertIn("insights", reflection)
        self.assertIn("improvement_needed", reflection)
        self.assertIn("next_actions", reflection)
    
    def test_online_learning(self):
        """Test online learning capabilities."""
        new_examples = [
            {
                "input_text": "What is ML?",
                "expected_output": "ML is machine learning",
                "actual_output": "ML stands for machine learning",
                "feedback_score": 0.7,
                "category": "knowledge"
            },
            {
                "input_text": "What is NLP?",
                "expected_output": "NLP is natural language processing",
                "actual_output": "NLP is natural language processing",
                "feedback_score": 0.9,
                "category": "knowledge"
            }
        ]
        
        result = self.self_improvement_agent.run_online_learning(new_examples)
        
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["examples_processed"], 2)
        self.assertIn("total_improvement", result)
        self.assertIn("learning_rate", result)
    
    def test_automated_fine_tuning(self):
        """Test automated fine-tuning process."""
        # Add enough examples for fine-tuning
        for i in range(15):
            self.self_improvement_agent.add_learning_example(
                input_text=f"Question {i}",
                expected_output=f"Expected {i}",
                actual_output=f"Actual {i}",
                feedback_score=0.5,  # Low performance to trigger improvement
                category="test"
            )
        
        result = self.self_improvement_agent.run_automated_fine_tuning()
        
        self.assertEqual(result["status"], "completed")
        self.assertIn("session_id", result)
        self.assertIn("examples_processed", result)
        self.assertIn("performance_gain", result)
        self.assertIn("improvements_made", result)
    
    def test_model_upgrade_queue(self):
        """Test model upgrade queue management."""
        upgrade_id = self.self_improvement_agent.add_model_upgrade(
            model_name="test_model",
            upgrade_type="fine_tune",
            current_performance=0.7,
            target_performance=0.8
        )
        
        self.assertIsNotNone(upgrade_id)
        self.assertEqual(len(self.self_improvement_agent.model_upgrade_queue), 1)
        
        upgrade = self.self_improvement_agent.model_upgrade_queue[0]
        self.assertEqual(upgrade.model_name, "test_model")
        self.assertEqual(upgrade.upgrade_type, "fine_tune")
        self.assertEqual(upgrade.status, "pending")
    
    def test_learning_statistics(self):
        """Test comprehensive learning statistics."""
        # Add some learning examples
        for i in range(5):
            self.self_improvement_agent.add_learning_example(
                input_text=f"Question {i}",
                expected_output=f"Expected {i}",
                actual_output=f"Actual {i}",
                feedback_score=0.7,
                category="test"
            )
        
        stats = self.self_improvement_agent.get_learning_statistics()
        
        self.assertIn("total_examples", stats)
        self.assertIn("average_feedback", stats)
        self.assertIn("recent_average_feedback", stats)
        self.assertIn("category_performance", stats)
        self.assertIn("improvement_sessions", stats)
        self.assertIn("model_upgrades", stats)
        self.assertIn("learning_enabled", stats)
        
        self.assertEqual(stats["total_examples"], 5)
        self.assertTrue(stats["learning_enabled"])
    
    def test_learning_data_export(self):
        """Test learning data export functionality."""
        # Add some learning examples
        self.self_improvement_agent.add_learning_example(
            input_text="Test question",
            expected_output="Expected answer",
            actual_output="Actual answer",
            feedback_score=0.8,
            category="test"
        )
        
        export_path = self.self_improvement_agent.export_learning_data("test_learning.json")
        
        self.assertIsNotNone(export_path)
        self.assertTrue(os.path.exists(export_path))
        
        # Check that export file contains valid JSON
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        self.assertIn("export_timestamp", export_data)
        self.assertIn("learning_examples", export_data)
        self.assertIn("improvement_sessions", export_data)
        self.assertIn("model_upgrades", export_data)
        self.assertIn("statistics", export_data)
    
    def test_self_improvement_operations(self):
        """Test self-improvement operations through generate method."""
        # Test self-reflection operation
        reflection_result = self.self_improvement_agent.generate("", operation="self_reflection")
        self.assertEqual(reflection_result["status"], "completed")
        
        # Test add example operation
        add_result = self.self_improvement_agent.generate(
            "",
            operation="add_example",
            input_text="Test input",
            expected_output="Expected output",
            actual_output="Actual output",
            feedback_score=0.8,
            category="test"
        )
        
        self.assertEqual(add_result["operation"], "add_example")
        self.assertTrue(add_result["success"])
        
        # Test fine-tuning operation
        # First add enough examples
        for i in range(15):
            self.self_improvement_agent.add_learning_example(
                input_text=f"Question {i}",
                expected_output=f"Expected {i}",
                actual_output=f"Actual {i}",
                feedback_score=0.5,
                category="test"
            )
        
        fine_tune_result = self.self_improvement_agent.generate(
            "", operation="fine_tuning"
        )
        
        self.assertEqual(fine_tune_result["status"], "completed")
        
        # Test statistics operation
        stats_result = self.self_improvement_agent.generate("", operation="statistics")
        self.assertEqual(stats_result["operation"], "statistics")
        self.assertIn("statistics", stats_result)
        
        # Test export operation
        export_result = self.self_improvement_agent.generate("", operation="export")
        self.assertEqual(export_result["operation"], "export")
        self.assertIn("success", export_result)


class TestPillarsIntegration(unittest.TestCase):
    """Test integration between all four pillars"""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.orchestrator = Orchestrator(max_workers=2)
        self.memory_agent = MemoryAgent(db_path=os.path.join(self.temp_dir, "memory"))
        self.metrics_agent = MetricsAgent(metrics_dir=os.path.join(self.temp_dir, "metrics"))
        self.self_improvement_agent = SelfImprovementAgent(learning_dir=os.path.join(self.temp_dir, "learning"))
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow using all four pillars."""
        # Step 1: Process a request through the orchestrator (Pillar 5)
        prompt = "What is artificial intelligence and how does it work?"
        result = self.orchestrator.handle(prompt)
        
        # Verify orchestrator response
        self.assertIn("response", result)
        self.assertIn("pipeline", result)
        self.assertIn("execution_stats", result)
        
        # Step 2: Store memory (Pillar 6)
        memory_result = self.memory_agent.store_memory(
            content=f"User asked: {prompt}. Response: {result['response']}",
            memory_type="conversation",
            metadata={"user_id": "test_user", "session_id": "test_session"}
        )
        self.assertIsNotNone(memory_result)
        
        # Step 3: Record metrics (Pillar 7)
        metrics_op_id = self.metrics_agent.start_operation("end_to_end_test", prompt)
        self.metrics_agent.end_operation(
            metrics_op_id,
            success=True,
            output_data=result["response"],
            tokens_used=len(prompt.split()) + len(result["response"].split())
        )
        
        # Step 4: Add learning example (Pillar 8)
        learning_id = self.self_improvement_agent.add_learning_example(
            input_text=prompt,
            expected_output="A comprehensive explanation of AI",
            actual_output=result["response"],
            feedback_score=0.8,
            category="knowledge"
        )
        self.assertIsNotNone(learning_id)
        
        # Step 5: Verify integration
        # Check that memory was stored
        memories = self.memory_agent.retrieve_memories("artificial intelligence")
        self.assertGreater(len(memories), 0)
        
        # Check that metrics were recorded
        summary = self.metrics_agent.get_performance_summary()
        self.assertGreater(summary["total_operations"], 0)
        
        # Check that learning example was added
        stats = self.self_improvement_agent.get_learning_statistics()
        self.assertEqual(stats["total_examples"], 1)
    
    def test_parallel_execution_with_memory_and_metrics(self):
        """Test parallel execution with memory and metrics integration."""
        # Create a test pipeline that uses memory and metrics
        test_prompt = "Tell me about machine learning"
        
        # Process through orchestrator
        result = self.orchestrator.handle(test_prompt)
        
        # Verify that memory and metrics were integrated
        self.assertIn("Memory", result["pipeline_results"].agent_results)
        self.assertIn("Metrics", result["pipeline_results"].agent_results)
        
        # Check that parallel execution was used where possible
        if result["pipeline_results"].parallel_execution:
            self.assertGreater(result["execution_stats"]["parallel_executions"], 0)
    
    def test_self_improvement_with_performance_data(self):
        """Test self-improvement using performance data from metrics."""
        # Generate some performance data
        for i in range(10):
            op_id = self.metrics_agent.start_operation(f"test_op_{i}")
            self.metrics_agent.end_operation(
                op_id,
                success=i < 8,  # 80% success rate
                output_data=f"output_{i}",
                tokens_used=50
            )
        
        # Add learning examples based on performance
        for i in range(10):
            feedback_score = 0.9 if i < 8 else 0.3  # Match success rate
            self.self_improvement_agent.add_learning_example(
                input_text=f"Question {i}",
                expected_output=f"Expected {i}",
                actual_output=f"Actual {i}",
                feedback_score=feedback_score,
                category="performance_test"
            )
        
        # Run self-reflection
        reflection = self.self_improvement_agent.run_self_reflection()
        
        # Verify that performance analysis was conducted
        self.assertEqual(reflection["status"], "completed")
        self.assertIn("performance_analysis", reflection)
        self.assertIn("improvement_needed", reflection)
        
        # Check that improvement opportunities were identified
        performance_analysis = reflection["performance_analysis"]
        self.assertGreater(len(performance_analysis["improvement_opportunities"]), 0)


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2) 