#!/usr/bin/env python3
"""
Evaluation Framework for Meta-Model AI Assistant
Provides comprehensive evaluation capabilities for model performance
"""

import time
import json
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import statistics

@dataclass
class TestCase:
    """A single test case for evaluation."""
    id: str
    input_text: str
    expected_output: str
    category: str
    difficulty: str
    metadata: Dict[str, Any]

@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    test_case_id: str
    actual_output: str
    expected_output: str
    success: bool
    accuracy_score: float
    latency: float
    tokens_used: int
    error_message: Optional[str] = None

class EvaluationFramework:
    def __init__(self, evaluation_dir: str = None):
        self.evaluation_dir = evaluation_dir or os.path.join(os.path.dirname(__file__), '..', 'evaluation')
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
        # Test suites
        self.test_suites = {
            'intent_classification': self._load_intent_classification_tests(),
            'knowledge_retrieval': self._load_knowledge_retrieval_tests(),
            'reasoning': self._load_reasoning_tests(),
            'memory': self._load_memory_tests(),
            'general': self._load_general_tests()
        }
        
        # Evaluation metrics
        self.evaluation_metrics = {
            'accuracy': 0.0,
            'latency': 0.0,
            'throughput': 0.0,
            'error_rate': 0.0,
            'user_satisfaction': 0.0
        }
        
    def _load_intent_classification_tests(self) -> List[TestCase]:
        """Load intent classification test cases."""
        return [
            TestCase(
                id="intent_001",
                input_text="What is artificial intelligence?",
                expected_output="knowledge_retrieval",
                category="intent_classification",
                difficulty="easy",
                metadata={"intent": "knowledge_retrieval"}
            ),
            TestCase(
                id="intent_002", 
                input_text="Can you help me plan my day?",
                expected_output="planning",
                category="intent_classification",
                difficulty="easy",
                metadata={"intent": "planning"}
            ),
            TestCase(
                id="intent_003",
                input_text="I need to understand how neural networks work",
                expected_output="reasoning",
                category="intent_classification", 
                difficulty="medium",
                metadata={"intent": "reasoning"}
            )
        ]
        
    def _load_knowledge_retrieval_tests(self) -> List[TestCase]:
        """Load knowledge retrieval test cases."""
        return [
            TestCase(
                id="retrieval_001",
                input_text="What are the main types of machine learning?",
                expected_output="supervised, unsupervised, and reinforcement learning",
                category="knowledge_retrieval",
                difficulty="medium",
                metadata={"topic": "machine_learning"}
            ),
            TestCase(
                id="retrieval_002",
                input_text="How does a transformer model work?",
                expected_output="attention mechanism and self-attention",
                category="knowledge_retrieval",
                difficulty="hard",
                metadata={"topic": "transformers"}
            )
        ]
        
    def _load_reasoning_tests(self) -> List[TestCase]:
        """Load reasoning test cases."""
        return [
            TestCase(
                id="reasoning_001",
                input_text="If a model has 95% accuracy on training data but 70% on test data, what's happening?",
                expected_output="overfitting",
                category="reasoning",
                difficulty="medium",
                metadata={"reasoning_type": "diagnostic"}
            ),
            TestCase(
                id="reasoning_002",
                input_text="What would be the best approach for a chatbot that needs to handle multiple languages?",
                expected_output="multilingual model or translation pipeline",
                category="reasoning",
                difficulty="hard",
                metadata={"reasoning_type": "strategic"}
            )
        ]
        
    def _load_memory_tests(self) -> List[TestCase]:
        """Load memory test cases."""
        return [
            TestCase(
                id="memory_001",
                input_text="Remember that I prefer Python over Java",
                expected_output="memory_stored",
                category="memory",
                difficulty="easy",
                metadata={"memory_type": "preference"}
            ),
            TestCase(
                id="memory_002",
                input_text="What did we discuss about AI safety earlier?",
                expected_output="retrieved_from_memory",
                category="memory",
                difficulty="medium",
                metadata={"memory_type": "recall"}
            )
        ]
        
    def _load_general_tests(self) -> List[TestCase]:
        """Load general test cases."""
        return [
            TestCase(
                id="general_001",
                input_text="Hello, how are you?",
                expected_output="greeting_response",
                category="general",
                difficulty="easy",
                metadata={"type": "greeting"}
            ),
            TestCase(
                id="general_002",
                input_text="Can you explain quantum computing in simple terms?",
                expected_output="quantum_computing_explanation",
                category="general",
                difficulty="hard",
                metadata={"topic": "quantum_computing"}
            )
        ]
        
    def run_evaluation(self, model_handler, test_suite: str = None, 
                      max_tests: int = None) -> Dict[str, Any]:
        """Run evaluation on the model."""
        if test_suite and test_suite in self.test_suites:
            test_cases = self.test_suites[test_suite]
        else:
            # Run all test suites
            test_cases = []
            for suite in self.test_suites.values():
                test_cases.extend(suite)
        
        if max_tests:
            test_cases = test_cases[:max_tests]
        
        results = []
        total_latency = 0.0
        total_tokens = 0
        successful_tests = 0
        
        for test_case in test_cases:
            try:
                # Start timing
                start_time = time.time()
                
                # Run the test
                actual_output = model_handler(test_case.input_text)
                
                # End timing
                end_time = time.time()
                latency = end_time - start_time
                total_latency += latency
                
                # Calculate accuracy (simple string similarity for now)
                accuracy_score = self._calculate_accuracy(actual_output, test_case.expected_output)
                
                # Determine success
                success = accuracy_score > 0.7  # 70% threshold
                if success:
                    successful_tests += 1
                
                # Create result
                result = EvaluationResult(
                    test_case_id=test_case.id,
                    actual_output=actual_output,
                    expected_output=test_case.expected_output,
                    success=success,
                    accuracy_score=accuracy_score,
                    latency=latency,
                    tokens_used=len(actual_output.split()) if actual_output else 0
                )
                
                results.append(result)
                total_tokens += result.tokens_used
                
            except Exception as e:
                # Handle errors
                result = EvaluationResult(
                    test_case_id=test_case.id,
                    actual_output="",
                    expected_output=test_case.expected_output,
                    success=False,
                    accuracy_score=0.0,
                    latency=0.0,
                    tokens_used=0,
                    error_message=str(e)
                )
                results.append(result)
        
        # Calculate overall metrics
        total_tests = len(results)
        accuracy = successful_tests / total_tests if total_tests > 0 else 0.0
        avg_latency = total_latency / total_tests if total_tests > 0 else 0.0
        error_rate = len([r for r in results if r.error_message]) / total_tests if total_tests > 0 else 0.0
        
        return {
            'evaluation_timestamp': datetime.now().isoformat(),
            'test_suite': test_suite or 'all',
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': total_tests - successful_tests,
            'accuracy': accuracy,
            'average_latency': avg_latency,
            'total_latency': total_latency,
            'error_rate': error_rate,
            'total_tokens_used': total_tokens,
            'average_tokens_per_test': total_tokens / total_tests if total_tests > 0 else 0,
            'results': [asdict(r) for r in results]
        }
        
    def _calculate_accuracy(self, actual: str, expected: str) -> float:
        """Calculate accuracy between actual and expected output."""
        if not actual or not expected:
            return 0.0
        
        # Simple word overlap for now
        actual_words = set(actual.lower().split())
        expected_words = set(expected.lower().split())
        
        if not expected_words:
            return 1.0 if not actual_words else 0.0
        
        intersection = actual_words.intersection(expected_words)
        union = actual_words.union(expected_words)
        
        return len(intersection) / len(union) if union else 0.0
        
    def run_benchmark(self, model_handler, iterations: int = 10) -> Dict[str, Any]:
        """Run performance benchmark."""
        latencies = []
        token_counts = []
        errors = []
        
        # Use a simple test case for benchmarking
        test_input = "What is artificial intelligence?"
        
        for i in range(iterations):
            try:
                start_time = time.time()
                output = model_handler(test_input)
                end_time = time.time()
                
                latency = end_time - start_time
                tokens = len(output.split()) if output else 0
                
                latencies.append(latency)
                token_counts.append(tokens)
                
            except Exception as e:
                errors.append(str(e))
        
        return {
            'benchmark_timestamp': datetime.now().isoformat(),
            'iterations': iterations,
            'average_latency': statistics.mean(latencies) if latencies else 0.0,
            'min_latency': min(latencies) if latencies else 0.0,
            'max_latency': max(latencies) if latencies else 0.0,
            'latency_std': statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            'average_tokens': statistics.mean(token_counts) if token_counts else 0.0,
            'total_errors': len(errors),
            'error_rate': len(errors) / iterations if iterations > 0 else 0.0,
            'throughput': iterations / sum(latencies) if latencies else 0.0
        }
        
    def generate_test_suite(self, category: str, num_tests: int = 10) -> List[TestCase]:
        """Generate a custom test suite."""
        test_cases = []
        
        if category == "intent_classification":
            intents = ["knowledge_retrieval", "reasoning", "planning", "memory"]
            questions = [
                "What is machine learning?",
                "How do neural networks work?",
                "Can you help me plan my project?",
                "Remember that I like Python"
            ]
            
            for i in range(num_tests):
                intent = random.choice(intents)
                question = random.choice(questions)
                
                test_cases.append(TestCase(
                    id=f"generated_intent_{i:03d}",
                    input_text=question,
                    expected_output=intent,
                    category="intent_classification",
                    difficulty="medium",
                    metadata={"generated": True, "intent": intent}
                ))
        
        return test_cases
        
    def export_evaluation_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Export evaluation results to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        filepath = os.path.join(self.evaluation_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        return filepath
        
    def load_evaluation_results(self, filename: str) -> Dict[str, Any]:
        """Load evaluation results from JSON."""
        filepath = os.path.join(self.evaluation_dir, filename)
        
        with open(filepath, 'r') as f:
            return json.load(f)
        
    def compare_evaluations(self, results1: Dict[str, Any], 
                          results2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two evaluation results."""
        return {
            'comparison_timestamp': datetime.now().isoformat(),
            'accuracy_improvement': results2['accuracy'] - results1['accuracy'],
            'latency_improvement': results1['average_latency'] - results2['average_latency'],
            'error_rate_improvement': results1['error_rate'] - results2['error_rate'],
            'baseline': results1,
            'improved': results2
        }
        
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a human-readable evaluation report."""
        report = f"""
# Evaluation Report
Generated: {results['evaluation_timestamp']}
Test Suite: {results['test_suite']}

## Summary
- Total Tests: {results['total_tests']}
- Successful Tests: {results['successful_tests']}
- Failed Tests: {results['failed_tests']}
- Accuracy: {results['accuracy']:.2%}
- Average Latency: {results['average_latency']:.3f}s
- Error Rate: {results['error_rate']:.2%}

## Performance Metrics
- Total Latency: {results['total_latency']:.3f}s
- Total Tokens Used: {results['total_tokens_used']}
- Average Tokens per Test: {results['average_tokens_per_test']:.1f}

## Recommendations
"""
        
        if results['accuracy'] < 0.8:
            report += "- Consider improving model accuracy\n"
        if results['average_latency'] > 5.0:
            report += "- Optimize for lower latency\n"
        if results['error_rate'] > 0.1:
            report += "- Investigate and fix error sources\n"
        
        return report 