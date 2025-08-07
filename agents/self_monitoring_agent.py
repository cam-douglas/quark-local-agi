"""
Self-Monitoring & Evaluation Agent
Pillar 21: Continuous feedback loops, performance measurement, drift detection, and automated testing
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque

from .base import Agent as BaseAgent
from core.metrics import MetricsCollector
from core.safety_enforcement import SafetyEnforcement


class PerformanceMetric(Enum):
    """Performance metrics for monitoring"""
    RESPONSE_TIME = "response_time"
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    SAFETY_SCORE = "safety_score"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    ERROR_RATE = "error_rate"
    USER_SATISFACTION = "user_satisfaction"


class DriftType(Enum):
    """Types of performance drift"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ACCURACY_DECLINE = "accuracy_decline"
    SAFETY_VIOLATION = "safety_violation"
    MEMORY_LEAK = "memory_leak"
    BEHAVIOR_CHANGE = "behavior_change"


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time"""
    timestamp: datetime
    metrics: Dict[str, float]
    agent_states: Dict[str, Dict[str, Any]]
    system_health: Dict[str, Any]
    drift_indicators: List[Dict[str, Any]]


@dataclass
class DriftAlert:
    """Alert for detected performance drift"""
    drift_type: DriftType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    timestamp: datetime
    metrics_affected: List[str]
    recommended_actions: List[str]


class SelfMonitoringAgent(BaseAgent):
    """
    Self-Monitoring & Evaluation Agent
    
    Implements continuous feedback loops, performance measurement, 
    drift detection, and automated testing for the quark system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("self_monitoring")
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.baseline_metrics = {}
        self.drift_thresholds = {
            "response_time": 0.2,  # 20% increase
            "accuracy": -0.05,     # 5% decrease
            "safety_score": -0.1,  # 10% decrease
            "error_rate": 0.1,     # 10% increase
        }
        
        # Drift detection
        self.drift_alerts = []
        self.drift_window = timedelta(hours=1)
        self.alert_cooldown = timedelta(minutes=30)
        
        # Testing framework
        self.test_suites = {}
        self.benchmark_results = {}
        self.automated_tests = []
        
        # Metrics collection
        self.metrics_collector = MetricsCollector()
        self.safety_enforcement = SafetyEnforcement()
        
        # Monitoring intervals
        self.monitoring_interval = 60  # seconds
        self.evaluation_interval = 300  # 5 minutes
        self.testing_interval = 3600   # 1 hour
        
        # Performance tracking
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Initialize monitoring
        self._initialize_monitoring()
    
    @property
    def name(self) -> str:
        """Get the agent name"""
        return self.model_name
    
    def load_model(self):
        """Load monitoring models and components"""
        try:
            # Initialize monitoring systems
            self._initialize_monitoring()
            return True
        except Exception as e:
            self.logger.error(f"Error loading monitoring models: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate monitoring response or perform monitoring operation.
        
        Args:
            prompt: Monitoring request or operation
            **kwargs: Additional parameters
            
        Returns:
            Monitoring operation result
        """
        try:
            # Parse the prompt to determine operation
            if "status" in prompt.lower():
                return self._get_system_status()
            elif "metrics" in prompt.lower():
                return self._get_current_metrics()
            elif "evaluation" in prompt.lower():
                return asyncio.run(self._run_comprehensive_evaluation())
            elif "test" in prompt.lower():
                return asyncio.run(self._run_all_test_suites())
            else:
                return {"error": f"Unknown monitoring operation: {prompt}"}
                
        except Exception as e:
            return {"error": f"Monitoring operation failed: {str(e)}"}
    
    def _initialize_monitoring(self):
        """Initialize monitoring systems"""
        self.logger.info("Initializing self-monitoring systems...")
        
        # Set up baseline metrics
        self._establish_baseline()
        
        # Initialize test suites
        self._setup_test_suites()
        
        # Start monitoring tasks if event loop is running
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._continuous_monitoring())
            asyncio.create_task(self._periodic_evaluation())
            asyncio.create_task(self._automated_testing())
        except RuntimeError:
            # No running event loop, skip async task creation
            self.logger.info("No running event loop, skipping monitoring tasks")
    
    def _establish_baseline(self):
        """Establish baseline performance metrics"""
        self.logger.info("Establishing performance baseline...")
        
        # Collect initial metrics
        initial_metrics = self._collect_system_metrics()
        
        # Set baseline for each metric
        for metric_name, value in initial_metrics.items():
            if isinstance(value, (int, float)):
                self.baseline_metrics[metric_name] = value
        
        self.logger.info(f"Baseline established with {len(self.baseline_metrics)} metrics")
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive system metrics"""
        metrics = {}
        
        # Performance metrics
        metrics["response_time"] = self._measure_response_time()
        metrics["memory_usage"] = self._get_memory_usage()
        metrics["cpu_usage"] = self._get_cpu_usage()
        metrics["error_rate"] = self._calculate_error_rate()
        
        # Quality metrics
        metrics["accuracy"] = self._estimate_accuracy()
        metrics["relevance"] = self._estimate_relevance()
        metrics["safety_score"] = self._get_safety_score()
        
        # System health
        metrics["uptime"] = time.time() - self.start_time
        metrics["total_requests"] = self.total_requests
        metrics["success_rate"] = self._calculate_success_rate()
        
        return metrics
    
    def _measure_response_time(self) -> float:
        """Measure current response time"""
        # Simulate response time measurement
        return np.random.normal(0.5, 0.1)  # 500ms ± 100ms
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        import psutil
        return psutil.virtual_memory().percent
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        import psutil
        return psutil.cpu_percent()
    
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    def _estimate_accuracy(self) -> float:
        """Estimate current accuracy based on recent performance"""
        # This would integrate with actual accuracy measurement
        return np.random.normal(0.85, 0.05)  # 85% ± 5%
    
    def _estimate_relevance(self) -> float:
        """Estimate current relevance score"""
        return np.random.normal(0.90, 0.03)  # 90% ± 3%
    
    def _get_safety_score(self) -> float:
        """Get current safety score"""
        return self.safety_enforcement.get_current_safety_score()
    
    def _calculate_success_rate(self) -> float:
        """Calculate current success rate"""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    async def _continuous_monitoring(self):
        """Continuous monitoring loop"""
        while True:
            try:
                # Collect current metrics
                current_metrics = self._collect_system_metrics()
                
                # Create performance snapshot
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.now(),
                    metrics=current_metrics,
                    agent_states=self._get_agent_states(),
                    system_health=self._get_system_health(),
                    drift_indicators=self._detect_drift_indicators(current_metrics)
                )
                
                # Store snapshot
                self.performance_history.append(snapshot)
                
                # Check for drift
                await self._check_for_drift(current_metrics)
                
                # Update metrics collector
                self.metrics_collector.record_metrics(current_metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    def _get_agent_states(self) -> Dict[str, Dict[str, Any]]:
        """Get current states of all agents"""
        # This would integrate with the actual agent system
        return {
            "orchestrator": {"status": "active", "load": 0.7},
            "memory_agent": {"status": "active", "memory_usage": 0.3},
            "reasoning_agent": {"status": "active", "processing": 0.5},
            "safety_agent": {"status": "active", "safety_score": 0.95}
        }
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        return {
            "overall_health": "good",
            "critical_alerts": 0,
            "warnings": 0,
            "last_maintenance": datetime.now() - timedelta(days=1),
            "system_load": 0.6
        }
    
    def _detect_drift_indicators(self, current_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Detect potential drift indicators"""
        indicators = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.baseline_metrics:
                baseline = self.baseline_metrics[metric_name]
                if baseline != 0:
                    change_percentage = (current_value - baseline) / baseline
                else:
                    change_percentage = 0.0
                
                # Check if change exceeds threshold
                if metric_name in self.drift_thresholds:
                    threshold = self.drift_thresholds[metric_name]
                    if abs(change_percentage) > abs(threshold):
                        indicators.append({
                            "metric": metric_name,
                            "baseline": baseline,
                            "current": current_value,
                            "change_percentage": change_percentage,
                            "threshold": threshold,
                            "severity": "high" if abs(change_percentage) > abs(threshold) * 2 else "medium"
                        })
        
        return indicators
    
    async def _check_for_drift(self, current_metrics: Dict[str, float]):
        """Check for performance drift and generate alerts"""
        drift_indicators = self._detect_drift_indicators(current_metrics)
        
        for indicator in drift_indicators:
            # Check if we should generate an alert
            if self._should_generate_alert(indicator):
                alert = self._create_drift_alert(indicator)
                self.drift_alerts.append(alert)
                
                # Log the alert
                self.logger.warning(f"Drift detected: {alert.description}")
                
                # Take automatic actions if needed
                await self._handle_drift_alert(alert)
    
    def _should_generate_alert(self, indicator: Dict[str, Any]) -> bool:
        """Determine if an alert should be generated"""
        # Check if we're in cooldown period
        if self.drift_alerts:
            last_alert_time = self.drift_alerts[-1].timestamp
            if datetime.now() - last_alert_time < self.alert_cooldown:
                return False
        
        # Check severity
        return indicator["severity"] in ["high", "critical"]
    
    def _create_drift_alert(self, indicator: Dict[str, Any]) -> DriftAlert:
        """Create a drift alert from an indicator"""
        drift_type = self._determine_drift_type(indicator["metric"])
        
        return DriftAlert(
            drift_type=drift_type,
            severity=indicator["severity"],
            description=f"Performance drift detected in {indicator['metric']}: "
                       f"{indicator['change_percentage']:.2%} change from baseline",
            timestamp=datetime.now(),
            metrics_affected=[indicator["metric"]],
            recommended_actions=self._get_recommended_actions(drift_type, indicator)
        )
    
    def _determine_drift_type(self, metric_name: str) -> DriftType:
        """Determine the type of drift based on metric"""
        drift_mapping = {
            "response_time": DriftType.PERFORMANCE_DEGRADATION,
            "accuracy": DriftType.ACCURACY_DECLINE,
            "safety_score": DriftType.SAFETY_VIOLATION,
            "memory_usage": DriftType.MEMORY_LEAK,
            "error_rate": DriftType.PERFORMANCE_DEGRADATION
        }
        return drift_mapping.get(metric_name, DriftType.BEHAVIOR_CHANGE)
    
    def _get_recommended_actions(self, drift_type: DriftType, indicator: Dict[str, Any]) -> List[str]:
        """Get recommended actions for a drift type"""
        actions = {
            DriftType.PERFORMANCE_DEGRADATION: [
                "Check system resources",
                "Review recent code changes",
                "Consider model optimization",
                "Monitor for patterns"
            ],
            DriftType.ACCURACY_DECLINE: [
                "Retrain models with recent data",
                "Check for data drift",
                "Review model parameters",
                "Consider ensemble methods"
            ],
            DriftType.SAFETY_VIOLATION: [
                "Review safety rules",
                "Check input validation",
                "Update safety thresholds",
                "Audit recent decisions"
            ],
            DriftType.MEMORY_LEAK: [
                "Check for memory leaks",
                "Review resource management",
                "Consider garbage collection",
                "Monitor memory usage patterns"
            ],
            DriftType.BEHAVIOR_CHANGE: [
                "Review recent changes",
                "Check for configuration issues",
                "Monitor for patterns",
                "Consider rollback if necessary"
            ]
        }
        return actions.get(drift_type, ["Investigate further"])
    
    async def _handle_drift_alert(self, alert: DriftAlert):
        """Handle a drift alert with automatic actions"""
        self.logger.info(f"Handling drift alert: {alert.drift_type.value}")
        
        if alert.severity == "critical":
            # Take immediate action
            await self._take_emergency_action(alert)
        elif alert.severity == "high":
            # Take preventive action
            await self._take_preventive_action(alert)
        else:
            # Log and monitor
            self.logger.info(f"Monitoring drift: {alert.description}")
    
    async def _take_emergency_action(self, alert: DriftAlert):
        """Take emergency action for critical drift"""
        self.logger.warning(f"Taking emergency action for {alert.drift_type.value}")
        
        if alert.drift_type == DriftType.SAFETY_VIOLATION:
            # Activate safety mode
            await self._activate_safety_mode()
        elif alert.drift_type == DriftType.PERFORMANCE_DEGRADATION:
            # Reduce load
            await self._reduce_system_load()
        elif alert.drift_type == DriftType.MEMORY_LEAK:
            # Force garbage collection
            await self._force_cleanup()
    
    async def _take_preventive_action(self, alert: DriftAlert):
        """Take preventive action for high severity drift"""
        self.logger.info(f"Taking preventive action for {alert.drift_type.value}")
        
        # Log the action
        self.logger.info(f"Preventive action taken: {alert.recommended_actions[0]}")
    
    async def _activate_safety_mode(self):
        """Activate enhanced safety mode"""
        self.logger.warning("Activating enhanced safety mode")
        # This would integrate with the safety system
        pass
    
    async def _reduce_system_load(self):
        """Reduce system load"""
        self.logger.info("Reducing system load")
        # This would integrate with the orchestrator
        pass
    
    async def _force_cleanup(self):
        """Force system cleanup"""
        self.logger.info("Forcing system cleanup")
        import gc
        gc.collect()
    
    async def _periodic_evaluation(self):
        """Periodic comprehensive evaluation"""
        while True:
            try:
                await asyncio.sleep(self.evaluation_interval)
                
                # Run comprehensive evaluation
                evaluation_results = await self._run_comprehensive_evaluation()
                
                # Update benchmarks
                self._update_benchmarks(evaluation_results)
                
                # Generate evaluation report
                await self._generate_evaluation_report(evaluation_results)
                
            except Exception as e:
                self.logger.error(f"Error in periodic evaluation: {e}")
    
    async def _run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive system evaluation"""
        self.logger.info("Running comprehensive evaluation...")
        
        evaluation = {
            "timestamp": datetime.now(),
            "performance_metrics": self._collect_system_metrics(),
            "agent_evaluation": await self._evaluate_agents(),
            "system_health": self._evaluate_system_health(),
            "safety_assessment": await self._evaluate_safety(),
            "memory_analysis": self._evaluate_memory_usage(),
            "recommendations": []
        }
        
        # Generate recommendations
        evaluation["recommendations"] = self._generate_recommendations(evaluation)
        
        return evaluation
    
    async def _evaluate_agents(self) -> Dict[str, Any]:
        """Evaluate performance of all agents"""
        agent_evaluation = {}
        
        # This would integrate with the actual agent system
        agents = ["orchestrator", "memory_agent", "reasoning_agent", "safety_agent"]
        
        for agent_name in agents:
            agent_evaluation[agent_name] = {
                "status": "active",
                "performance": np.random.normal(0.8, 0.1),
                "efficiency": np.random.normal(0.85, 0.05),
                "reliability": np.random.normal(0.9, 0.03)
            }
        
        return agent_evaluation
    
    def _evaluate_system_health(self) -> Dict[str, Any]:
        """Evaluate overall system health"""
        return {
            "overall_health": "good",
            "critical_issues": 0,
            "warnings": len(self.drift_alerts),
            "uptime": time.time() - self.start_time,
            "resource_utilization": {
                "cpu": self._get_cpu_usage(),
                "memory": self._get_memory_usage(),
                "disk": 0.3  # Placeholder
            }
        }
    
    async def _evaluate_safety(self) -> Dict[str, Any]:
        """Evaluate safety systems"""
        return {
            "safety_score": self._get_safety_score(),
            "violations_detected": 0,
            "safety_rules_active": 10,
            "last_safety_audit": datetime.now() - timedelta(hours=2)
        }
    
    def _evaluate_memory_usage(self) -> Dict[str, Any]:
        """Evaluate memory usage patterns"""
        return {
            "current_usage": self._get_memory_usage(),
            "peak_usage": 85.0,  # Placeholder
            "memory_leaks_detected": 0,
            "garbage_collection_frequency": 0.1
        }
    
    def _generate_recommendations(self, evaluation: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation"""
        recommendations = []
        
        # Performance recommendations
        if evaluation["performance_metrics"]["response_time"] > 1.0:
            recommendations.append("Consider optimizing response time")
        
        if evaluation["performance_metrics"]["memory_usage"] > 80:
            recommendations.append("Monitor memory usage closely")
        
        if evaluation["performance_metrics"]["error_rate"] > 0.05:
            recommendations.append("Investigate error rate increase")
        
        # Safety recommendations
        if evaluation["safety_assessment"]["safety_score"] < 0.8:
            recommendations.append("Review safety protocols")
        
        return recommendations
    
    def _update_benchmarks(self, evaluation_results: Dict[str, Any]):
        """Update performance benchmarks"""
        self.benchmark_results[datetime.now()] = evaluation_results
    
    async def _generate_evaluation_report(self, evaluation_results: Dict[str, Any]):
        """Generate and store evaluation report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "evaluation": evaluation_results,
            "drift_alerts": [asdict(alert) for alert in self.drift_alerts[-10:]],  # Last 10 alerts
            "performance_trends": self._calculate_performance_trends(),
            "recommendations": evaluation_results["recommendations"]
        }
        
        # Store report
        await self._store_evaluation_report(report)
    
    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        if len(self.performance_history) < 2:
            return {}
        
        trends = {}
        recent_snapshots = list(self.performance_history)[-10:]  # Last 10 snapshots
        
        for metric in ["response_time", "accuracy", "safety_score", "memory_usage"]:
            values = [snapshot.metrics.get(metric, 0) for snapshot in recent_snapshots]
            if values:
                trends[metric] = {
                    "current": values[-1],
                    "average": np.mean(values),
                    "trend": "increasing" if values[-1] > values[0] else "decreasing",
                    "volatility": np.std(values)
                }
        
        return trends
    
    async def _store_evaluation_report(self, report: Dict[str, Any]):
        """Store evaluation report"""
        # This would integrate with the data storage system
        filename = f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(f"data/exports/{filename}", "w") as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Evaluation report stored: {filename}")
        except Exception as e:
            self.logger.error(f"Error storing evaluation report: {e}")
    
    def _setup_test_suites(self):
        """Set up automated test suites"""
        self.test_suites = {
            "performance": self._performance_test_suite(),
            "safety": self._safety_test_suite(),
            "accuracy": self._accuracy_test_suite(),
            "integration": self._integration_test_suite()
        }
    
    def _performance_test_suite(self) -> Dict[str, Any]:
        """Performance test suite"""
        return {
            "response_time_test": {
                "description": "Test response time under load",
                "threshold": 2.0,  # seconds
                "weight": 0.3
            },
            "memory_usage_test": {
                "description": "Test memory usage patterns",
                "threshold": 85.0,  # percentage
                "weight": 0.2
            },
            "throughput_test": {
                "description": "Test system throughput",
                "threshold": 100,  # requests per minute
                "weight": 0.5
            }
        }
    
    def _safety_test_suite(self) -> Dict[str, Any]:
        """Safety test suite"""
        return {
            "content_filter_test": {
                "description": "Test content filtering effectiveness",
                "threshold": 0.95,  # 95% accuracy
                "weight": 0.4
            },
            "ethical_compliance_test": {
                "description": "Test ethical compliance",
                "threshold": 0.9,  # 90% compliance
                "weight": 0.3
            },
            "alignment_test": {
                "description": "Test value alignment",
                "threshold": 0.85,  # 85% alignment
                "weight": 0.3
            }
        }
    
    def _accuracy_test_suite(self) -> Dict[str, Any]:
        """Accuracy test suite"""
        return {
            "qa_accuracy_test": {
                "description": "Test question-answering accuracy",
                "threshold": 0.8,  # 80% accuracy
                "weight": 0.4
            },
            "reasoning_accuracy_test": {
                "description": "Test reasoning accuracy",
                "threshold": 0.75,  # 75% accuracy
                "weight": 0.3
            },
            "memory_accuracy_test": {
                "description": "Test memory retrieval accuracy",
                "threshold": 0.85,  # 85% accuracy
                "weight": 0.3
            }
        }
    
    def _integration_test_suite(self) -> Dict[str, Any]:
        """Integration test suite"""
        return {
            "agent_coordination_test": {
                "description": "Test agent coordination",
                "threshold": 0.9,  # 90% success rate
                "weight": 0.4
            },
            "pipeline_integration_test": {
                "description": "Test pipeline integration",
                "threshold": 0.95,  # 95% success rate
                "weight": 0.3
            },
            "data_flow_test": {
                "description": "Test data flow integrity",
                "threshold": 0.98,  # 98% success rate
                "weight": 0.3
            }
        }
    
    async def _automated_testing(self):
        """Automated testing loop"""
        while True:
            try:
                await asyncio.sleep(self.testing_interval)
                
                # Run all test suites
                test_results = await self._run_all_test_suites()
                
                # Analyze results
                await self._analyze_test_results(test_results)
                
                # Generate test report
                await self._generate_test_report(test_results)
                
            except Exception as e:
                self.logger.error(f"Error in automated testing: {e}")
    
    async def _run_all_test_suites(self) -> Dict[str, Any]:
        """Run all automated test suites"""
        self.logger.info("Running automated test suites...")
        
        results = {}
        for suite_name, test_suite in self.test_suites.items():
            results[suite_name] = await self._run_test_suite(suite_name, test_suite)
        
        return results
    
    async def _run_test_suite(self, suite_name: str, test_suite: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific test suite"""
        results = {
            "suite_name": suite_name,
            "timestamp": datetime.now(),
            "tests": {},
            "overall_score": 0.0,
            "passed_tests": 0,
            "total_tests": len(test_suite)
        }
        
        total_weight = sum(test["weight"] for test in test_suite.values())
        
        for test_name, test_config in test_suite.items():
            test_result = await self._run_single_test(test_name, test_config)
            results["tests"][test_name] = test_result
            
            if test_result["passed"]:
                results["passed_tests"] += 1
                results["overall_score"] += test_config["weight"] / total_weight
        
        return results
    
    async def _run_single_test(self, test_name: str, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test"""
        try:
            # Simulate test execution
            actual_value = await self._execute_test(test_name, test_config)
            threshold = test_config["threshold"]
            
            passed = self._evaluate_test_result(test_name, actual_value, threshold)
            
            return {
                "test_name": test_name,
                "description": test_config["description"],
                "threshold": threshold,
                "actual_value": actual_value,
                "passed": passed,
                "weight": test_config["weight"]
            }
            
        except Exception as e:
            return {
                "test_name": test_name,
                "description": test_config["description"],
                "threshold": test_config["threshold"],
                "actual_value": None,
                "passed": False,
                "error": str(e),
                "weight": test_config["weight"]
            }
    
    async def _execute_test(self, test_name: str, test_config: Dict[str, Any]) -> float:
        """Execute a specific test"""
        # This would integrate with actual test execution
        if "response_time" in test_name:
            return np.random.normal(1.5, 0.3)  # 1.5s ± 0.3s
        elif "memory" in test_name:
            return np.random.normal(70, 10)  # 70% ± 10%
        elif "accuracy" in test_name:
            return np.random.normal(0.85, 0.05)  # 85% ± 5%
        elif "safety" in test_name:
            return np.random.normal(0.95, 0.02)  # 95% ± 2%
        else:
            return np.random.normal(0.9, 0.05)  # 90% ± 5%
    
    def _evaluate_test_result(self, test_name: str, actual_value: float, threshold: float) -> bool:
        """Evaluate if a test result passes the threshold"""
        if actual_value is None:
            return False
        
        # For most tests, lower is better (except accuracy, safety, etc.)
        if any(metric in test_name for metric in ["accuracy", "safety", "compliance", "success"]):
            return actual_value >= threshold
        else:
            return actual_value <= threshold
    
    async def _analyze_test_results(self, test_results: Dict[str, Any]):
        """Analyze test results and take action if needed"""
        overall_score = 0.0
        total_weight = 0.0
        
        for suite_name, suite_results in test_results.items():
            overall_score += suite_results["overall_score"]
            total_weight += 1.0
        
        if total_weight > 0:
            overall_score /= total_weight
        
        # Take action based on overall score
        if overall_score < 0.7:
            self.logger.warning(f"Low test score detected: {overall_score:.2f}")
            await self._handle_low_test_score(overall_score, test_results)
        elif overall_score < 0.85:
            self.logger.info(f"Moderate test score: {overall_score:.2f}")
        else:
            self.logger.info(f"Good test score: {overall_score:.2f}")
    
    async def _handle_low_test_score(self, score: float, test_results: Dict[str, Any]):
        """Handle low test scores"""
        self.logger.warning(f"Handling low test score: {score:.2f}")
        
        # Identify failing tests
        failing_tests = []
        for suite_name, suite_results in test_results.items():
            for test_name, test_result in suite_results["tests"].items():
                if not test_result["passed"]:
                    failing_tests.append(f"{suite_name}.{test_name}")
        
        # Log failing tests
        if failing_tests:
            self.logger.warning(f"Failing tests: {', '.join(failing_tests)}")
        
        # Take corrective action
        await self._take_corrective_action(failing_tests)
    
    async def _take_corrective_action(self, failing_tests: List[str]):
        """Take corrective action for failing tests"""
        self.logger.info(f"Taking corrective action for {len(failing_tests)} failing tests")
        
        # This would integrate with the actual system to take corrective actions
        for test in failing_tests:
            if "performance" in test:
                await self._optimize_performance()
            elif "safety" in test:
                await self._enhance_safety()
            elif "accuracy" in test:
                await self._improve_accuracy()
            elif "integration" in test:
                await self._fix_integration_issues()
    
    async def _optimize_performance(self):
        """Optimize system performance"""
        self.logger.info("Optimizing system performance")
        # This would integrate with the orchestrator
        pass
    
    async def _enhance_safety(self):
        """Enhance safety systems"""
        self.logger.info("Enhancing safety systems")
        # This would integrate with the safety system
        pass
    
    async def _improve_accuracy(self):
        """Improve system accuracy"""
        self.logger.info("Improving system accuracy")
        # This would integrate with the learning system
        pass
    
    async def _fix_integration_issues(self):
        """Fix integration issues"""
        self.logger.info("Fixing integration issues")
        # This would integrate with the system components
        pass
    
    async def _generate_test_report(self, test_results: Dict[str, Any]):
        """Generate test report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "test_results": test_results,
            "summary": self._generate_test_summary(test_results),
            "recommendations": self._generate_test_recommendations(test_results)
        }
        
        # Store report
        await self._store_test_report(report)
    
    def _generate_test_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate test summary"""
        total_tests = 0
        passed_tests = 0
        overall_score = 0.0
        
        for suite_results in test_results.values():
            total_tests += suite_results["total_tests"]
            passed_tests += suite_results["passed_tests"]
            overall_score += suite_results["overall_score"]
        
        if len(test_results) > 0:
            overall_score /= len(test_results)
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0.0,
            "overall_score": overall_score
        }
    
    def _generate_test_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        for suite_name, suite_results in test_results.items():
            if suite_results["overall_score"] < 0.8:
                recommendations.append(f"Improve {suite_name} test suite performance")
            
            for test_name, test_result in suite_results["tests"].items():
                if not test_result["passed"]:
                    recommendations.append(f"Fix failing test: {suite_name}.{test_name}")
        
        return recommendations
    
    async def _store_test_report(self, report: Dict[str, Any]):
        """Store test report"""
        filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(f"data/exports/{filename}", "w") as f:
                json.dump(report, f, indent=2, default=str)
            self.logger.info(f"Test report stored: {filename}")
        except Exception as e:
            self.logger.error(f"Error storing test report: {e}")
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages for self-monitoring"""
        self.total_requests += 1
        
        try:
            # Record request
            start_time = time.time()
            
            # Process the message
            response = await self._process_monitoring_request(message)
            
            # Record successful request
            self.successful_requests += 1
            
            # Record response time
            response_time = time.time() - start_time
            
            return {
                "status": "success",
                "response": response,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            # Record failed request
            self.failed_requests += 1
            self.logger.error(f"Error processing message: {e}")
            
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _process_monitoring_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a monitoring request"""
        request_type = message.get("type", "status")
        
        if request_type == "status":
            return await self._get_system_status()
        elif request_type == "metrics":
            return await self._get_current_metrics()
        elif request_type == "drift_alerts":
            return await self._get_drift_alerts()
        elif request_type == "evaluation":
            return await self._get_latest_evaluation()
        elif request_type == "test_results":
            return await self._get_latest_test_results()
        elif request_type == "recommendations":
            return await self._get_recommendations()
        else:
            raise ValueError(f"Unknown request type: {request_type}")
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "status": "operational",
            "uptime": time.time() - self.start_time,
            "total_requests": self.total_requests,
            "success_rate": self._calculate_success_rate(),
            "active_alerts": len([a for a in self.drift_alerts if a.severity in ["high", "critical"]]),
            "last_evaluation": datetime.now() - timedelta(minutes=5),
            "next_evaluation": datetime.now() + timedelta(minutes=5)
        }
    
    async def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return self._collect_system_metrics()
    
    async def _get_drift_alerts(self) -> Dict[str, Any]:
        """Get recent drift alerts"""
        recent_alerts = [alert for alert in self.drift_alerts 
                        if datetime.now() - alert.timestamp < timedelta(hours=24)]
        
        return {
            "total_alerts": len(self.drift_alerts),
            "recent_alerts": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.severity == "critical"]),
            "alerts": [asdict(alert) for alert in recent_alerts[-10:]]  # Last 10 alerts
        }
    
    async def _get_latest_evaluation(self) -> Dict[str, Any]:
        """Get latest evaluation results"""
        if not self.benchmark_results:
            return {"status": "no_evaluations_yet"}
        
        latest_evaluation = max(self.benchmark_results.keys())
        return {
            "timestamp": latest_evaluation.isoformat(),
            "results": self.benchmark_results[latest_evaluation]
        }
    
    async def _get_latest_test_results(self) -> Dict[str, Any]:
        """Get latest test results"""
        # This would return the most recent test results
        return {
            "status": "test_results_available",
            "last_test_run": datetime.now() - timedelta(hours=1),
            "next_test_run": datetime.now() + timedelta(hours=1)
        }
    
    async def _get_recommendations(self) -> Dict[str, Any]:
        """Get current recommendations"""
        current_metrics = self._collect_system_metrics()
        recommendations = []
        
        # Performance recommendations
        if current_metrics.get("response_time", 0) > 1.0:
            recommendations.append("Consider optimizing response time")
        
        if current_metrics.get("memory_usage", 0) > 80:
            recommendations.append("Monitor memory usage closely")
        
        if current_metrics.get("error_rate", 0) > 0.05:
            recommendations.append("Investigate error rate increase")
        
        # Safety recommendations
        if current_metrics.get("safety_score", 1.0) < 0.8:
            recommendations.append("Review safety protocols")
        
        return {
            "recommendations": recommendations,
            "priority": "medium" if recommendations else "low",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent"""
        return {
            "name": "SelfMonitoringAgent",
            "description": "Continuous self-monitoring and evaluation agent",
            "capabilities": [
                "Performance monitoring",
                "Drift detection",
                "Automated testing",
                "Evaluation reporting",
                "Alert generation"
            ],
            "status": "active",
            "metrics": {
                "total_requests": self.total_requests,
                "success_rate": self._calculate_success_rate(),
                "active_alerts": len(self.drift_alerts),
                "uptime": time.time() - self.start_time
            }
        } 