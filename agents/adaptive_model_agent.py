"""
Adaptive Model Selection Agent
Pillar 23: Dynamic model selection based on task complexity, cost/latency trade-offs, and prompt analysis
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import hashlib

from .base import Agent as BaseAgent


class ModelType(Enum):
    """Types of models available"""
    TINY_QA = "tiny_qa"
    SMALL_GENERAL = "small_general"
    MEDIUM_REASONING = "medium_reasoning"
    LARGE_PLANNER = "large_planner"
    SPECIALIZED_CODER = "specialized_coder"
    MULTIMODAL = "multimodal"


class TaskComplexity(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class SelectionStrategy(Enum):
    """Model selection strategies"""
    PERFORMANCE_OPTIMIZED = "performance_optimized"
    COST_OPTIMIZED = "cost_optimized"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


@dataclass
class ModelSpec:
    """Model specification"""
    model_id: str
    model_type: ModelType
    parameters: int  # Number of parameters in millions
    latency_ms: float  # Average latency in milliseconds
    cost_per_token: float  # Cost per token
    accuracy_score: float  # Accuracy score (0-1)
    capabilities: List[str]  # List of capabilities
    max_context_length: int  # Maximum context length


@dataclass
class TaskAnalysis:
    """Analysis of task requirements"""
    complexity: TaskComplexity
    estimated_tokens: int
    required_capabilities: List[str]
    performance_requirements: Dict[str, Any]
    cost_constraints: Dict[str, Any]
    latency_requirements: Dict[str, Any]


@dataclass
class ModelSelection:
    """Model selection result"""
    selected_model: ModelSpec
    selection_reason: str
    confidence_score: float
    alternatives: List[ModelSpec]
    estimated_cost: float
    estimated_latency: float
    selection_strategy: SelectionStrategy


class AdaptiveModelAgent(BaseAgent):
    """
    Adaptive Model Selection Agent
    
    Dynamically chooses the right model for the task based on
    prompt complexity, cost/latency trade-offs, and performance requirements.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("adaptive_model")
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        
        # Model registry
        self.model_registry = {}
        self.model_performance_history = defaultdict(list)
        
        # Selection strategies
        self.selection_strategies = {
            SelectionStrategy.PERFORMANCE_OPTIMIZED: self._performance_optimized_selection,
            SelectionStrategy.COST_OPTIMIZED: self._cost_optimized_selection,
            SelectionStrategy.BALANCED: self._balanced_selection,
            SelectionStrategy.ADAPTIVE: self._adaptive_selection
        }
        
        # Task analysis
        self.task_analyzers = {
            "complexity": self._analyze_complexity,
            "token_estimation": self._estimate_tokens,
            "capability_requirements": self._analyze_capabilities,
            "performance_requirements": self._analyze_performance_requirements
        }
        
        # Performance tracking
        self.selection_stats = defaultdict(int)
        self.model_performance = defaultdict(list)
        self.cost_tracking = defaultdict(float)
        
        # Configuration
        self.default_strategy = SelectionStrategy.BALANCED
        self.performance_threshold = 0.8
        self.cost_threshold = 0.1  # $0.10 per request
        self.latency_threshold = 5000  # 5 seconds
        
        # Initialize model registry
        self._initialize_model_registry()
        
        # Start performance monitoring if event loop is running
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._monitor_model_performance())
        except RuntimeError:
            # No running event loop, skip async task creation
            self.logger.info("No running event loop, skipping performance monitoring")
    
    def load_model(self):
        """Load adaptive model selection components"""
        try:
            # Initialize model registry
            self._initialize_model_registry()
            return True
        except Exception as e:
            print(f"Error loading adaptive model components: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generate model selection or perform adaptive model operation.
        
        Args:
            prompt: Task description or operation
            **kwargs: Additional parameters
            
        Returns:
            Model selection result
        """
        try:
            # Parse the prompt to determine operation
            if "task" in prompt.lower() or "select" in prompt.lower():
                return asyncio.run(self._select_model(
                    asyncio.run(self._analyze_task(prompt, {})),
                    SelectionStrategy.BALANCED
                ))
            elif "registry" in prompt.lower():
                return asyncio.run(self.get_model_registry())
            elif "stats" in prompt.lower():
                return asyncio.run(self.get_performance_stats())
            else:
                return {"error": f"Unknown adaptive model operation: {prompt}"}
                
        except Exception as e:
            return {"error": f"Adaptive model operation failed: {str(e)}"}
    
    @property
    def name(self) -> str:
        """Get the agent name"""
        return self.model_name
    
    # Class attributes for dataclasses
    TaskAnalysis = TaskAnalysis
    ModelSpec = ModelSpec
    ModelSelection = ModelSelection
    TaskComplexity = TaskComplexity
    SelectionStrategy = SelectionStrategy
    
    def _initialize_model_registry(self):
        """Initialize the model registry with available models"""
        self.logger.info("Initializing model registry...")
        
        # Register available models
        self._register_models()
        
        # Set up performance tracking
        self._setup_performance_tracking()
        
        self.logger.info(f"Model registry initialized with {len(self.model_registry)} models")
    
    def _register_models(self):
        """Register available models"""
        models = [
            ModelSpec(
                model_id="tiny-qa-1b",
                model_type=ModelType.TINY_QA,
                parameters=1,
                latency_ms=50,
                cost_per_token=0.000001,
                accuracy_score=0.75,
                capabilities=["qa", "simple_reasoning"],
                max_context_length=2048
            ),
            ModelSpec(
                model_id="small-general-3b",
                model_type=ModelType.SMALL_GENERAL,
                parameters=3,
                latency_ms=150,
                cost_per_token=0.000003,
                accuracy_score=0.82,
                capabilities=["qa", "general", "simple_reasoning", "text_generation"],
                max_context_length=4096
            ),
            ModelSpec(
                model_id="medium-reasoning-7b",
                model_type=ModelType.MEDIUM_REASONING,
                parameters=7,
                latency_ms=300,
                cost_per_token=0.000007,
                accuracy_score=0.88,
                capabilities=["qa", "reasoning", "planning", "text_generation"],
                max_context_length=8192
            ),
            ModelSpec(
                model_id="large-planner-13b",
                model_type=ModelType.LARGE_PLANNER,
                parameters=13,
                latency_ms=600,
                cost_per_token=0.000015,
                accuracy_score=0.92,
                capabilities=["qa", "reasoning", "planning", "complex_reasoning", "text_generation"],
                max_context_length=16384
            ),
            ModelSpec(
                model_id="specialized-coder-7b",
                model_type=ModelType.SPECIALIZED_CODER,
                parameters=7,
                latency_ms=400,
                cost_per_token=0.000010,
                accuracy_score=0.85,
                capabilities=["coding", "debugging", "code_generation", "code_analysis"],
                max_context_length=8192
            ),
            ModelSpec(
                model_id="multimodal-8b",
                model_type=ModelType.MULTIMODAL,
                parameters=8,
                latency_ms=800,
                cost_per_token=0.000020,
                accuracy_score=0.80,
                capabilities=["image_understanding", "multimodal_qa", "image_generation"],
                max_context_length=4096
            )
        ]
        
        for model in models:
            self.model_registry[model.model_id] = model
    
    def _setup_performance_tracking(self):
        """Set up performance tracking for models"""
        for model_id in self.model_registry:
            self.model_performance[model_id] = []
            self.cost_tracking[model_id] = 0.0
    
    async def _monitor_model_performance(self):
        """Monitor model performance and update statistics"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                
                # Update performance statistics
                await self._update_performance_statistics()
                
                # Clean up old performance data
                self._cleanup_old_performance_data()
                
            except Exception as e:
                self.logger.error(f"Error in performance monitoring: {e}")
    
    async def _update_performance_statistics(self):
        """Update performance statistics for all models"""
        for model_id in self.model_registry:
            # Calculate recent performance metrics
            recent_performance = self._calculate_recent_performance(model_id)
            
            # Update model performance
            if recent_performance:
                self.model_performance[model_id].append({
                    "timestamp": datetime.now(),
                    "performance": recent_performance
                })
    
    def _calculate_recent_performance(self, model_id: str) -> Dict[str, Any]:
        """Calculate recent performance for a model"""
        # This would integrate with actual model performance tracking
        return {
            "accuracy": np.random.normal(0.85, 0.05),
            "latency": np.random.normal(300, 50),
            "throughput": np.random.normal(100, 10),
            "error_rate": np.random.normal(0.02, 0.01)
        }
    
    def _cleanup_old_performance_data(self):
        """Clean up old performance data"""
        cutoff_time = datetime.now() - timedelta(days=7)
        
        for model_id in self.model_performance:
            self.model_performance[model_id] = [
                entry for entry in self.model_performance[model_id]
                if entry["timestamp"] > cutoff_time
            ]
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages for model selection"""
        start_time = time.time()
        
        try:
            # Extract task information
            task_description = message.get("task_description", "")
            strategy = message.get("strategy", self.default_strategy.value)
            constraints = message.get("constraints", {})
            
            # Analyze task
            task_analysis = await self._analyze_task(task_description, constraints)
            
            # Select model
            selection = await self._select_model(task_analysis, SelectionStrategy(strategy))
            
            # Update statistics
            self.selection_stats[strategy] += 1
            
            return {
                "status": "success",
                "selection": asdict(selection),
                "task_analysis": asdict(task_analysis),
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error in model selection: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _analyze_task(self, task_description: str, constraints: Dict[str, Any]) -> TaskAnalysis:
        """Analyze task requirements"""
        self.logger.info(f"Analyzing task: {task_description[:100]}...")
        
        # Analyze complexity
        complexity = await self._analyze_complexity(task_description)
        
        # Estimate tokens
        estimated_tokens = await self._estimate_tokens(task_description)
        
        # Analyze required capabilities
        required_capabilities = await self._analyze_capabilities(task_description)
        
        # Analyze performance requirements
        performance_requirements = await self._analyze_performance_requirements(constraints)
        
        # Analyze cost constraints
        cost_constraints = self._analyze_cost_constraints(constraints)
        
        # Analyze latency requirements
        latency_requirements = self._analyze_latency_requirements(constraints)
        
        return TaskAnalysis(
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            required_capabilities=required_capabilities,
            performance_requirements=performance_requirements,
            cost_constraints=cost_constraints,
            latency_requirements=latency_requirements
        )
    
    async def _analyze_complexity(self, task_description: str) -> TaskComplexity:
        """Analyze task complexity"""
        # Simple heuristics for complexity analysis
        text_length = len(task_description)
        word_count = len(task_description.split())
        
        # Check for complexity indicators
        complexity_indicators = [
            "complex", "advanced", "sophisticated", "detailed", "comprehensive",
            "analysis", "reasoning", "planning", "strategy", "optimization",
            "performance", "implications", "algorithms", "processing", "computational",
            "memory", "scalability", "factors"
        ]
        
        indicator_count = sum(1 for indicator in complexity_indicators 
                            if indicator.lower() in task_description.lower())
        
        # Determine complexity level
        if text_length < 100 and word_count < 20 and indicator_count == 0:
            return TaskComplexity.SIMPLE
        elif text_length < 300 and word_count < 50 and indicator_count <= 1:
            return TaskComplexity.MODERATE
        elif text_length < 1000 and word_count < 200 and indicator_count <= 2:
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.VERY_COMPLEX
    
    async def _estimate_tokens(self, task_description: str) -> int:
        """Estimate number of tokens for the task"""
        # Simple token estimation (rough approximation)
        word_count = len(task_description.split())
        char_count = len(task_description)
        
        # Rough estimation: 1 token ≈ 4 characters or 0.75 words
        token_estimate = max(word_count * 1.33, char_count / 4)
        
        return int(token_estimate)
    
    async def _analyze_capabilities(self, task_description: str) -> List[str]:
        """Analyze required capabilities for the task"""
        capabilities = []
        
        # Check for different capability requirements
        if any(word in task_description.lower() for word in ["code", "program", "debug", "function"]):
            capabilities.append("coding")
        
        if any(word in task_description.lower() for word in ["image", "picture", "visual", "photo"]):
            capabilities.append("image_understanding")
        
        if any(word in task_description.lower() for word in ["reason", "logic", "deduce", "infer"]):
            capabilities.append("reasoning")
        
        if any(word in task_description.lower() for word in ["plan", "strategy", "approach", "method"]):
            capabilities.append("planning")
        
        if any(word in task_description.lower() for word in ["generate", "create", "write", "compose"]):
            capabilities.append("text_generation")
        
        if any(word in task_description.lower() for word in ["question", "answer", "what", "how", "why"]):
            capabilities.append("qa")
        
        # Default capabilities
        if not capabilities:
            capabilities = ["general"]
        
        return capabilities
    
    async def _analyze_performance_requirements(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance requirements from constraints"""
        return {
            "min_accuracy": constraints.get("min_accuracy", 0.7),
            "max_latency": constraints.get("max_latency", 5000),
            "min_throughput": constraints.get("min_throughput", 10),
            "max_error_rate": constraints.get("max_error_rate", 0.05)
        }
    
    def _analyze_cost_constraints(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cost constraints"""
        return {
            "max_cost_per_request": constraints.get("max_cost_per_request", 0.1),
            "budget_available": constraints.get("budget_available", float('inf')),
            "cost_optimization": constraints.get("cost_optimization", False)
        }
    
    def _analyze_latency_requirements(self, constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze latency requirements"""
        return {
            "max_latency_ms": constraints.get("max_latency_ms", 5000),
            "latency_optimization": constraints.get("latency_optimization", False),
            "real_time_required": constraints.get("real_time_required", False)
        }
    
    async def _select_model(self, task_analysis: TaskAnalysis, strategy: SelectionStrategy) -> ModelSelection:
        """Select the best model for the task"""
        self.logger.info(f"Selecting model with strategy: {strategy.value}")
        
        # Get selection function
        selection_func = self.selection_strategies.get(strategy)
        if not selection_func:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        # Perform model selection
        selected_model, reason, confidence = await selection_func(task_analysis)
        
        # Get alternatives
        alternatives = self._get_alternatives(selected_model, task_analysis)
        
        # Calculate estimates
        estimated_cost = self._estimate_cost(selected_model, task_analysis.estimated_tokens)
        estimated_latency = self._estimate_latency(selected_model, task_analysis.estimated_tokens)
        
        return ModelSelection(
            selected_model=selected_model,
            selection_reason=reason,
            confidence_score=confidence,
            alternatives=alternatives,
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency,
            selection_strategy=strategy
        )
    
    async def _performance_optimized_selection(self, task_analysis: TaskAnalysis) -> Tuple[ModelSpec, str, float]:
        """Select model optimized for performance"""
        candidates = []
        
        for model in self.model_registry.values():
            # Check if model meets requirements
            if self._meets_requirements(model, task_analysis):
                # Calculate performance score
                performance_score = self._calculate_performance_score(model, task_analysis)
                candidates.append((model, performance_score))
        
        if not candidates:
            # Fallback to largest model
            largest_model = max(self.model_registry.values(), key=lambda m: m.parameters)
            return largest_model, "Fallback to largest model due to requirements", 0.5
        
        # Select best performing model
        best_model, best_score = max(candidates, key=lambda x: x[1])
        
        return best_model, f"Selected for best performance score: {best_score:.3f}", best_score
    
    async def _cost_optimized_selection(self, task_analysis: TaskAnalysis) -> Tuple[ModelSpec, str, float]:
        """Select model optimized for cost"""
        candidates = []
        
        for model in self.model_registry.values():
            # Check if model meets requirements
            if self._meets_requirements(model, task_analysis):
                # Calculate cost efficiency
                cost_efficiency = self._calculate_cost_efficiency(model, task_analysis)
                candidates.append((model, cost_efficiency))
        
        if not candidates:
            # Fallback to smallest model
            smallest_model = min(self.model_registry.values(), key=lambda m: m.parameters)
            return smallest_model, "Fallback to smallest model due to requirements", 0.5
        
        # Select most cost-efficient model
        best_model, best_efficiency = max(candidates, key=lambda x: x[1])
        
        return best_model, f"Selected for best cost efficiency: {best_efficiency:.3f}", best_efficiency
    
    async def _balanced_selection(self, task_analysis: TaskAnalysis) -> Tuple[ModelSpec, str, float]:
        """Select model with balanced performance and cost"""
        candidates = []
        
        for model in self.model_registry.values():
            # Check if model meets requirements
            if self._meets_requirements(model, task_analysis):
                # Calculate balanced score
                balanced_score = self._calculate_balanced_score(model, task_analysis)
                candidates.append((model, balanced_score))
        
        if not candidates:
            # Fallback to medium model
            medium_models = [m for m in self.model_registry.values() if m.parameters <= 7]
            if medium_models:
                fallback_model = max(medium_models, key=lambda m: m.parameters)
                return fallback_model, "Fallback to medium model due to requirements", 0.5
            else:
                fallback_model = list(self.model_registry.values())[0]
                return fallback_model, "Fallback to available model", 0.3
        
        # Select model with best balanced score
        best_model, best_score = max(candidates, key=lambda x: x[1])
        
        return best_model, f"Selected for best balanced score: {best_score:.3f}", best_score
    
    async def _adaptive_selection(self, task_analysis: TaskAnalysis) -> Tuple[ModelSpec, str, float]:
        """Adaptive model selection based on task characteristics"""
        # Analyze task characteristics
        if task_analysis.complexity in [TaskComplexity.SIMPLE, TaskComplexity.MODERATE]:
            # Use smaller models for simple tasks
            return await self._cost_optimized_selection(task_analysis)
        elif task_analysis.complexity == TaskComplexity.COMPLEX:
            # Use balanced approach for complex tasks
            return await self._balanced_selection(task_analysis)
        else:
            # Use performance-optimized approach for very complex tasks
            return await self._performance_optimized_selection(task_analysis)
    
    def _meets_requirements(self, model: ModelSpec, task_analysis: TaskAnalysis) -> bool:
        """Check if model meets task requirements"""
        # Check capabilities
        for required_capability in task_analysis.required_capabilities:
            if required_capability not in model.capabilities:
                return False
        
        # Check context length
        if task_analysis.estimated_tokens > model.max_context_length:
            return False
        
        # Check performance requirements
        if model.accuracy_score < task_analysis.performance_requirements.get("min_accuracy", 0.7):
            return False
        
        if model.latency_ms > task_analysis.performance_requirements.get("max_latency", 5000):
            return False
        
        return True
    
    def _calculate_performance_score(self, model: ModelSpec, task_analysis: TaskAnalysis) -> float:
        """Calculate performance score for a model"""
        # Base score from model accuracy
        score = model.accuracy_score
        
        # Adjust for latency (lower is better)
        latency_factor = max(0, 1 - (model.latency_ms / 5000))
        score *= 0.7 + 0.3 * latency_factor
        
        # Adjust for recent performance if available
        if model.model_id in self.model_performance and self.model_performance[model.model_id]:
            recent_perf = self.model_performance[model.model_id][-1]["performance"]
            score = (score + recent_perf["accuracy"]) / 2
        
        return score
    
    def _calculate_cost_efficiency(self, model: ModelSpec, task_analysis: TaskAnalysis) -> float:
        """Calculate cost efficiency for a model"""
        # Calculate cost for the task
        estimated_cost = self._estimate_cost(model, task_analysis.estimated_tokens)
        
        # Calculate efficiency (higher accuracy per cost unit)
        efficiency = model.accuracy_score / max(estimated_cost, 0.000001)
        
        return efficiency
    
    def _calculate_balanced_score(self, model: ModelSpec, task_analysis: TaskAnalysis) -> float:
        """Calculate balanced score considering performance and cost"""
        performance_score = self._calculate_performance_score(model, task_analysis)
        cost_efficiency = self._calculate_cost_efficiency(model, task_analysis)
        
        # Normalize cost efficiency
        normalized_cost_efficiency = min(cost_efficiency / 1000, 1.0)  # Normalize to 0-1
        
        # Combine scores (60% performance, 40% cost efficiency)
        balanced_score = 0.6 * performance_score + 0.4 * normalized_cost_efficiency
        
        return balanced_score
    
    def _get_alternatives(self, selected_model: ModelSpec, task_analysis: TaskAnalysis) -> List[ModelSpec]:
        """Get alternative models for the task"""
        alternatives = []
        
        for model in self.model_registry.values():
            if model.model_id != selected_model.model_id and self._meets_requirements(model, task_analysis):
                alternatives.append(model)
        
        # Sort by parameter count (smaller first)
        alternatives.sort(key=lambda m: m.parameters)
        
        return alternatives[:3]  # Return top 3 alternatives
    
    def _estimate_cost(self, model: ModelSpec, estimated_tokens: int) -> float:
        """Estimate cost for using the model"""
        return model.cost_per_token * estimated_tokens
    
    def _estimate_latency(self, model: ModelSpec, estimated_tokens: int) -> float:
        """Estimate latency for using the model"""
        # Base latency plus token-dependent latency
        base_latency = model.latency_ms
        token_latency = estimated_tokens * 0.1  # 0.1ms per token (rough estimate)
        
        return base_latency + token_latency
    
    async def get_model_registry(self) -> Dict[str, Any]:
        """Get information about all available models"""
        return {
            "models": {model_id: asdict(model) for model_id, model in self.model_registry.items()},
            "total_models": len(self.model_registry),
            "model_types": [model_type.value for model_type in ModelType],
            "selection_strategies": [strategy.value for strategy in SelectionStrategy]
        }
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all models"""
        stats = {}
        
        for model_id in self.model_registry:
            if self.model_performance[model_id]:
                recent_perf = self.model_performance[model_id][-1]["performance"]
                stats[model_id] = {
                    "recent_accuracy": recent_perf["accuracy"],
                    "recent_latency": recent_perf["latency"],
                    "recent_throughput": recent_perf["throughput"],
                    "recent_error_rate": recent_perf["error_rate"],
                    "total_cost": self.cost_tracking[model_id]
                }
        
        return {
            "model_performance": stats,
            "selection_stats": dict(self.selection_stats),
            "last_updated": datetime.now().isoformat()
        }
    
    async def add_model(self, model_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new model to the registry"""
        try:
            # Create model spec
            model = ModelSpec(
                model_id=model_spec["model_id"],
                model_type=ModelType(model_spec["model_type"]),
                parameters=model_spec["parameters"],
                latency_ms=model_spec["latency_ms"],
                cost_per_token=model_spec["cost_per_token"],
                accuracy_score=model_spec["accuracy_score"],
                capabilities=model_spec["capabilities"],
                max_context_length=model_spec["max_context_length"]
            )
            
            # Add to registry
            self.model_registry[model.model_id] = model
            
            # Initialize performance tracking
            self.model_performance[model.model_id] = []
            self.cost_tracking[model.model_id] = 0.0
            
            return {
                "status": "success",
                "message": f"Model {model.model_id} added successfully",
                "model": asdict(model)
            }
            
        except Exception as e:
            self.logger.error(f"Error adding model: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def remove_model(self, model_id: str) -> Dict[str, Any]:
        """Remove a model from the registry"""
        try:
            if model_id in self.model_registry:
                del self.model_registry[model_id]
                
                # Clean up performance tracking
                if model_id in self.model_performance:
                    del self.model_performance[model_id]
                
                if model_id in self.cost_tracking:
                    del self.cost_tracking[model_id]
                
                return {
                    "status": "success",
                    "message": f"Model {model_id} removed successfully"
                }
            else:
                return {
                    "status": "error",
                    "error": f"Model {model_id} not found"
                }
                
        except Exception as e:
            self.logger.error(f"Error removing model: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent"""
        return {
            "name": "AdaptiveModelAgent",
            "description": "Adaptive model selection agent with dynamic model choice based on task requirements",
            "capabilities": [
                "Task complexity analysis",
                "Model performance tracking",
                "Cost optimization",
                "Latency optimization",
                "Balanced selection",
                "Adaptive selection"
            ],
            "status": "active",
            "registered_models": len(self.model_registry),
            "selection_strategies": [strategy.value for strategy in SelectionStrategy],
            "model_types": [model_type.value for model_type in ModelType],
            "stats": {
                "total_selections": sum(self.selection_stats.values()),
                "selection_distribution": dict(self.selection_stats)
            }
        } 