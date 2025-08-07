#!/usr/bin/env python3
"""
ASYNC ORCHESTRATOR
==================

Advanced asynchronous orchestrator for parallel multi-agent execution.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

from core.router import Router
from agents.nlu_agent import NLUAgent
from agents.retrieval_agent import RetrievalAgent
from agents.reasoning_agent import ReasoningAgent
from agents.planning_agent import PlanningAgent
from agents.memory_agent import MemoryAgent
from agents.metrics_agent import MetricsAgent
from agents.self_improvement_agent import SelfImprovementAgent
from agents.code_generation_agent import CodeGenerationAgent
from agents.coding_assistant_agent import CodingAssistantAgent
from core.context_window_manager import ContextWindowManager
from core.memory_eviction import MemoryEvictionManager
from core.capability_bootstrapping import CapabilityBootstrapping
from core.safety_enforcement import get_safety_enforcement
from core.immutable_safety_rules import SecurityError
from core.streaming_manager import get_streaming_manager
from core.cloud_integration import get_cloud_integration
from core.web_browser import get_web_browser

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of agent execution."""
    agent_name: str
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class AsyncOrchestrator:
    """Advanced asynchronous orchestrator for parallel multi-agent execution."""
    
    def __init__(self, max_workers: int = 4):
        """Initialize the async orchestrator."""
        self.max_workers = max_workers
        
        # Initialize components
        self.router = Router()
        self.agents = {
            "NLU":        NLUAgent(model_name="facebook/bart-large-mnli"),
            "Retrieval":  RetrievalAgent(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            "Reasoning":  ReasoningAgent(model_name="google/flan-t5-small"),
            "Planning":   PlanningAgent(model_name="google/flan-t5-small"),
            "Memory":     MemoryAgent(),
            "Metrics":    MetricsAgent(),
            "SelfImprovement": SelfImprovementAgent(),
            "CodeGeneration": CodeGenerationAgent(),
            "CodingAssistant": CodingAssistantAgent(),
        }
        
        # Initialize supporting systems
        self.context_manager = ContextWindowManager()
        self.memory_eviction_manager = MemoryEvictionManager(self.agents["Memory"])
        self.metrics_agent = self.agents["Metrics"]
        self.self_improvement_agent = self.agents["SelfImprovement"]
        self.capability_bootstrapping = CapabilityBootstrapping()
        self.safety_enforcement = get_safety_enforcement()
        self.streaming_manager = get_streaming_manager()
        self.cloud_integration = get_cloud_integration()
        self.web_browser = get_web_browser()
        
        # Execution pool
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # Performance tracking
        self.execution_stats = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_execution_time": 0.0
        }
        
        self.performance_metrics = {
            "parallel_execution_count": 0,
            "concurrent_agents": 0,
            "throughput": 0.0
        }
        
        # Start a new session
        self.context_manager.start_session()
        
        # Preload models asynchronously
        asyncio.create_task(self._preload_models())
        
        logger.info(f"AsyncOrchestrator initialized with {max_workers} workers")
    
    async def _preload_models(self):
        """Preload all models asynchronously."""
        logger.info("🔄 Preloading models asynchronously...")
        
        preload_tasks = []
        for agent_name, agent in self.agents.items():
            task = asyncio.create_task(self._preload_agent_model(agent_name, agent))
            preload_tasks.append(task)
        
        await asyncio.gather(*preload_tasks, return_exceptions=True)
        logger.info("✅ All models preloaded")
    
    async def _preload_agent_model(self, agent_name: str, agent: Any):
        """Preload a single agent's model."""
        try:
            await asyncio.to_thread(agent._ensure_model)
            logger.debug(f"✅ Model preloaded for {agent_name}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to preload model for {agent_name}: {e}")
    
    async def handle(self, prompt: str) -> Dict[str, Any]:
        """Handle user input with parallel execution."""
        start_time = time.time()
        
        try:
            # Step 1: Safety validation
            safety_result = await self._validate_safety(prompt)
            if not safety_result["safe"]:
                return {
                    "error": f"Input blocked for safety reasons: {safety_result['reason']}",
                    "blocked": True
                }
            
            # Step 2: Start metrics tracking
            operation_id = self.metrics_agent.start_operation("async_orchestrator_handle", prompt)
            
            # Step 3: Add to context
            self.context_manager.add_message("user", prompt)
            
            # Step 4: Route and create parallel tasks
            category = self.router.route(prompt)
            pipeline = self._get_pipeline_for_category(category)
            
            if not pipeline:
                error_msg = f"No pipeline defined for '{category}'"
                self.metrics_agent.end_operation(operation_id, success=False, error_message=error_msg)
                return {"error": error_msg}
            
            # Step 5: Execute pipeline with parallel processing
            results = await self._execute_pipeline_parallel(prompt, pipeline, category)
            
            # Step 6: Update metrics
            execution_time = time.time() - start_time
            self.metrics_agent.end_operation(operation_id, success=True, execution_time=execution_time)
            
            # Step 7: Update performance metrics
            self._update_performance_metrics(execution_time, len(pipeline))
            
            return {
                "category": category,
                "results": results,
                "execution_time": execution_time,
                "parallel_execution": True,
                "performance_metrics": self.performance_metrics.copy()
            }
            
        except Exception as e:
            logger.error(f"Error in async orchestrator: {e}")
            return {
                "error": f"Orchestration error: {str(e)}",
                "execution_time": time.time() - start_time
            }
    
    async def _validate_safety(self, prompt: str) -> Dict[str, Any]:
        """Validate input safety asynchronously."""
        try:
            return await asyncio.to_thread(
                self.safety_enforcement.validate_action,
                "process_user_input",
                {"input": prompt, "operation": "async_orchestrator_handle"}
            )
        except SecurityError as e:
            return {"safe": False, "reason": f"Safety system error: {str(e)}"}
    
    def _get_pipeline_for_category(self, category: str) -> List[str]:
        """Get pipeline for a category with async optimizations."""
        pipelines = {
            "Natural Language Understanding": ["NLU"],
            "Knowledge Retrieval": ["Retrieval"],
            "Reasoning": ["Retrieval", "Reasoning"],
            "Planning": ["Retrieval", "Reasoning", "Planning"],
            "Memory & Context": ["Memory", "Retrieval", "Reasoning"],
            "Metrics & Evaluation": ["Metrics", "Retrieval", "Reasoning"],
            "Self-Improvement": ["SelfImprovement", "Retrieval", "Reasoning"],
            "Streaming & Real-Time": ["Retrieval", "Reasoning"],
            "Testing & Quality": ["Retrieval", "Reasoning"],
            "Deployment & Scaling": ["Retrieval", "Reasoning"],
            "Async & Parallel": ["Retrieval", "Reasoning"],
            "Front-end & UI": ["Retrieval", "Reasoning"],
            "Safety & Alignment": ["Retrieval", "Reasoning"],
            "Meta-Learning": ["Retrieval", "Reasoning"],
            "Knowledge Graphs": ["Retrieval", "Reasoning"],
            "Generalized Reasoning": ["Retrieval", "Reasoning"],
            "Social Intelligence": ["Retrieval", "Reasoning"],
            "Autonomous Goals": ["Retrieval", "Reasoning"],
            "Governance & Ethics": ["Retrieval", "Reasoning"],
            "Programming & Code Generation": ["CodingAssistant", "Retrieval", "Reasoning"],
        }
        return pipelines.get(category, ["Retrieval", "Reasoning"])
    
    async def _execute_pipeline_parallel(self, prompt: str, pipeline: List[str], category: str) -> Dict[str, Any]:
        """Execute pipeline with parallel processing."""
        # Create tasks for parallel execution
        tasks = []
        current_input = prompt
        results = {}
        
        for i, agent_name in enumerate(pipeline):
            agent = self.agents[agent_name]
            
            # Determine if this agent can run in parallel
            can_run_parallel = self._can_run_parallel(agent_name, i, pipeline)
            
            if can_run_parallel:
                # Create async task for parallel execution
                task = asyncio.create_task(
                    self._execute_agent_async(agent_name, agent, current_input, category)
                )
                tasks.append((agent_name, task))
            else:
                # Execute sequentially if dependencies exist
                if tasks:
                    # Wait for previous tasks to complete
                    for name, task in tasks:
                        result = await task
                        results[name] = result
                        if name == "Memory" and result.get("memories"):
                            memory_context = "\n".join([m["content"] for m in result["memories"]])
                            current_input = f"{prompt}\n\nRelevant memories:\n{memory_context}"
                    
                    tasks = []  # Clear completed tasks
                
                # Execute current agent
                result = await self._execute_agent_async(agent_name, agent, current_input, category)
                results[agent_name] = result
                
                # Update input for next agent
                if agent_name == "Memory" and result.get("memories"):
                    memory_context = "\n".join([m["content"] for m in result["memories"]])
                    current_input = f"{prompt}\n\nRelevant memories:\n{memory_context}"
        
        # Wait for remaining parallel tasks
        if tasks:
            parallel_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            for (agent_name, _), result in zip(tasks, parallel_results):
                if isinstance(result, Exception):
                    results[agent_name] = {"error": str(result)}
                else:
                    results[agent_name] = result
        
        return results
    
    def _can_run_parallel(self, agent_name: str, position: int, pipeline: List[str]) -> bool:
        """Determine if an agent can run in parallel."""
        # Memory agent should run early and can run in parallel
        if agent_name == "Memory":
            return True
        
        # Metrics agent can run in parallel
        if agent_name == "Metrics":
            return True
        
        # SelfImprovement can run in parallel
        if agent_name == "SelfImprovement":
            return True
        
        # Retrieval can run in parallel
        if agent_name == "Retrieval":
            return True
        
        # NLU can run in parallel
        if agent_name == "NLU":
            return True
        
        # Reasoning and Planning should run sequentially after Retrieval
        if agent_name in ["Reasoning", "Planning"]:
            return False
        
        return True
    
    async def _execute_agent_async(self, agent_name: str, agent: Any, input_data: str, category: str) -> Dict[str, Any]:
        """Execute an agent asynchronously."""
        start_time = time.time()
        
        try:
            # Special handling for different agents
            if agent_name == "Memory":
                result = await asyncio.to_thread(
                    agent.generate, input_data, operation="retrieve"
                )
            elif agent_name == "SelfImprovement":
                result = await asyncio.to_thread(
                    agent.generate, input_data, operation="self_reflection"
                )
                
                # Add learning opportunities
                user_interactions = [{'category': category, 'input': input_data}]
                learning_opportunities = await asyncio.to_thread(
                    self.capability_bootstrapping.identify_learning_opportunities,
                    user_interactions
                )
                result['learning_opportunities'] = learning_opportunities
            else:
                # Standard agent execution
                result = await asyncio.to_thread(agent.generate, input_data)
            
            execution_time = time.time() - start_time
            
            # Update execution stats
            self.execution_stats["total_tasks"] += 1
            self.execution_stats["completed_tasks"] += 1
            self.execution_stats["average_execution_time"] = (
                (self.execution_stats["average_execution_time"] * (self.execution_stats["completed_tasks"] - 1) + execution_time) /
                self.execution_stats["completed_tasks"]
            )
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "agent_name": agent_name
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error executing agent {agent_name}: {e}")
            
            self.execution_stats["total_tasks"] += 1
            self.execution_stats["failed_tasks"] += 1
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "agent_name": agent_name
            }
    
    def _update_performance_metrics(self, execution_time: float, pipeline_length: int):
        """Update performance metrics."""
        self.performance_metrics["parallel_execution_count"] += 1
        self.performance_metrics["concurrent_agents"] = max(
            self.performance_metrics["concurrent_agents"],
            pipeline_length
        )
        self.performance_metrics["throughput"] = 1.0 / execution_time if execution_time > 0 else 0.0
    
    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "execution_stats": self.execution_stats.copy(),
            "performance_metrics": self.performance_metrics.copy(),
            "thread_pool_size": self.thread_pool._max_workers
        }
    
    async def shutdown(self):
        """Shutdown the orchestrator gracefully."""
        logger.info("🛑 Shutting down AsyncOrchestrator...")
        self.thread_pool.shutdown(wait=True)
        logger.info("✅ AsyncOrchestrator shutdown complete")


# Global instance for easy access
_async_orchestrator_instance = None


def get_async_orchestrator() -> AsyncOrchestrator:
    """Get the global async orchestrator instance."""
    global _async_orchestrator_instance
    if _async_orchestrator_instance is None:
        _async_orchestrator_instance = AsyncOrchestrator()
    return _async_orchestrator_instance


async def shutdown_async_orchestrator():
    """Shutdown the global async orchestrator instance."""
    global _async_orchestrator_instance
    if _async_orchestrator_instance:
        await _async_orchestrator_instance.shutdown()
        _async_orchestrator_instance = None
