#!/usr/bin/env python3
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from core.router import Router
from core.use_cases_tasks import PILLARS
from agents.nlu_agent import NLUAgent
from agents.retrieval_agent import RetrievalAgent
from agents.reasoning_agent import ReasoningAgent
from agents.planning_agent import PlanningAgent
from agents.memory_agent import MemoryAgent
from agents.metrics_agent import MetricsAgent
from agents.self_improvement_agent import SelfImprovementAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from agents.safety_agent import SafetyAgent
from agents.emotional_intelligence_agent import EmotionalIntelligenceAgent
from agents.social_understanding_agent import SocialUnderstandingAgent
from agents.creative_intelligence_agent import CreativeIntelligenceAgent
from agents.autonomous_decision_agent import AutonomousDecisionAgent
from agents.tool_discovery_agent import ToolDiscoveryAgent
from agents.negotiation_agent import NegotiationAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.streaming_agent import StreamingAgent
from agents.continuous_learning_agent import ContinuousLearningAgent
from agents.self_monitoring_agent import SelfMonitoringAgent
from agents.rag_agent import RAGAgent
from agents.adaptive_model_agent import AdaptiveModelAgent
from agents.response_generation_agent import ResponseGenerationAgent
from agents.dataset_discovery_agent import DatasetDiscoveryAgent
from agents.continuous_training_agent import ContinuousTrainingAgent
from agents.code_generation_agent import CodeGenerationAgent
from agents.coding_assistant_agent import CodingAssistantAgent
from core.context_window_manager import ContextWindowManager
from core.memory_eviction import MemoryEvictionManager
from core.capability_bootstrapping import CapabilityBootstrapping
from core.safety_enforcement import get_safety_enforcement, safe_action, safe_response
from core.immutable_safety_rules import SecurityError
from core.streaming_manager import get_streaming_manager
from core.cloud_integration import get_cloud_integration
from core.web_browser import get_web_browser
from meta_learning.meta_learning_orchestrator import MetaLearningOrchestrator

@dataclass
class AgentResult:
    """Result from agent execution."""
    agent_name: str
    success: bool
    output: Any
    execution_time: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class PipelineResult:
    """Result from pipeline execution."""
    category: str
    confidence: float
    pipeline: List[str]
    agent_results: Dict[str, AgentResult]
    final_response: str
    total_execution_time: float
    parallel_execution: bool

# Define for each top‐level category the sequence of agents to invoke
PIPELINES = {
        # Core Pillars (1-10)
        "Natural Language Understanding": ["NLU", "ResponseGeneration"],
        "Knowledge Retrieval":         ["Retrieval", "ResponseGeneration"],
        "Reasoning":                   ["Retrieval", "Reasoning", "ResponseGeneration"],
        "Planning":                    ["Retrieval", "Reasoning", "Planning", "ResponseGeneration"],
        "Memory & Context":            ["Memory", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Metrics & Evaluation":        ["Metrics", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Self-Improvement":            ["SelfImprovement", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Streaming & Real-Time":       ["Streaming", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Testing & Quality":           ["Retrieval", "Reasoning", "ResponseGeneration"],
        "Deployment & Scaling":        ["Retrieval", "Reasoning", "ResponseGeneration"],
        
        # Advanced Intelligence Pillars (11-20)
        "Async & Parallel":            ["Retrieval", "Reasoning", "ResponseGeneration"],
        "Front-end & UI":              ["Retrieval", "Reasoning", "ResponseGeneration"],
        "Safety & Alignment":          ["Safety", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Meta-Learning":               ["MetaLearning", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Knowledge Graphs":            ["KnowledgeGraph", "ResponseGeneration"],
        "Generalized Reasoning":       ["Retrieval", "Reasoning", "ResponseGeneration"],
        "Social Intelligence":          ["SocialUnderstanding", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Autonomous Goals":            ["AutonomousDecision", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Governance & Ethics":         ["Retrieval", "Reasoning", "ResponseGeneration"],
        "RAG Systems":                 ["RAG", "Retrieval", "Reasoning", "ResponseGeneration"],
        
        # Superintelligence Foundation Pillars (21-30)
        "Self-Monitoring":             ["SelfMonitoring", "Retrieval", "Reasoning", "ResponseGeneration"],
        "RAG":                        ["RAG", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Adaptive Model Selection":    ["AdaptiveModel", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Advanced Reasoning":          ["Retrieval", "Reasoning", "ResponseGeneration"],
        "Meta-Cognitive Abilities":    ["MetaLearning", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Self-Improvement Systems":    ["SelfImprovement", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Explainability":              ["Explainability", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Multi-Agent Negotiation":     ["Negotiation", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Tool Discovery":              ["ToolDiscovery", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Autonomous Decision Making":  ["AutonomousDecision", "Retrieval", "Reasoning", "ResponseGeneration"],
        
        # Advanced Intelligence Pillars (31-33)
        "Creative Intelligence":        ["CreativeIntelligence", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Emotional Intelligence":      ["EmotionalIntelligence", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Social Understanding":        ["SocialUnderstanding", "Retrieval", "Reasoning", "ResponseGeneration"],
        
        # Continuous Learning & Monitoring
        "Continuous Learning":         ["ContinuousLearning", "Retrieval", "Reasoning", "ResponseGeneration"],
        "Self-Monitoring":             ["SelfMonitoring", "Retrieval", "Reasoning", "ResponseGeneration"],
        
        # Programming & Development
        "Programming & Code Generation": ["CodingAssistant", "Retrieval", "Reasoning", "ResponseGeneration"],
    }

# Define which agents can run in parallel
PARALLEL_AGENTS = {
    "Retrieval": True,
    "Memory": True,
    "Metrics": True,
    "Safety": True,
    "Streaming": True,
    "RAG": True,
    "SelfMonitoring": True,
    "ContinuousLearning": True,
    "AdaptiveModel": True,
    "CodeGeneration": False,  # Sequential due to code generation complexity
    "CodingAssistant": False,  # Sequential due to natural language processing
    
    # Sequential agents due to dependencies or complexity
    "NLU": False,  # Sequential due to intent classification
    "Reasoning": False,  # Sequential due to dependency on Retrieval
    "Planning": False,  # Sequential due to dependency on Reasoning
    "SelfImprovement": False,  # Sequential due to self-reflection
    "KnowledgeGraph": False,  # Sequential due to complex reasoning
    "EmotionalIntelligence": False,  # Sequential due to emotional analysis
    "SocialUnderstanding": False,  # Sequential due to social analysis
    "CreativeIntelligence": False,  # Sequential due to creative generation
    "AutonomousDecision": False,  # Sequential due to decision making
    "ToolDiscovery": False,  # Sequential due to tool evaluation
    "Negotiation": False,  # Sequential due to multi-agent coordination
    "Explainability": False,  # Sequential due to explanation generation
}

class Orchestrator:
    def __init__(self, max_workers: int = 4):
        self.router    = Router()
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        self.agents    = {
            # Core Pillars (1-10)
            "NLU":        NLUAgent(model_name="facebook/bart-large-mnli"),
            "Retrieval":  RetrievalAgent(
                              model_name="sentence-transformers/all-MiniLM-L6-v2"
                          ),
            "Reasoning":  ReasoningAgent(model_name="google/flan-t5-small"),
            "Planning":   PlanningAgent(model_name="google/flan-t5-small"),
            "Memory":     MemoryAgent(),
            "Metrics":    MetricsAgent(),
            "SelfImprovement": SelfImprovementAgent(),
            "Streaming":  StreamingAgent(),
            "Safety":     SafetyAgent(),
            
            # Advanced Intelligence Pillars (11-20)
            "KnowledgeGraph": KnowledgeGraphAgent(),
            "SocialUnderstanding": SocialUnderstandingAgent(),
            "AutonomousDecision": AutonomousDecisionAgent(),
            "RAG":        RAGAgent(),
            
            # Superintelligence Foundation Pillars (21-30)
            "SelfMonitoring": SelfMonitoringAgent(),
            "AdaptiveModel": AdaptiveModelAgent(),
            "Explainability": ExplainabilityAgent(),
            "Negotiation": NegotiationAgent(),
            "ToolDiscovery": ToolDiscoveryAgent(),
            
            # Advanced Intelligence Pillars (31-33)
            "CreativeIntelligence": CreativeIntelligenceAgent(),
            "EmotionalIntelligence": EmotionalIntelligenceAgent(),
            
            # Continuous Learning & Monitoring
            "ContinuousLearning": ContinuousLearningAgent(),
            
            # Dataset Discovery & Continuous Training
            "DatasetDiscovery": DatasetDiscoveryAgent(),
            "ContinuousTraining": ContinuousTrainingAgent(),
            
            # Code Generation
            "CodeGeneration": CodeGenerationAgent(),
            "CodingAssistant": CodingAssistantAgent(),
            
            # Response Generation
            "ResponseGeneration": ResponseGenerationAgent(),
        }
        
        # Initialize memory system
        self.context_manager = ContextWindowManager()
        self.memory_eviction_manager = MemoryEvictionManager(self.agents["Memory"])
        
        # Initialize metrics system
        self.metrics_agent = self.agents["Metrics"]
        
        # Initialize self-improvement system
        self.self_improvement_agent = self.agents["SelfImprovement"]
        self.capability_bootstrapping = CapabilityBootstrapping()
        
        # Initialize meta-learning system (Pillar 16)
        self.meta_learning_orchestrator = MetaLearningOrchestrator()
        
        # Initialize safety enforcement
        self.safety_enforcement = get_safety_enforcement()
        
        # Initialize streaming manager
        self.streaming_manager = get_streaming_manager()
        
        # Initialize cloud integration
        self.cloud_integration = get_cloud_integration()
        
        # Initialize web browser
        self.web_browser = get_web_browser()
        
        # Performance tracking
        self.execution_stats = {
            "total_requests": 0,
            "parallel_executions": 0,
            "average_execution_time": 0.0,
            "success_rate": 0.0
        }
        
        # Start a new session
        self.context_manager.start_session()
        
        # preload every model
        for agent in self.agents.values():
            agent._ensure_model()

    def _can_run_parallel(self, agent_name: str, position: int, pipeline: List[str]) -> bool:
        """Check if an agent can run in parallel with others."""
        if not PARALLEL_AGENTS.get(agent_name, False):
            return False
            
        # Check dependencies - some agents must run sequentially
        if agent_name == "Reasoning" and "Retrieval" in pipeline[:position]:
            return False
        if agent_name == "Planning" and ("Retrieval" in pipeline[:position] or "Reasoning" in pipeline[:position]):
            return False
            
        return True

    def _execute_agent_parallel(self, agent_name: str, input_data: str, operation: str = "generate") -> AgentResult:
        """Execute an agent in parallel."""
        start_time = time.time()
        agent = self.agents[agent_name]
        
        try:
            if operation == "generate":
                output = agent.generate(input_data)
            elif operation == "retrieve":
                output = agent.generate(input_data, operation="retrieve")
            elif operation == "store":
                output = agent.generate(input_data, operation="store")
            else:
                output = agent.generate(input_data)
                
            execution_time = time.time() - start_time
            
            return AgentResult(
                agent_name=agent_name,
                success=True,
                output=output,
                execution_time=execution_time,
                metadata={"operation": operation}
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                agent_name=agent_name,
                success=False,
                output=None,
                execution_time=execution_time,
                error=str(e),
                metadata={"operation": operation}
            )

    def _execute_pipeline_parallel(self, prompt: str, pipeline: List[str], category: str) -> PipelineResult:
        """Execute pipeline with parallel execution where possible."""
        start_time = time.time()
        agent_results = {}
        current_input = prompt
        
        # Group agents that can run in parallel
        parallel_groups = []
        sequential_agents = []
        
        for i, agent_name in enumerate(pipeline):
            if self._can_run_parallel(agent_name, i, pipeline):
                if not parallel_groups or len(parallel_groups[-1]) >= 3:  # Max 3 parallel agents
                    parallel_groups.append([agent_name])
                else:
                    parallel_groups[-1].append(agent_name)
            else:
                sequential_agents.append(agent_name)
        
        # Execute parallel groups
        for group in parallel_groups:
            futures = {}
            for agent_name in group:
                future = self.thread_pool.submit(
                    self._execute_agent_parallel, 
                    agent_name, 
                    current_input
                )
                futures[future] = agent_name
            
            # Collect results
            for future in as_completed(futures):
                agent_name = futures[future]
                result = future.result()
                agent_results[agent_name] = result
                
                # Update input for next agents
                if result.success and result.output:
                    if isinstance(result.output, dict):
                        if "text" in result.output:
                            current_input = f"{current_input}\n\n{result.output['text']}"
                        elif "response" in result.output:
                            current_input = f"{current_input}\n\n{result.output['response']}"
                        else:
                            current_input = f"{current_input}\n\n{str(result.output)}"
                    elif isinstance(result.output, str):
                        current_input = f"{current_input}\n\n{result.output}"
        
        # Execute sequential agents
        for agent_name in sequential_agents:
            # Special handling for ResponseGeneration agent
            if agent_name == "ResponseGeneration":
                # Pass pipeline results to ResponseGeneration
                pipeline_results_dict = {name: result.output for name, result in agent_results.items() if result.success}
                agent = self.agents[agent_name]
                start_time = time.time()
                
                try:
                    output = agent.generate(current_input, pipeline_results=pipeline_results_dict)
                    execution_time = time.time() - start_time
                    result = AgentResult(
                        agent_name=agent_name,
                        success=True,
                        output=output,
                        execution_time=execution_time,
                        metadata={"operation": "generate"}
                    )
                except Exception as e:
                    execution_time = time.time() - start_time
                    result = AgentResult(
                        agent_name=agent_name,
                        success=False,
                        output=None,
                        execution_time=execution_time,
                        error=str(e),
                        metadata={"operation": "generate"}
                    )
            else:
                result = self._execute_agent_parallel(agent_name, current_input)
            
            agent_results[agent_name] = result
            
            # Update input for next agents
            if result.success and result.output:
                if isinstance(result.output, dict):
                    if "text" in result.output:
                        current_input = f"{current_input}\n\n{result.output['text']}"
                    elif "response" in result.output:
                        current_input = f"{current_input}\n\n{result.output['response']}"
                    else:
                        current_input = f"{current_input}\n\n{str(result.output)}"
                elif isinstance(result.output, str):
                    current_input = f"{current_input}\n\n{result.output}"
        
        total_execution_time = time.time() - start_time
        
        # Generate final response from the last successful agent
        final_response = prompt  # Default to original prompt
        for agent_name in reversed(pipeline):
            if agent_name in agent_results and agent_results[agent_name].success:
                result = agent_results[agent_name]
                if isinstance(result.output, dict):
                    if "conclusion" in result.output:
                        final_response = result.output["conclusion"]
                    elif "text" in result.output:
                        final_response = result.output["text"]
                    elif "response" in result.output:
                        final_response = result.output["response"]
                    else:
                        final_response = str(result.output)
                elif isinstance(result.output, str):
                    final_response = result.output
                break
        
        return PipelineResult(
            category=category,
            confidence=0.8,  # Will be updated by router
            pipeline=pipeline,
            agent_results=agent_results,
            final_response=final_response,
            total_execution_time=total_execution_time,
            parallel_execution=len(parallel_groups) > 0
        )

    def handle(self, prompt: str):
        # Step 1: Safety validation of user input
        try:
            safety_result = self.safety_enforcement.validate_action("process_user_input", {
                "input": prompt,
                "operation": "orchestrator_handle"
            })
            
            if not safety_result["safe"]:
                # Return a PipelineResult with error information
                return PipelineResult(
                    category="Safety Violation",
                    confidence=0.0,
                    pipeline=[],
                    agent_results={},
                    final_response=f"Safety violation: {safety_result['reason']}",
                    total_execution_time=0.0,
                    parallel_execution=False
                )
        except SecurityError as e:
            return PipelineResult(
                category="Security Error",
                confidence=0.0,
                pipeline=[],
                agent_results={},
                final_response=f"Security error: {e}",
                total_execution_time=0.0,
                parallel_execution=False
            )
        except Exception as e:
            return PipelineResult(
                category="Safety Error",
                confidence=0.0,
                pipeline=[],
                agent_results={},
                final_response=f"Safety validation error: {e}",
                total_execution_time=0.0,
                parallel_execution=False
            )

        # Step 2: Intent classification and routing
        try:
            intent_result = self.router.classify_intent(prompt)
            category = intent_result.get("category", "Unknown")
            confidence = intent_result.get("confidence", 0.0)
            
            # Log the intent classification
            self.metrics_agent.record_intent_classification(category, confidence, prompt)
            
        except Exception as e:
            return PipelineResult(
                category="Intent Classification Error",
                confidence=0.0,
                pipeline=[],
                agent_results={},
                final_response=f"Intent classification error: {e}",
                total_execution_time=0.0,
                parallel_execution=False
            )

        # Step 3: Get the appropriate pipeline
        pipeline = PIPELINES.get(category, ["Retrieval", "Reasoning"])
        
        # Step 4: Execute the pipeline with parallel execution
        try:
            pipeline_result = self._execute_pipeline_parallel(prompt, pipeline, category)
            
            # Update execution stats
            self.execution_stats["total_requests"] += 1
            if pipeline_result.parallel_execution:
                self.execution_stats["parallel_executions"] += 1
            
            # Calculate success rate
            successful_agents = sum(1 for result in pipeline_result.agent_results.values() if result.success)
            total_agents = len(pipeline_result.agent_results)
            if total_agents > 0:
                success_rate = successful_agents / total_agents
                self.execution_stats["success_rate"] = (
                    (self.execution_stats["success_rate"] * (self.execution_stats["total_requests"] - 1) + success_rate) 
                    / self.execution_stats["total_requests"]
                )
            
            # Update average execution time
            self.execution_stats["average_execution_time"] = (
                (self.execution_stats["average_execution_time"] * (self.execution_stats["total_requests"] - 1) + pipeline_result.total_execution_time) 
                / self.execution_stats["total_requests"]
            )
            
        except Exception as e:
            return PipelineResult(
                category="Pipeline Execution Error",
                confidence=0.0,
                pipeline=pipeline,
                agent_results={},
                final_response=f"Pipeline execution error: {e}",
                total_execution_time=0.0,
                parallel_execution=False
            )

        # Step 5: Store in memory
        try:
            memory_result = self.agents["Memory"].generate(
                f"User: {prompt}\nSystem: {pipeline_result.final_response}",
                operation="store"
            )
            pipeline_result.agent_results["Memory"] = AgentResult(
                agent_name="Memory",
                success=True,
                output=memory_result,
                execution_time=0.0,
                metadata={"operation": "store"}
            )
        except Exception as e:
            pipeline_result.agent_results["Memory"] = AgentResult(
                agent_name="Memory",
                success=False,
                output=None,
                execution_time=0.0,
                error=f"Memory storage error: {e}",
                metadata={"operation": "store"}
            )

        # Step 6: Record metrics
        try:
            self.metrics_agent.record_request_metrics(category, len(pipeline), pipeline_result.agent_results)
        except Exception as e:
            pipeline_result.agent_results["Metrics"] = AgentResult(
                agent_name="Metrics",
                success=False,
                output=None,
                execution_time=0.0,
                error=f"Metrics recording error: {e}",
                metadata={"operation": "record"}
            )

        # Step 7: Safety validation of final response
        try:
            safety_result = self.safety_enforcement.validate_action("generate_response", {
                "input": prompt,
                "output": pipeline_result.final_response,
                "operation": "orchestrator_response"
            })
            
            if not safety_result["safe"]:
                # Modify the pipeline result to indicate safety violation
                pipeline_result.final_response = f"Safety violation in response: {safety_result['reason']}"
                pipeline_result.category = "Safety Violation"
                return pipeline_result
        except SecurityError as e:
            pipeline_result.final_response = f"Security error in response: {e}"
            pipeline_result.category = "Security Error"
            return pipeline_result
        except Exception as e:
            pipeline_result.final_response = f"Response safety validation error: {e}"
            pipeline_result.category = "Safety Error"
            return pipeline_result

        # Step 8: Return the final result
        return pipeline_result

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get orchestrator performance statistics."""
        return {
            "total_requests": self.execution_stats.get("total_requests", 0),
            "execution_stats": self.execution_stats,
            "parallel_agents": PARALLEL_AGENTS,
            "max_workers": self.max_workers,
            "active_threads": len(self.thread_pool._threads) if hasattr(self.thread_pool, '_threads') else 0
        }

    def shutdown(self):
        """Shutdown the orchestrator and thread pool."""
        self.thread_pool.shutdown(wait=True)

