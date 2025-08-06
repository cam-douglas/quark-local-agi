#!/usr/bin/env python3
from core.router import Router
from core.use_cases_tasks import PILLARS
from agents.nlu_agent import NLUAgent
from agents.retrieval_agent import RetrievalAgent
from agents.reasoning_agent import ReasoningAgent
from agents.planning_agent import PlanningAgent
from agents.memory_agent import MemoryAgent
from agents.metrics_agent import MetricsAgent
from agents.self_improvement_agent import SelfImprovementAgent
from core.context_window_manager import ContextWindowManager
from core.memory_eviction import MemoryEvictionManager
from core.capability_bootstrapping import CapabilityBootstrapping
from core.safety_enforcement import get_safety_enforcement, safe_action, safe_response
from core.immutable_safety_rules import SecurityError
from core.streaming_manager import get_streaming_manager
from core.cloud_integration import get_cloud_integration
from core.web_browser import get_web_browser

# Define for each top‐level category the sequence of agents to invoke
PIPELINES = {
    "Natural Language Understanding": ["NLU"],
    "Knowledge Retrieval":         ["Retrieval"],
    "Reasoning":                   ["Retrieval", "Reasoning"],
    "Planning":                    ["Retrieval", "Reasoning", "Planning"],
    "Memory & Context":            ["Memory", "Retrieval", "Reasoning"],
    "Metrics & Evaluation":        ["Metrics", "Retrieval", "Reasoning"],
    # Fallback pipelines for other pillars
    "Self-Improvement":            ["SelfImprovement", "Retrieval", "Reasoning"],
    "Streaming & Real-Time":       ["Retrieval", "Reasoning"],
    "Testing & Quality":           ["Retrieval", "Reasoning"],
    "Deployment & Scaling":        ["Retrieval", "Reasoning"],
    "Async & Parallel":            ["Retrieval", "Reasoning"],
    "Front-end & UI":              ["Retrieval", "Reasoning"],
    "Safety & Alignment":          ["Retrieval", "Reasoning"],
    "Meta-Learning":               ["Retrieval", "Reasoning"],
    "Knowledge Graphs":            ["Retrieval", "Reasoning"],
    "Generalized Reasoning":       ["Retrieval", "Reasoning"],
    "Social Intelligence":          ["Retrieval", "Reasoning"],
    "Autonomous Goals":            ["Retrieval", "Reasoning"],
    "Governance & Ethics":         ["Retrieval", "Reasoning"],
}

class Orchestrator:
    def __init__(self):
        self.router    = Router()
        self.agents    = {
            "NLU":        NLUAgent(model_name="facebook/bart-large-mnli"),
            "Retrieval":  RetrievalAgent(
                              model_name="sentence-transformers/all-MiniLM-L6-v2"
                          ),
            "Reasoning":  ReasoningAgent(model_name="google/flan-t5-small"),
            "Planning":   PlanningAgent(model_name="google/flan-t5-small"),
            "Memory":     MemoryAgent(),
            "Metrics":    MetricsAgent(),
            "SelfImprovement": SelfImprovementAgent(),
        }
        
        # Initialize memory system
        self.context_manager = ContextWindowManager()
        self.memory_eviction_manager = MemoryEvictionManager(self.agents["Memory"])
        
        # Initialize metrics system
        self.metrics_agent = self.agents["Metrics"]
        
        # Initialize self-improvement system
        self.self_improvement_agent = self.agents["SelfImprovement"]
        self.capability_bootstrapping = CapabilityBootstrapping()
        
        # Initialize safety enforcement
        self.safety_enforcement = get_safety_enforcement()
        
        # Initialize streaming manager
        self.streaming_manager = get_streaming_manager()
        
        # Initialize cloud integration
        self.cloud_integration = get_cloud_integration()
        
        # Initialize web browser
        self.web_browser = get_web_browser()
        
        # Start a new session
        self.context_manager.start_session()
        
        # preload every model
        for agent in self.agents.values():
            agent._ensure_model()

    def handle(self, prompt: str):
        # Step 1: Safety validation of user input
        try:
            safety_result = self.safety_enforcement.validate_action("process_user_input", {
                "input": prompt,
                "operation": "orchestrator_handle"
            })
            
            if not safety_result["safe"]:
                return {
                    "error": f"Input blocked for safety reasons: {safety_result['reason']}",
                    "blocked": True
                }
        except SecurityError as e:
            return {
                "error": f"Safety system error: {str(e)}",
                "blocked": True
            }
        
        # Step 2: Start metrics tracking
        operation_id = self.metrics_agent.start_operation("orchestrator_handle", prompt)
        
        # Step 3: Add user input to context
        self.context_manager.add_message("user", prompt)
        
        # Step 4: figure out what the user really wants
        category = self.router.route(prompt)

        # Step 4: pick the pipeline for that category
        pipeline = PIPELINES.get(category, [])

        if not pipeline:
            error_msg = f"No pipeline defined for '{category}'"
            self.metrics_agent.end_operation(operation_id, success=False, error_message=error_msg)
            return {"error": error_msg}

        # We'll carry along a "current_input" that agents read from
        current_input = prompt
        results = {}

        for name in pipeline:
            agent = self.agents[name]

            # Special handling for Memory agent
            if name == "Memory":
                # Get relevant memories and add to context
                memory_result = agent.generate(current_input, operation="retrieve")
                if memory_result.get("memories"):
                    memory_context = "\n".join([m["content"] for m in memory_result["memories"]])
                    current_input = f"{prompt}\n\nRelevant memories:\n{memory_context}"
                results[name] = memory_result
                continue
                
            # Special handling for SelfImprovement agent
            elif name == "SelfImprovement":
                # Run self-reflection and capability analysis
                self_improvement_result = agent.generate(current_input, operation="self_reflection")
                
                # Identify learning opportunities
                user_interactions = [{'category': category, 'input': prompt}]
                learning_opportunities = self.capability_bootstrapping.identify_learning_opportunities(user_interactions)
                
                self_improvement_result['learning_opportunities'] = learning_opportunities
                results[name] = self_improvement_result
                continue

            # Some agents (like Retrieval) might return e.g. embeddings or docs;
            # we capture their raw output and then build the next prompt.
            output = agent.generate(current_input)

            results[name] = output

            # If this was a retrieval step, let's prepend the docs into the next input:
            if name == "Retrieval":
                # assume output is a list of strings or a single string
                docs = (
                    "\n".join(output)
                    if isinstance(output, (list, tuple))
                    else str(output)
                )
                current_input = f"{prompt}\n\nRelevant knowledge:\n{docs}"

            # If it was reasoning, we want its text answer to carry forward
            elif name == "Reasoning":
                # reasoning_agent returns a string
                current_input = output
                
                # Store the reasoning result in memory
                if isinstance(output, str) and output.strip():
                    self.agents["Memory"].generate(
                        output, 
                        operation="store",
                        content=output,
                        memory_type="reasoning"
                    )

            # Planning will dispatch sub‐tasks itself, so we don't re‐feed its output
            # back into our pipeline.

        # Step 4: Store the interaction in memory
        if "Reasoning" in results and isinstance(results["Reasoning"], str):
            self.agents["Memory"].generate(
                prompt,
                operation="store",
                content=f"User: {prompt}\nAssistant: {results['Reasoning']}",
                memory_type="conversation"
            )

        # Step 5: Add assistant response to context
        if "Reasoning" in results and isinstance(results["Reasoning"], str):
            self.context_manager.add_message("assistant", results["Reasoning"])

        # Step 6: Safety validation of final response
        reasoning_output = results.get("Reasoning", "")
        if reasoning_output:
            try:
                response_validation = self.safety_enforcement.validate_response(
                    reasoning_output,
                    ["text_generation", "reasoning", "conversation"]
                )
                
                if not response_validation["valid"]:
                    reasoning_output = f"I apologize, but I cannot provide that response as it may not be truthful or safe. {response_validation['reason']}"
                    results["Reasoning"] = reasoning_output
            except Exception as e:
                reasoning_output = f"I apologize, but there was an error validating my response for safety. Please try rephrasing your question."
                results["Reasoning"] = reasoning_output
        
        # Step 7: End metrics tracking
        self.metrics_agent.end_operation(
            operation_id, 
            success=True,
            output_data=reasoning_output,
            tokens_used=len(reasoning_output.split()) if reasoning_output else 0
        )

        return {
            "category": category,
            "results": results,
            "context_stats": self.context_manager.get_context_stats(),
            "metrics": self.metrics_agent.get_performance_summary(),
            "safety_validated": True
        }

