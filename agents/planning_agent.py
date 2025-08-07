#!/usr/bin/env python3
"""
Planning Agent - Pillar 5
Handles task planning, goal decomposition, and execution strategy generation
"""

import os
import sys
import time
import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents.base import Agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlanStatus(Enum):
    """Status of a planning step."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class PlanPriority(Enum):
    """Priority levels for planning steps."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PlanStep:
    """A single step in a plan."""
    id: str
    title: str
    description: str
    status: PlanStatus
    priority: PlanPriority
    estimated_time: float  # in minutes
    dependencies: List[str]
    resources: List[str]
    notes: str
    timestamp: datetime

@dataclass
class Plan:
    """A complete plan with multiple steps."""
    id: str
    title: str
    description: str
    goal: str
    steps: List[PlanStep]
    total_estimated_time: float
    priority: PlanPriority
    status: PlanStatus
    created_at: datetime
    updated_at: datetime

@dataclass
class PlanningResult:
    """Result from planning agent."""
    plan: Plan
    confidence: float
    reasoning: str
    alternatives: List[Plan]
    metadata: Dict[str, Any]
    timestamp: datetime

class PlanningAgent(Agent):
    """Planning Agent for task decomposition and execution strategy generation."""
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        super().__init__("planning")
        self.model_name = model_name
        
        # Planning configuration
        self.max_steps_per_plan = 20
        self.min_confidence_threshold = 0.6
        self.planning_templates = self._load_planning_templates()
        
        # Plan storage
        self.plans = {}
        self.plan_counter = 0
        
        # Load models
        self.load_model()
        
    def load_model(self):
        """Load planning models for task decomposition."""
        logger.info("Loading planning models...")
        
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
            
            # Text generation model for plan creation
            self.planning_model = pipeline(
                "text2text-generation",
                model=self.model_name,
                max_length=512
            )
            
            # Load tokenizer for text processing
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info("✅ Planning models loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Error loading planning models: {e}")
            self.planning_model = None
            self.tokenizer = None
    
    def generate(self, prompt: str, **kwargs) -> PlanningResult:
        """Generate a plan for the given task or goal."""
        try:
            # Ensure models are loaded
            self._ensure_model()
            
            # Parse the planning request
            goal, context = self._parse_planning_request(prompt, kwargs)
            
            # Generate the plan
            plan = self._create_plan(goal, context)
            
            # Calculate confidence
            confidence = self._calculate_plan_confidence(plan)
            
            # Generate reasoning
            reasoning = self._generate_planning_reasoning(plan, context)
            
            # Generate alternative plans
            alternatives = self._generate_alternative_plans(goal, context)
            
            # Create planning result
            result = PlanningResult(
                plan=plan,
                confidence=confidence,
                reasoning=reasoning,
                alternatives=alternatives,
                metadata={"context": context},
                timestamp=datetime.now()
            )
            
            # Store the plan
            self.plans[plan.id] = plan
            
            return result
            
        except Exception as e:
            logger.error(f"Error in plan generation: {e}")
            return self._create_fallback_plan(prompt)
    
    def _parse_planning_request(self, prompt: str, kwargs: Dict) -> Tuple[str, Dict[str, Any]]:
        """Parse the planning request to extract goal and context."""
        # Extract goal from prompt
        goal = prompt.strip()
        
        # Extract context from kwargs
        context = {
            "priority": kwargs.get("priority", PlanPriority.MEDIUM),
            "time_constraint": kwargs.get("time_constraint", None),
            "resource_constraints": kwargs.get("resource_constraints", []),
            "dependencies": kwargs.get("dependencies", []),
            "complexity": kwargs.get("complexity", "medium"),
            "domain": kwargs.get("domain", "general")
        }
        
        return goal, context
    
    def _create_plan(self, goal: str, context: Dict[str, Any]) -> Plan:
        """Create a plan for the given goal."""
        # Generate plan ID
        plan_id = f"plan_{self.plan_counter:04d}"
        self.plan_counter += 1
        
        # Create plan title and description
        title = f"Plan for: {goal[:50]}{'...' if len(goal) > 50 else ''}"
        description = f"Comprehensive plan to achieve: {goal}"
        
        # Generate steps
        steps = self._generate_plan_steps(goal, context)
        
        # Calculate total estimated time
        total_time = sum(step.estimated_time for step in steps)
        
        # Create plan
        plan = Plan(
            id=plan_id,
            title=title,
            description=description,
            goal=goal,
            steps=steps,
            total_estimated_time=total_time,
            priority=context["priority"],
            status=PlanStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        return plan
    
    def _generate_plan_steps(self, goal: str, context: Dict[str, Any]) -> List[PlanStep]:
        """Generate plan steps for the given goal."""
        if not self.planning_model:
            return self._fallback_step_generation(goal, context)
        
        try:
            # Create planning prompt
            planning_prompt = self._create_planning_prompt(goal, context)
            
            # Generate plan using the model
            response = self.planning_model(
                planning_prompt,
                max_length=512,
                num_return_sequences=1
            )[0]["generated_text"]
            
            # Parse the generated response into steps
            steps = self._parse_generated_steps(response, context)
            
            # If model didn't generate enough steps, add fallback steps
            if len(steps) < 3:
                fallback_steps = self._fallback_step_generation(goal, context)
                steps.extend(fallback_steps[len(steps):])
            
            return steps[:self.max_steps_per_plan]
            
        except Exception as e:
            logger.error(f"Error generating plan steps: {e}")
            return self._fallback_step_generation(goal, context)
    
    def _create_planning_prompt(self, goal: str, context: Dict[str, Any]) -> str:
        """Create a prompt for the planning model."""
        template = self.planning_templates.get(context["domain"], self.planning_templates["general"])
        
        prompt = f"""
Goal: {goal}
Priority: {context['priority'].value}
Complexity: {context['complexity']}
Time Constraint: {context.get('time_constraint', 'None')}
Resources: {', '.join(context.get('resource_constraints', []))}

{template}

Please create a step-by-step plan to achieve this goal. Each step should be clear and actionable.
"""
        
        return prompt.strip()
    
    def _parse_generated_steps(self, response: str, context: Dict[str, Any]) -> List[PlanStep]:
        """Parse the generated response into plan steps."""
        steps = []
        lines = response.split('\n')
        
        step_id = 1
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Try to extract step information
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.')):
                # Numbered step
                step_text = line[2:].strip()
            elif line.startswith(('-', '*', '•')):
                # Bullet point step
                step_text = line[1:].strip()
            else:
                # Regular line, treat as step
                step_text = line
            
            if step_text:
                # Create step
                step = PlanStep(
                    id=f"step_{step_id}",
                    title=f"Step {step_id}",
                    description=step_text,
                    status=PlanStatus.PENDING,
                    priority=context["priority"],
                    estimated_time=self._estimate_step_time(step_text, context),
                    dependencies=[],
                    resources=[],
                    notes="",
                    timestamp=datetime.now()
                )
                steps.append(step)
                step_id += 1
        
        return steps
    
    def _estimate_step_time(self, step_description: str, context: Dict[str, Any]) -> float:
        """Estimate the time required for a step."""
        # Simple heuristic-based estimation
        words = step_description.lower().split()
        
        # Time estimation based on keywords
        time_keywords = {
            "research": 30,
            "analyze": 45,
            "create": 60,
            "build": 90,
            "implement": 120,
            "test": 30,
            "review": 20,
            "document": 30,
            "deploy": 60,
            "setup": 45,
            "configure": 30,
            "install": 30,
            "learn": 120,
            "study": 90,
            "practice": 60,
            "write": 45,
            "design": 90,
            "plan": 30,
            "organize": 30,
            "prepare": 30
        }
        
        # Find matching keywords
        estimated_time = 15  # Default 15 minutes
        for keyword, time_minutes in time_keywords.items():
            if keyword in words:
                estimated_time = max(estimated_time, time_minutes)
        
        # Adjust based on complexity
        if context["complexity"] == "high":
            estimated_time *= 1.5
        elif context["complexity"] == "low":
            estimated_time *= 0.7
        
        return estimated_time
    
    def _fallback_step_generation(self, goal: str, context: Dict[str, Any]) -> List[PlanStep]:
        """Generate fallback steps when model is not available."""
        # Generic step template
        generic_steps = [
            "Define the problem and requirements",
            "Research and gather information",
            "Analyze the situation and constraints",
            "Create a detailed plan",
            "Implement the solution",
            "Test and validate the results",
            "Review and refine the approach",
            "Document the process and outcomes"
        ]
        
        steps = []
        for i, step_desc in enumerate(generic_steps, 1):
            step = PlanStep(
                id=f"step_{i}",
                title=f"Step {i}",
                description=step_desc,
                status=PlanStatus.PENDING,
                priority=context["priority"],
                estimated_time=self._estimate_step_time(step_desc, context),
                dependencies=[],
                resources=[],
                notes="",
                timestamp=datetime.now()
            )
            steps.append(step)
        
        return steps
    
    def _calculate_plan_confidence(self, plan: Plan) -> float:
        """Calculate confidence score for the generated plan."""
        if not plan.steps:
            return 0.0
        
        # Base confidence
        confidence = 0.5
        
        # Boost confidence based on number of steps
        if 3 <= len(plan.steps) <= 10:
            confidence += 0.2
        elif len(plan.steps) > 10:
            confidence += 0.1
        
        # Boost confidence based on step quality
        detailed_steps = sum(1 for step in plan.steps if len(step.description) > 20)
        if detailed_steps >= len(plan.steps) * 0.7:
            confidence += 0.2
        
        # Boost confidence if total time is reasonable
        if 30 <= plan.total_estimated_time <= 480:  # 30 minutes to 8 hours
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_planning_reasoning(self, plan: Plan, context: Dict[str, Any]) -> str:
        """Generate reasoning for the planning decisions."""
        reasoning = f"This plan was created to achieve: {plan.goal}\n\n"
        
        reasoning += f"Key considerations:\n"
        reasoning += f"- Priority: {plan.priority.value}\n"
        reasoning += f"- Complexity: {context['complexity']}\n"
        reasoning += f"- Total estimated time: {plan.total_estimated_time:.0f} minutes\n"
        reasoning += f"- Number of steps: {len(plan.steps)}\n\n"
        
        reasoning += "The plan follows a logical progression from analysis to implementation to validation."
        
        if context.get("time_constraint"):
            reasoning += f"\nTime constraint of {context['time_constraint']} was considered in the planning."
        
        if context.get("resource_constraints"):
            reasoning += f"\nResource constraints were factored into the plan."
        
        return reasoning
    
    def _generate_alternative_plans(self, goal: str, context: Dict[str, Any]) -> List[Plan]:
        """Generate alternative plans for the same goal."""
        alternatives = []
        
        # Generate alternative with different priority
        if context["priority"] != PlanPriority.LOW:
            alt_context = context.copy()
            alt_context["priority"] = PlanPriority.LOW
            alt_plan = self._create_plan(goal, alt_context)
            alternatives.append(alt_plan)
        
        # Generate alternative with different complexity
        if context["complexity"] != "low":
            alt_context = context.copy()
            alt_context["complexity"] = "low"
            alt_plan = self._create_plan(goal, alt_context)
            alternatives.append(alt_plan)
        
        return alternatives[:2]  # Limit to 2 alternatives
    
    def _create_fallback_plan(self, prompt: str) -> PlanningResult:
        """Create a fallback plan when planning fails."""
        goal = prompt.strip()
        context = {"priority": PlanPriority.MEDIUM, "complexity": "medium"}
        
        plan = self._create_plan(goal, context)
        
        return PlanningResult(
            plan=plan,
            confidence=0.5,
            reasoning="Fallback plan generated due to model unavailability.",
            alternatives=[],
            metadata={"fallback": True},
            timestamp=datetime.now()
        )
    
    def _load_planning_templates(self) -> Dict[str, str]:
        """Load planning templates for different domains."""
        return {
            "general": """
Create a step-by-step plan with the following format:
1. [Step description]
2. [Step description]
3. [Step description]
...

Each step should be actionable and specific.
""",
            "software": """
Create a software development plan with the following steps:
1. Requirements analysis
2. System design
3. Implementation
4. Testing
5. Deployment
6. Documentation
""",
            "research": """
Create a research plan with the following steps:
1. Literature review
2. Hypothesis formation
3. Methodology design
4. Data collection
5. Analysis
6. Conclusion and reporting
""",
            "business": """
Create a business plan with the following steps:
1. Market research
2. Strategy development
3. Resource planning
4. Implementation
5. Monitoring and evaluation
6. Scaling and growth
"""
        }
    
    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Get a specific plan by ID."""
        return self.plans.get(plan_id)
    
    def update_plan_status(self, plan_id: str, step_id: str, status: PlanStatus):
        """Update the status of a plan step."""
        plan = self.plans.get(plan_id)
        if plan:
            for step in plan.steps:
                if step.id == step_id:
                    step.status = status
                    plan.updated_at = datetime.now()
                    break
    
    def get_planning_statistics(self) -> Dict[str, Any]:
        """Get statistics about planning performance."""
        total_plans = len(self.plans)
        completed_plans = sum(1 for plan in self.plans.values() if plan.status == PlanStatus.COMPLETED)
        
        return {
            "total_plans": total_plans,
            "completed_plans": completed_plans,
            "completion_rate": completed_plans / total_plans if total_plans > 0 else 0,
            "average_steps_per_plan": sum(len(plan.steps) for plan in self.plans.values()) / total_plans if total_plans > 0 else 0,
            "model_loaded": self.planning_model is not None
        }

