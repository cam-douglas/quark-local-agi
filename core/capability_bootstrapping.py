#!/usr/bin/env python3
"""
Capability Bootstrapping for Quark AI Assistant
Enables the AI to develop new capabilities through self-directed learning
"""

import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import random
from core.safety_guardrails import SafetyGuardrails, ChangeType, ChangeSeverity

@dataclass
class Capability:
    """A capability that can be developed."""
    name: str
    description: str
    category: str
    complexity: str  # easy, medium, hard
    prerequisites: List[str]
    learning_examples: List[Dict]
    performance_metrics: Dict[str, float]
    status: str  # not_started, learning, mastered, failed
    created_at: float
    last_updated: float

@dataclass
class LearningTask:
    """A specific learning task for capability development."""
    task_id: str
    capability_name: str
    task_type: str  # practice, evaluation, integration
    input_data: str
    expected_output: str
    actual_output: str
    success: bool
    feedback_score: float
    timestamp: float
    metadata: Dict[str, Any]

class CapabilityBootstrapping:
    def __init__(self, capabilities_dir: str = None):
        self.capabilities_dir = capabilities_dir or os.path.join(os.path.dirname(__file__), '..', 'capabilities')
        os.makedirs(self.capabilities_dir, exist_ok=True)
        
        # Capability definitions
        self.capabilities = self._load_capability_definitions()
        self.active_capabilities = {}
        self.learning_tasks = []
        
        # Bootstrapping settings
        self.auto_bootstrapping = True
        self.learning_threshold = 0.8
        self.max_concurrent_capabilities = 3
        
        # Initialize safety guardrails
        self.safety_guardrails = SafetyGuardrails()
        
    def _load_capability_definitions(self) -> Dict[str, Capability]:
        """Load predefined capability definitions."""
        capabilities = {}
        
        # Basic capabilities
        capabilities['text_summarization'] = Capability(
            name="text_summarization",
            description="Summarize long text into concise summaries",
            category="text_processing",
            complexity="medium",
            prerequisites=[],
            learning_examples=[],
            performance_metrics={},
            status="not_started",
            created_at=time.time(),
            last_updated=time.time()
        )
        
        capabilities['code_generation'] = Capability(
            name="code_generation",
            description="Generate code based on natural language descriptions",
            category="programming",
            complexity="hard",
            prerequisites=["text_understanding"],
            learning_examples=[],
            performance_metrics={},
            status="not_started",
            created_at=time.time(),
            last_updated=time.time()
        )
        
        capabilities['mathematical_reasoning'] = Capability(
            name="mathematical_reasoning",
            description="Solve mathematical problems and equations",
            category="reasoning",
            complexity="medium",
            prerequisites=[],
            learning_examples=[],
            performance_metrics={},
            status="not_started",
            created_at=time.time(),
            last_updated=time.time()
        )
        
        capabilities['creative_writing'] = Capability(
            name="creative_writing",
            description="Generate creative content like stories and poems",
            category="creative",
            complexity="medium",
            prerequisites=[],
            learning_examples=[],
            performance_metrics={},
            status="not_started",
            created_at=time.time(),
            last_updated=time.time()
        )
        
        capabilities['data_analysis'] = Capability(
            name="data_analysis",
            description="Analyze and interpret data sets",
            category="analytics",
            complexity="hard",
            prerequisites=["mathematical_reasoning"],
            learning_examples=[],
            performance_metrics={},
            status="not_started",
            created_at=time.time(),
            last_updated=time.time()
        )
        
        return capabilities
        
    def identify_learning_opportunities(self, user_interactions: List[Dict]) -> List[Dict]:
        """Identify opportunities for capability development."""
        opportunities = []
        
        # Analyze user interactions for patterns
        interaction_patterns = defaultdict(int)
        for interaction in user_interactions:
            category = interaction.get('category', 'general')
            interaction_patterns[category] += 1
        
        # Identify gaps in current capabilities
        for capability_name, capability in self.capabilities.items():
            if capability.status == "not_started":
                # Check if this capability would be useful based on interactions
                usefulness_score = self._calculate_capability_usefulness(capability, interaction_patterns)
                
                if usefulness_score > 0.5:  # Threshold for considering a capability
                    opportunities.append({
                        'capability_name': capability_name,
                        'capability': capability,
                        'usefulness_score': usefulness_score,
                        'priority': 'high' if usefulness_score > 0.8 else 'medium',
                        'reason': f"Would improve {capability.category} tasks"
                    })
        
        return sorted(opportunities, key=lambda x: x['usefulness_score'], reverse=True)
        
    def _calculate_capability_usefulness(self, capability: Capability, 
                                       interaction_patterns: Dict[str, int]) -> float:
        """Calculate how useful a capability would be based on interaction patterns."""
        # Simple heuristic: more interactions in related categories = higher usefulness
        category_interactions = interaction_patterns.get(capability.category, 0)
        total_interactions = sum(interaction_patterns.values())
        
        if total_interactions == 0:
            return 0.0
        
        # Base score from category interactions
        category_score = category_interactions / total_interactions
        
        # Adjust based on complexity (easier capabilities get higher scores)
        complexity_multiplier = {
            'easy': 1.2,
            'medium': 1.0,
            'hard': 0.8
        }.get(capability.complexity, 1.0)
        
        return category_score * complexity_multiplier
        
    def start_capability_learning(self, capability_name: str, user_confirmation: bool = False) -> Dict[str, Any]:
        """Start learning a new capability with safety checks."""
        if capability_name not in self.capabilities:
            return {'status': 'error', 'message': f'Capability {capability_name} not found'}
        
        capability = self.capabilities[capability_name]
        
        # Check prerequisites
        if capability.prerequisites:
            missing_prereqs = []
            for prereq in capability.prerequisites:
                if prereq not in self.capabilities or self.capabilities[prereq].status != "mastered":
                    missing_prereqs.append(prereq)
            
            if missing_prereqs:
                return {
                    'status': 'prerequisites_not_met',
                    'missing_prereqs': missing_prereqs
                }
        
        # Check if we can start learning (not too many active capabilities)
        active_count = len([c for c in self.capabilities.values() if c.status == "learning"])
        if active_count >= self.max_concurrent_capabilities:
            return {
                'status': 'too_many_active',
                'active_capabilities': active_count,
                'max_allowed': self.max_concurrent_capabilities
            }
        
        # Safety check for capability learning
        impact_analysis = {
            'affects_core_functionality': False,
            'data_modification': False,
            'performance_impact': 'low',
            'user_experience_impact': 'low',
            'capability_name': capability_name,
            'capability_category': capability.category,
            'capability_complexity': capability.complexity
        }
        
        risk_assessment = self.safety_guardrails.assess_change_risk(
            ChangeType.CAPABILITY_LEARNING, impact_analysis
        )
        
        # Propose change for safety review
        change_id = self.safety_guardrails.propose_change(
            change_type=ChangeType.CAPABILITY_LEARNING,
            description=f"Start learning capability: {capability_name}",
            impact_analysis=impact_analysis,
            severity=ChangeSeverity(risk_assessment['severity'])
        )
        
        if change_id == "safety_disabled":
            # Proceed without safety checks
            pass
        elif change_id == "rate_limited":
            return {'status': 'rate_limited', 'message': 'Too many changes in the last hour'}
        else:
            # Check if change requires confirmation
            pending_changes = self.safety_guardrails.get_pending_changes()
            for change in pending_changes:
                if change['change_id'] == change_id and change['requires_confirmation']:
                    if not user_confirmation:
                        return {
                            'status': 'confirmation_required',
                            'change_id': change_id,
                            'risk_assessment': risk_assessment,
                            'message': 'User confirmation required for this capability learning'
                        }
            
            # Approve the change
            approval_result = self.safety_guardrails.approve_change(change_id, user_confirmation)
            if approval_result['status'] != 'approved':
                return approval_result
        
        # Start learning
        capability.status = "learning"
        capability.last_updated = time.time()
        
        # Generate initial learning tasks
        learning_tasks = self._generate_learning_tasks(capability)
        
        return {
            'status': 'started',
            'capability_name': capability_name,
            'learning_tasks': len(learning_tasks),
            'estimated_duration': self._estimate_learning_duration(capability),
            'safety_approved': True,
            'risk_assessment': risk_assessment
        }
        
    def _generate_learning_tasks(self, capability: Capability) -> List[LearningTask]:
        """Generate learning tasks for a capability."""
        tasks = []
        
        # Generate practice tasks based on capability type
        if capability.category == "text_processing":
            tasks.extend(self._generate_text_processing_tasks(capability))
        elif capability.category == "programming":
            tasks.extend(self._generate_programming_tasks(capability))
        elif capability.category == "reasoning":
            tasks.extend(self._generate_reasoning_tasks(capability))
        elif capability.category == "creative":
            tasks.extend(self._generate_creative_tasks(capability))
        elif capability.category == "analytics":
            tasks.extend(self._generate_analytics_tasks(capability))
        
        return tasks
        
    def _generate_text_processing_tasks(self, capability: Capability) -> List[LearningTask]:
        """Generate text processing learning tasks."""
        tasks = []
        
        if capability.name == "text_summarization":
            sample_texts = [
                "Artificial intelligence is a field of computer science that aims to create machines capable of intelligent behavior. It encompasses various techniques including machine learning, natural language processing, and computer vision.",
                "Climate change refers to long-term shifts in global weather patterns and average temperatures. It is primarily caused by human activities such as burning fossil fuels and deforestation.",
                "The human brain is the most complex organ in the body, containing approximately 86 billion neurons. It is responsible for all cognitive functions including thinking, memory, and consciousness."
            ]
            
            for i, text in enumerate(sample_texts):
                task = LearningTask(
                    task_id=f"summarization_task_{i}",
                    capability_name=capability.name,
                    task_type="practice",
                    input_data=text,
                    expected_output="Summarize the key points of this text",
                    actual_output="",
                    success=False,
                    feedback_score=0.0,
                    timestamp=time.time(),
                    metadata={'text_length': len(text)}
                )
                tasks.append(task)
        
        return tasks
        
    def _generate_programming_tasks(self, capability: Capability) -> List[LearningTask]:
        """Generate programming learning tasks."""
        tasks = []
        
        if capability.name == "code_generation":
            sample_requests = [
                "Write a Python function to calculate the factorial of a number",
                "Create a function to reverse a string in JavaScript",
                "Write a SQL query to find all users who registered in the last 30 days"
            ]
            
            for i, request in enumerate(sample_requests):
                task = LearningTask(
                    task_id=f"code_task_{i}",
                    capability_name=capability.name,
                    task_type="practice",
                    input_data=request,
                    expected_output="Generate code based on this request",
                    actual_output="",
                    success=False,
                    feedback_score=0.0,
                    timestamp=time.time(),
                    metadata={'language': 'python' if 'Python' in request else 'javascript' if 'JavaScript' in request else 'sql'}
                )
                tasks.append(task)
        
        return tasks
        
    def _generate_reasoning_tasks(self, capability: Capability) -> List[LearningTask]:
        """Generate reasoning learning tasks."""
        tasks = []
        
        if capability.name == "mathematical_reasoning":
            sample_problems = [
                "Solve: 2x + 5 = 13",
                "What is 15% of 200?",
                "If a train travels 120 km in 2 hours, what is its average speed?"
            ]
            
            for i, problem in enumerate(sample_problems):
                task = LearningTask(
                    task_id=f"math_task_{i}",
                    capability_name=capability.name,
                    task_type="practice",
                    input_data=problem,
                    expected_output="Solve this mathematical problem step by step",
                    actual_output="",
                    success=False,
                    feedback_score=0.0,
                    timestamp=time.time(),
                    metadata={'problem_type': 'algebra' if 'x' in problem else 'percentage' if '%' in problem else 'speed'}
                )
                tasks.append(task)
        
        return tasks
        
    def _generate_creative_tasks(self, capability: Capability) -> List[LearningTask]:
        """Generate creative learning tasks."""
        tasks = []
        
        if capability.name == "creative_writing":
            sample_prompts = [
                "Write a short story about a robot learning to paint",
                "Create a poem about artificial intelligence",
                "Write a dialogue between two AI systems discussing consciousness"
            ]
            
            for i, prompt in enumerate(sample_prompts):
                task = LearningTask(
                    task_id=f"creative_task_{i}",
                    capability_name=capability.name,
                    task_type="practice",
                    input_data=prompt,
                    expected_output="Generate creative content based on this prompt",
                    actual_output="",
                    success=False,
                    feedback_score=0.0,
                    timestamp=time.time(),
                    metadata={'genre': 'story' if 'story' in prompt else 'poem' if 'poem' in prompt else 'dialogue'}
                )
                tasks.append(task)
        
        return tasks
        
    def _generate_analytics_tasks(self, capability: Capability) -> List[LearningTask]:
        """Generate analytics learning tasks."""
        tasks = []
        
        if capability.name == "data_analysis":
            sample_datasets = [
                "Analyze sales data: [100, 150, 200, 175, 225] for months Jan-May",
                "Interpret survey results: 60% positive, 25% neutral, 15% negative",
                "Calculate correlation between study hours and test scores"
            ]
            
            for i, dataset in enumerate(sample_datasets):
                task = LearningTask(
                    task_id=f"analytics_task_{i}",
                    capability_name=capability.name,
                    task_type="practice",
                    input_data=dataset,
                    expected_output="Analyze this data and provide insights",
                    actual_output="",
                    success=False,
                    feedback_score=0.0,
                    timestamp=time.time(),
                    metadata={'data_type': 'time_series' if 'months' in dataset else 'survey' if 'survey' in dataset else 'correlation'}
                )
                tasks.append(task)
        
        return tasks
        
    def _estimate_learning_duration(self, capability: Capability) -> int:
        """Estimate learning duration in minutes."""
        complexity_durations = {
            'easy': 30,
            'medium': 60,
            'hard': 120
        }
        return complexity_durations.get(capability.complexity, 60)
        
    def evaluate_capability_progress(self, capability_name: str) -> Dict[str, Any]:
        """Evaluate progress on a capability."""
        if capability_name not in self.capabilities:
            return {'status': 'error', 'message': f'Capability {capability_name} not found'}
        
        capability = self.capabilities[capability_name]
        
        # Get learning tasks for this capability
        capability_tasks = [task for task in self.learning_tasks if task.capability_name == capability_name]
        
        if not capability_tasks:
            return {
                'status': 'no_progress',
                'capability_name': capability_name,
                'message': 'No learning tasks completed yet'
            }
        
        # Calculate progress metrics
        total_tasks = len(capability_tasks)
        completed_tasks = len([task for task in capability_tasks if task.success])
        average_score = sum(task.feedback_score for task in capability_tasks) / total_tasks
        
        # Determine if capability is mastered
        if completed_tasks >= 5 and average_score >= self.learning_threshold:
            capability.status = "mastered"
            capability.last_updated = time.time()
            status = "mastered"
        elif completed_tasks >= 10 and average_score < 0.5:
            capability.status = "failed"
            capability.last_updated = time.time()
            status = "failed"
        else:
            status = "learning"
        
        return {
            'status': status,
            'capability_name': capability_name,
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'completion_rate': completed_tasks / total_tasks if total_tasks > 0 else 0,
            'average_score': average_score,
            'progress_percentage': min(100, (average_score / self.learning_threshold) * 100)
        }
        
    def get_capability_status(self) -> Dict[str, Any]:
        """Get status of all capabilities."""
        status_summary = {
            'total_capabilities': len(self.capabilities),
            'not_started': 0,
            'learning': 0,
            'mastered': 0,
            'failed': 0,
            'capabilities': {}
        }
        
        for name, capability in self.capabilities.items():
            status_summary['capabilities'][name] = {
                'status': capability.status,
                'category': capability.category,
                'complexity': capability.complexity,
                'created_at': capability.created_at,
                'last_updated': capability.last_updated
            }
            status_summary[capability.status] += 1
        
        return status_summary
        
    def export_capability_data(self, filename: str = None) -> str:
        """Export capability data to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capability_data_{timestamp}.json"
        
        filepath = os.path.join(self.capabilities_dir, filename)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'capabilities': {name: asdict(cap) for name, cap in self.capabilities.items()},
            'learning_tasks': [asdict(task) for task in self.learning_tasks],
            'status_summary': self.get_capability_status()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath 