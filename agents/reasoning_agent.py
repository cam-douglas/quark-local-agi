#!/usr/bin/env python3
"""
Reasoning Agent for Quark AI Assistant
=========================================

Manages generalized reasoning operations including logical inference,
causal reasoning, analogical reasoning, and multi-step problem solving.

Part of Pillar 18: Generalized Reasoning
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from agents.base import Agent
from reasoning.generalized_reasoning import GeneralizedReasoning, ReasoningType, ReasoningStep, ReasoningChain


class ReasoningAgent(Agent):
    """Advanced reasoning agent for generalized reasoning operations."""
    
    def __init__(self, model_name: str = "reasoning_agent", reasoning_dir: str = None):
        super().__init__(model_name)
        self.reasoning_dir = reasoning_dir or os.path.join(os.path.dirname(__file__), '..', 'reasoning_data')
        os.makedirs(self.reasoning_dir, exist_ok=True)
        
        # Initialize generalized reasoning system
        self.reasoning_engine = GeneralizedReasoning(self.reasoning_dir)
        
        # Reasoning operation settings
        self.auto_explanation = True
        self.confidence_threshold = 0.5
        self.max_reasoning_steps = 10
        
        # Reasoning tracking
        self.reasoning_operations = []
        self.reasoning_performance = {
            'total_operations': 0,
            'successful_operations': 0,
            'average_confidence': 0.0,
            'reasoning_types_used': {}
        }
    
    def load_model(self):
        """Load reasoning models and components."""
        try:
            # Initialize generalized reasoning system
            self.reasoning_engine = GeneralizedReasoning(self.reasoning_dir)
            
            return True
        except Exception as e:
            print(f"Error loading reasoning models: {e}")
            return False
    
    def generate(self, input_data: str, operation: str = "deductive_reasoning", **kwargs) -> Dict[str, Any]:
        """
        Generate reasoning operations or perform reasoning management.
        
        Args:
            input_data: Problem statement or premises
            operation: Reasoning operation to perform
            **kwargs: Additional parameters
            
        Returns:
            Reasoning operation result
        """
        try:
            if operation == "deductive_reasoning":
                return self._deductive_reasoning(input_data, **kwargs)
            elif operation == "causal_reasoning":
                return self._causal_reasoning(input_data, **kwargs)
            elif operation == "analogical_reasoning":
                return self._analogical_reasoning(input_data, **kwargs)
            elif operation == "multi_step_problem_solving":
                return self._multi_step_problem_solving(input_data, **kwargs)
            elif operation == "get_reasoning_stats":
                return self._get_reasoning_stats()
            elif operation == "export_reasoning_data":
                return self._export_reasoning_data(**kwargs)
            elif operation == "explain_reasoning":
                return self._explain_reasoning(input_data, **kwargs)
            elif operation == "validate_reasoning":
                return self._validate_reasoning(input_data, **kwargs)
            else:
                return {"error": f"Unknown reasoning operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Reasoning operation failed: {str(e)}"}
    
    def _deductive_reasoning(self, premises: str, target_conclusion: str = None) -> Dict[str, Any]:
        """Perform deductive reasoning from premises."""
        try:
            # Parse premises
            if isinstance(premises, str):
                premises_list = [p.strip() for p in premises.split('\n') if p.strip()]
            else:
                premises_list = premises
            
            # Perform deductive reasoning
            result = self.reasoning_engine.deductive_reasoning(premises_list, target_conclusion)
            
            # Track operation
            self.reasoning_operations.append({
                'operation': 'deductive_reasoning',
                'premises_count': len(premises_list),
                'target_conclusion': target_conclusion,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to perform deductive reasoning: {str(e)}"}
    
    def _causal_reasoning(self, events: str, target_event: str = None) -> Dict[str, Any]:
        """Perform causal reasoning about events."""
        try:
            # Parse events
            if isinstance(events, str):
                events_list = [e.strip() for e in events.split('\n') if e.strip()]
            else:
                events_list = events
            
            # Perform causal reasoning
            result = self.reasoning_engine.causal_reasoning(events_list, target_event)
            
            # Track operation
            self.reasoning_operations.append({
                'operation': 'causal_reasoning',
                'events_count': len(events_list),
                'target_event': target_event,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to perform causal reasoning: {str(e)}"}
    
    def _analogical_reasoning(self, domains: str, mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """Perform analogical reasoning between domains."""
        try:
            # Parse domains
            if isinstance(domains, str):
                # Assume format: "Source domain: ... Target domain: ..."
                parts = domains.split('Target domain:')
                if len(parts) == 2:
                    source_domain = parts[0].replace('Source domain:', '').strip()
                    target_domain = parts[1].strip()
                else:
                    return {"error": "Invalid domain format. Expected 'Source domain: ... Target domain: ...'"}
            else:
                source_domain = domains.get('source', '')
                target_domain = domains.get('target', '')
            
            # Perform analogical reasoning
            result = self.reasoning_engine.analogical_reasoning(source_domain, target_domain, mapping)
            
            # Track operation
            self.reasoning_operations.append({
                'operation': 'analogical_reasoning',
                'source_domain': source_domain,
                'target_domain': target_domain,
                'mapping': mapping,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to perform analogical reasoning: {str(e)}"}
    
    def _multi_step_problem_solving(self, problem: str, steps: List[str] = None) -> Dict[str, Any]:
        """Solve complex problems through multi-step reasoning."""
        try:
            # Perform multi-step problem solving
            result = self.reasoning_engine.multi_step_problem_solving(problem, steps)
            
            # Track operation
            self.reasoning_operations.append({
                'operation': 'multi_step_problem_solving',
                'problem': problem,
                'steps_provided': steps is not None,
                'steps_count': len(steps) if steps else 0,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to perform multi-step problem solving: {str(e)}"}
    
    def _get_reasoning_stats(self) -> Dict[str, Any]:
        """Get comprehensive reasoning statistics."""
        try:
            # Get engine stats
            engine_stats = self.reasoning_engine.get_reasoning_stats()
            
            # Get agent performance stats
            performance_stats = {
                'total_operations': self.reasoning_performance['total_operations'],
                'successful_operations': self.reasoning_performance['successful_operations'],
                'success_rate': (self.reasoning_performance['successful_operations'] / 
                               self.reasoning_performance['total_operations'] 
                               if self.reasoning_performance['total_operations'] > 0 else 0),
                'average_confidence': self.reasoning_performance['average_confidence'],
                'reasoning_types_used': self.reasoning_performance['reasoning_types_used']
            }
            
            # Get recent operations
            recent_operations = self.reasoning_operations[-10:] if self.reasoning_operations else []
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "engine_stats": engine_stats,
                "performance_stats": performance_stats,
                "recent_operations": recent_operations,
                "settings": {
                    "auto_explanation": self.auto_explanation,
                    "confidence_threshold": self.confidence_threshold,
                    "max_reasoning_steps": self.max_reasoning_steps
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to get reasoning stats: {str(e)}"}
    
    def _export_reasoning_data(self, filename: str = None) -> Dict[str, Any]:
        """Export reasoning data to JSON."""
        try:
            export_file = self.reasoning_engine.export_reasoning_data(filename)
            
            return {
                "status": "success",
                "export_file": export_file,
                "export_timestamp": datetime.now().isoformat(),
                "message": f"Reasoning data exported to: {export_file}"
            }
            
        except Exception as e:
            return {"error": f"Failed to export reasoning data: {str(e)}"}
    
    def _explain_reasoning(self, reasoning_chain_id: str) -> Dict[str, Any]:
        """Generate detailed explanation of a reasoning chain."""
        try:
            # Find the reasoning chain
            target_chain = None
            for chain in self.reasoning_engine.reasoning_chains:
                if chain.chain_id == reasoning_chain_id:
                    target_chain = chain
                    break
            
            if not target_chain:
                return {"error": f"Reasoning chain {reasoning_chain_id} not found"}
            
            # Generate explanation
            explanation = {
                "chain_id": target_chain.chain_id,
                "problem": target_chain.problem,
                "reasoning_type": target_chain.reasoning_type.value,
                "overall_confidence": target_chain.overall_confidence,
                "steps": []
            }
            
            for step in target_chain.steps:
                step_explanation = {
                    "step_id": step.step_id,
                    "reasoning_type": step.reasoning_type.value,
                    "premises": step.premises,
                    "conclusion": step.conclusion,
                    "confidence": step.confidence,
                    "explanation": step.explanation
                }
                explanation["steps"].append(step_explanation)
            
            return {
                "status": "success",
                "explanation": explanation,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to explain reasoning: {str(e)}"}
    
    def _validate_reasoning(self, reasoning_chain_id: str) -> Dict[str, Any]:
        """Validate the correctness of a reasoning chain."""
        try:
            # Find the reasoning chain
            target_chain = None
            for chain in self.reasoning_engine.reasoning_chains:
                if chain.chain_id == reasoning_chain_id:
                    target_chain = chain
                    break
            
            if not target_chain:
                return {"error": f"Reasoning chain {reasoning_chain_id} not found"}
            
            # Validate the reasoning chain
            validation_results = {
                "chain_id": target_chain.chain_id,
                "is_valid": True,
                "issues": [],
                "strengths": [],
                "overall_score": 0.0
            }
            
            # Check each step
            step_scores = []
            for step in target_chain.steps:
                step_score = self._validate_reasoning_step(step)
                step_scores.append(step_score)
                
                if step_score < 0.5:
                    validation_results["issues"].append(f"Step {step.step_id}: Low confidence ({step_score:.2f})")
                else:
                    validation_results["strengths"].append(f"Step {step.step_id}: Good reasoning")
            
            # Calculate overall score
            if step_scores:
                validation_results["overall_score"] = sum(step_scores) / len(step_scores)
                if validation_results["overall_score"] < 0.5:
                    validation_results["is_valid"] = False
            
            return {
                "status": "success",
                "validation": validation_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to validate reasoning: {str(e)}"}
    
    def _validate_reasoning_step(self, step: ReasoningStep) -> float:
        """Validate a single reasoning step."""
        # Simple validation based on confidence and reasoning type
        base_score = step.confidence
        
        # Adjust score based on reasoning type
        if step.reasoning_type == ReasoningType.DEDUCTIVE:
            # Deductive reasoning should have high confidence
            if step.confidence < 0.8:
                base_score *= 0.8
        elif step.reasoning_type == ReasoningType.CAUSAL:
            # Causal reasoning can have moderate confidence
            if step.confidence < 0.6:
                base_score *= 0.9
        elif step.reasoning_type == ReasoningType.ANALOGICAL:
            # Analogical reasoning can have lower confidence
            if step.confidence < 0.4:
                base_score *= 0.7
        
        return min(base_score, 1.0)
    
    def _update_performance_stats(self, result: Dict[str, Any]):
        """Update reasoning performance statistics."""
        self.reasoning_performance['total_operations'] += 1
        
        if result.get('status') == 'success':
            self.reasoning_performance['successful_operations'] += 1
        
        # Update confidence
        confidence = result.get('confidence', 0.0)
        current_avg = self.reasoning_performance['average_confidence']
        total_ops = self.reasoning_performance['total_operations']
        self.reasoning_performance['average_confidence'] = (
            (current_avg * (total_ops - 1) + confidence) / total_ops
        )
        
        # Update reasoning type usage
        reasoning_type = result.get('reasoning_type', 'unknown')
        if reasoning_type not in self.reasoning_performance['reasoning_types_used']:
            self.reasoning_performance['reasoning_types_used'][reasoning_type] = 0
        self.reasoning_performance['reasoning_types_used'][reasoning_type] += 1
    
    def get_reasoning_recommendations(self, context: str = None) -> Dict[str, Any]:
        """Get reasoning-related recommendations."""
        try:
            recommendations = []
            
            # Check reasoning performance
            success_rate = (self.reasoning_performance['successful_operations'] / 
                          self.reasoning_performance['total_operations'] 
                          if self.reasoning_performance['total_operations'] > 0 else 0)
            
            if success_rate < 0.7:
                recommendations.append({
                    'type': 'performance',
                    'priority': 'high',
                    'message': f'Reasoning success rate is {success_rate:.1%}, consider improving reasoning strategies',
                    'action': 'review_reasoning_methods'
                })
            
            # Check confidence levels
            avg_confidence = self.reasoning_performance['average_confidence']
            if avg_confidence < 0.6:
                recommendations.append({
                    'type': 'confidence',
                    'priority': 'medium',
                    'message': f'Average confidence is {avg_confidence:.1%}, consider more robust reasoning',
                    'action': 'improve_reasoning_confidence'
                })
            
            # Check reasoning type distribution
            type_usage = self.reasoning_performance['reasoning_types_used']
            if len(type_usage) < 3:
                recommendations.append({
                    'type': 'diversity',
                    'priority': 'low',
                    'message': f'Using only {len(type_usage)} reasoning types, consider diversifying',
                    'action': 'explore_different_reasoning_types'
                })
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "recommendations_count": len(recommendations)
            }
            
        except Exception as e:
            return {"error": f"Failed to get recommendations: {str(e)}"}

