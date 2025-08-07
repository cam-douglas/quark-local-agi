#!/usr/bin/env python3
"""
Generalized Reasoning Engine for Quark AI Assistant
=====================================================

Implements advanced reasoning capabilities including abstract reasoning,
logical inference, causal reasoning, and multi-step problem solving.

Part of Pillar 18: Generalized Reasoning
"""

import os
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
import networkx as nx
from collections import defaultdict

from agents.base import Agent


class ReasoningType(Enum):
    """Types of reasoning operations."""
    DEDUCTIVE = "deductive"      # Logical deduction from premises
    INDUCTIVE = "inductive"       # Generalization from specific cases
    ABDUCTIVE = "abductive"       # Best explanation inference
    CAUSAL = "causal"            # Cause-and-effect reasoning
    ANALOGICAL = "analogical"     # Similarity-based reasoning
    SPATIAL = "spatial"          # Spatial relationship reasoning
    TEMPORAL = "temporal"        # Temporal relationship reasoning


class ConfidenceLevel(Enum):
    """Confidence levels for reasoning results."""
    CERTAIN = "certain"          # 100% confidence
    HIGH = "high"               # 80-99% confidence
    MEDIUM = "medium"           # 50-79% confidence
    LOW = "low"                 # 20-49% confidence
    UNCERTAIN = "uncertain"     # <20% confidence


@dataclass
class ReasoningStep:
    """Individual step in a reasoning process."""
    step_id: str
    reasoning_type: ReasoningType
    premises: List[str]
    conclusion: str
    confidence: float
    explanation: str
    dependencies: List[str]  # IDs of dependent steps
    timestamp: float


@dataclass
class ReasoningChain:
    """Complete reasoning chain with multiple steps."""
    chain_id: str
    problem: str
    steps: List[ReasoningStep]
    final_conclusion: str
    overall_confidence: float
    reasoning_type: ReasoningType
    timestamp: float


@dataclass
class LogicalRule:
    """Logical rule for deductive reasoning."""
    rule_id: str
    name: str
    premises: List[str]
    conclusion: str
    rule_type: str  # "modus_ponens", "modus_tollens", "syllogism", etc.
    confidence: float


class GeneralizedReasoning:
    """Advanced generalized reasoning engine with multiple reasoning types."""
    
    def __init__(self, reasoning_dir: str = None):
        self.reasoning_dir = reasoning_dir or os.path.join(os.path.dirname(__file__), '..', 'reasoning_data')
        os.makedirs(self.reasoning_dir, exist_ok=True)
        
        # Reasoning components
        self.logical_rules = {}
        self.causal_models = {}
        self.analogical_patterns = {}
        
        # Reasoning tracking
        self.reasoning_chains = []
        self.reasoning_stats = {
            'total_chains': 0,
            'successful_chains': 0,
            'average_confidence': 0.0,
            'reasoning_types_used': defaultdict(int)
        }
        
        # Load reasoning components
        self._load_logical_rules()
        self._load_causal_models()
        self._load_analogical_patterns()
    
    def _load_logical_rules(self):
        """Load logical rules for deductive reasoning."""
        self.logical_rules = {
            'modus_ponens': LogicalRule(
                rule_id='modus_ponens',
                name='Modus Ponens',
                premises=['If P then Q', 'P'],
                conclusion='Q',
                rule_type='deductive',
                confidence=1.0
            ),
            'modus_tollens': LogicalRule(
                rule_id='modus_tollens',
                name='Modus Tollens',
                premises=['If P then Q', 'Not Q'],
                conclusion='Not P',
                rule_type='deductive',
                confidence=1.0
            ),
            'syllogism': LogicalRule(
                rule_id='syllogism',
                name='Syllogism',
                premises=['All A are B', 'All B are C'],
                conclusion='All A are C',
                rule_type='deductive',
                confidence=1.0
            ),
            'disjunctive_syllogism': LogicalRule(
                rule_id='disjunctive_syllogism',
                name='Disjunctive Syllogism',
                premises=['P or Q', 'Not P'],
                conclusion='Q',
                rule_type='deductive',
                confidence=1.0
            )
        }
    
    def _load_causal_models(self):
        """Load causal reasoning models."""
        self.causal_models = {
            'direct_causation': {
                'pattern': r'(\w+)\s+causes\s+(\w+)',
                'confidence': 0.8,
                'description': 'Direct cause-and-effect relationship'
            },
            'necessary_condition': {
                'pattern': r'(\w+)\s+is\s+necessary\s+for\s+(\w+)',
                'confidence': 0.7,
                'description': 'Necessary condition relationship'
            },
            'sufficient_condition': {
                'pattern': r'(\w+)\s+is\s+sufficient\s+for\s+(\w+)',
                'confidence': 0.7,
                'description': 'Sufficient condition relationship'
            }
        }
    
    def _load_analogical_patterns(self):
        """Load analogical reasoning patterns."""
        self.analogical_patterns = {
            'structural_similarity': {
                'pattern': r'(\w+)\s+is\s+like\s+(\w+)\s+because\s+(\w+)',
                'confidence': 0.6,
                'description': 'Structural similarity analogy'
            },
            'functional_similarity': {
                'pattern': r'(\w+)\s+works\s+like\s+(\w+)',
                'confidence': 0.5,
                'description': 'Functional similarity analogy'
            }
        }
    
    def deductive_reasoning(self, premises: List[str], target_conclusion: str = None) -> Dict[str, Any]:
        """Perform deductive reasoning from premises."""
        try:
            chain_id = f"deductive_{int(time.time())}"
            steps = []
            current_premises = premises.copy()
            
            # Apply logical rules
            for rule_name, rule in self.logical_rules.items():
                if self._can_apply_rule(rule, current_premises):
                    conclusion = self._apply_logical_rule(rule, current_premises)
                    
                    step = ReasoningStep(
                        step_id=f"{chain_id}_step_{len(steps)}",
                        reasoning_type=ReasoningType.DEDUCTIVE,
                        premises=rule.premises,
                        conclusion=conclusion,
                        confidence=rule.confidence,
                        explanation=f"Applied {rule.name} rule",
                        dependencies=[],
                        timestamp=time.time()
                    )
                    steps.append(step)
                    current_premises.append(conclusion)
            
            # Check if target conclusion was reached
            final_conclusion = None
            if target_conclusion:
                if target_conclusion in current_premises:
                    final_conclusion = target_conclusion
                else:
                    # Try to derive target conclusion
                    derived = self._derive_conclusion(current_premises, target_conclusion)
                    if derived:
                        final_conclusion = target_conclusion
            else:
                final_conclusion = current_premises[-1] if current_premises else None
            
            # Calculate overall confidence
            overall_confidence = sum(step.confidence for step in steps) / len(steps) if steps else 0.0
            
            reasoning_chain = ReasoningChain(
                chain_id=chain_id,
                problem="Deductive reasoning from premises",
                steps=steps,
                final_conclusion=final_conclusion,
                overall_confidence=overall_confidence,
                reasoning_type=ReasoningType.DEDUCTIVE,
                timestamp=time.time()
            )
            
            self.reasoning_chains.append(reasoning_chain)
            self._update_stats(reasoning_chain)
            
            return {
                "status": "success",
                "chain_id": chain_id,
                "steps": len(steps),
                "final_conclusion": final_conclusion,
                "confidence": overall_confidence,
                "reasoning_type": "deductive"
            }
            
        except Exception as e:
            return {"error": f"Deductive reasoning failed: {str(e)}"}
    
    def _can_apply_rule(self, rule: LogicalRule, premises: List[str]) -> bool:
        """Check if a logical rule can be applied to the premises."""
        # Simple pattern matching for rule application
        for premise in rule.premises:
            if not any(self._matches_pattern(premise, p) for p in premises):
                return False
        return True
    
    def _matches_pattern(self, pattern: str, text: str) -> bool:
        """Check if text matches a logical pattern."""
        # Convert pattern to regex-like matching
        pattern_words = pattern.lower().split()
        text_words = text.lower().split()
        
        # Simple word-based matching
        pattern_set = set(pattern_words)
        text_set = set(text_words)
        
        return len(pattern_set.intersection(text_set)) >= len(pattern_set) * 0.7
    
    def _apply_logical_rule(self, rule: LogicalRule, premises: List[str]) -> str:
        """Apply a logical rule to derive a conclusion."""
        # Simple rule application based on rule type
        if rule.rule_type == 'modus_ponens':
            # Find "If P then Q" and "P" to conclude "Q"
            for premise in premises:
                if 'if' in premise.lower() and 'then' in premise.lower():
                    # Extract Q from "If P then Q"
                    parts = premise.lower().split('then')
                    if len(parts) == 2:
                        q_part = parts[1].strip()
                        # Check if we have the antecedent
                        antecedent = parts[0].replace('if', '').strip()
                        if any(antecedent in p.lower() for p in premises):
                            return q_part
        
        # Default: return the rule's conclusion
        return rule.conclusion
    
    def _derive_conclusion(self, premises: List[str], target: str) -> bool:
        """Try to derive a target conclusion from premises."""
        # Simple derivation check
        target_words = set(target.lower().split())
        for premise in premises:
            premise_words = set(premise.lower().split())
            if len(target_words.intersection(premise_words)) >= len(target_words) * 0.8:
                return True
        return False
    
    def causal_reasoning(self, events: List[str], target_event: str = None) -> Dict[str, Any]:
        """Perform causal reasoning about events."""
        try:
            chain_id = f"causal_{int(time.time())}"
            steps = []
            
            # Analyze causal relationships
            causal_relationships = []
            for event in events:
                for model_name, model in self.causal_models.items():
                    matches = re.findall(model['pattern'], event, re.IGNORECASE)
                    for match in matches:
                        causal_relationships.append({
                            'cause': match[0],
                            'effect': match[1],
                            'pattern': model_name,
                            'confidence': model['confidence']
                        })
            
            # Create reasoning steps for each causal relationship
            for i, rel in enumerate(causal_relationships):
                step = ReasoningStep(
                    step_id=f"{chain_id}_step_{i}",
                    reasoning_type=ReasoningType.CAUSAL,
                    premises=[f"{rel['cause']} causes {rel['effect']}"],
                    conclusion=f"{rel['cause']} is the cause of {rel['effect']}",
                    confidence=rel['confidence'],
                    explanation=f"Identified {rel['pattern']} relationship",
                    dependencies=[],
                    timestamp=time.time()
                )
                steps.append(step)
            
            # Find causal chains
            causal_chains = self._find_causal_chains(causal_relationships)
            
            final_conclusion = None
            if target_event:
                # Try to find causes of target event
                causes = self._find_causes_of_event(causal_relationships, target_event)
                if causes:
                    final_conclusion = f"Causes of {target_event}: {', '.join(causes)}"
            
            overall_confidence = sum(step.confidence for step in steps) / len(steps) if steps else 0.0
            
            reasoning_chain = ReasoningChain(
                chain_id=chain_id,
                problem="Causal reasoning about events",
                steps=steps,
                final_conclusion=final_conclusion,
                overall_confidence=overall_confidence,
                reasoning_type=ReasoningType.CAUSAL,
                timestamp=time.time()
            )
            
            self.reasoning_chains.append(reasoning_chain)
            self._update_stats(reasoning_chain)
            
            return {
                "status": "success",
                "chain_id": chain_id,
                "causal_relationships": len(causal_relationships),
                "causal_chains": len(causal_chains),
                "final_conclusion": final_conclusion,
                "confidence": overall_confidence,
                "reasoning_type": "causal"
            }
            
        except Exception as e:
            return {"error": f"Causal reasoning failed: {str(e)}"}
    
    def _find_causal_chains(self, relationships: List[Dict]) -> List[List[str]]:
        """Find chains of causal relationships."""
        # Build causal graph
        causal_graph = nx.DiGraph()
        for rel in relationships:
            causal_graph.add_edge(rel['cause'], rel['effect'])
        
        # Find all paths
        chains = []
        for source in causal_graph.nodes():
            for target in causal_graph.nodes():
                if source != target:
                    try:
                        path = nx.shortest_path(causal_graph, source, target)
                        if len(path) > 1:
                            chains.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        return chains
    
    def _find_causes_of_event(self, relationships: List[Dict], event: str) -> List[str]:
        """Find all causes of a specific event."""
        causes = []
        for rel in relationships:
            if rel['effect'].lower() == event.lower():
                causes.append(rel['cause'])
        return causes
    
    def analogical_reasoning(self, source_domain: str, target_domain: str, 
                           mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """Perform analogical reasoning between domains."""
        try:
            chain_id = f"analogical_{int(time.time())}"
            steps = []
            
            # Analyze analogical patterns
            analogical_insights = []
            for pattern_name, pattern in self.analogical_patterns.items():
                # Look for analogical relationships in the domains
                source_matches = re.findall(pattern['pattern'], source_domain, re.IGNORECASE)
                target_matches = re.findall(pattern['pattern'], target_domain, re.IGNORECASE)
                
                if source_matches and target_matches:
                    analogical_insights.append({
                        'pattern': pattern_name,
                        'source_analogy': source_matches[0],
                        'target_analogy': target_matches[0],
                        'confidence': pattern['confidence']
                    })
            
            # Create reasoning steps
            for i, insight in enumerate(analogical_insights):
                step = ReasoningStep(
                    step_id=f"{chain_id}_step_{i}",
                    reasoning_type=ReasoningType.ANALOGICAL,
                    premises=[f"Source: {insight['source_analogy']}", f"Target: {insight['target_analogy']}"],
                    conclusion=f"Analogical mapping: {insight['source_analogy']} → {insight['target_analogy']}",
                    confidence=insight['confidence'],
                    explanation=f"Identified {insight['pattern']} analogy",
                    dependencies=[],
                    timestamp=time.time()
                )
                steps.append(step)
            
            # Generate analogical conclusions
            analogical_conclusions = []
            if mapping:
                for source_concept, target_concept in mapping.items():
                    analogical_conclusions.append(f"{source_concept} corresponds to {target_concept}")
            
            final_conclusion = "; ".join(analogical_conclusions) if analogical_conclusions else "No clear analogical mapping found"
            overall_confidence = sum(step.confidence for step in steps) / len(steps) if steps else 0.0
            
            reasoning_chain = ReasoningChain(
                chain_id=chain_id,
                problem="Analogical reasoning between domains",
                steps=steps,
                final_conclusion=final_conclusion,
                overall_confidence=overall_confidence,
                reasoning_type=ReasoningType.ANALOGICAL,
                timestamp=time.time()
            )
            
            self.reasoning_chains.append(reasoning_chain)
            self._update_stats(reasoning_chain)
            
            return {
                "status": "success",
                "chain_id": chain_id,
                "analogical_insights": len(analogical_insights),
                "final_conclusion": final_conclusion,
                "confidence": overall_confidence,
                "reasoning_type": "analogical"
            }
            
        except Exception as e:
            return {"error": f"Analogical reasoning failed: {str(e)}"}
    
    def multi_step_problem_solving(self, problem: str, steps: List[str] = None) -> Dict[str, Any]:
        """Solve complex problems through multi-step reasoning."""
        try:
            chain_id = f"multistep_{int(time.time())}"
            reasoning_steps = []
            
            # If no steps provided, decompose the problem
            if not steps:
                steps = self._decompose_problem(problem)
            
            # Execute each step
            current_context = problem
            for i, step_description in enumerate(steps):
                # Determine reasoning type for this step
                reasoning_type = self._determine_reasoning_type(step_description)
                
                # Execute the step
                step_result = self._execute_reasoning_step(step_description, current_context, reasoning_type)
                
                reasoning_step = ReasoningStep(
                    step_id=f"{chain_id}_step_{i}",
                    reasoning_type=reasoning_type,
                    premises=[current_context],
                    conclusion=step_result['conclusion'],
                    confidence=step_result['confidence'],
                    explanation=step_result['explanation'],
                    dependencies=[],
                    timestamp=time.time()
                )
                reasoning_steps.append(reasoning_step)
                
                # Update context for next step
                current_context = step_result['conclusion']
            
            final_conclusion = current_context
            overall_confidence = sum(step.confidence for step in reasoning_steps) / len(reasoning_steps) if reasoning_steps else 0.0
            
            reasoning_chain = ReasoningChain(
                chain_id=chain_id,
                problem=problem,
                steps=reasoning_steps,
                final_conclusion=final_conclusion,
                overall_confidence=overall_confidence,
                reasoning_type=ReasoningType.DEDUCTIVE,  # Default for multi-step
                timestamp=time.time()
            )
            
            self.reasoning_chains.append(reasoning_chain)
            self._update_stats(reasoning_chain)
            
            return {
                "status": "success",
                "chain_id": chain_id,
                "steps_executed": len(reasoning_steps),
                "final_conclusion": final_conclusion,
                "confidence": overall_confidence,
                "reasoning_type": "multi_step"
            }
            
        except Exception as e:
            return {"error": f"Multi-step problem solving failed: {str(e)}"}
    
    def _decompose_problem(self, problem: str) -> List[str]:
        """Decompose a complex problem into simpler steps."""
        # Simple problem decomposition based on keywords
        steps = []
        
        if 'if' in problem.lower() and 'then' in problem.lower():
            steps.append("Identify the conditional relationship")
            steps.append("Check if the antecedent is true")
            steps.append("Apply modus ponens if applicable")
        
        elif 'cause' in problem.lower() or 'effect' in problem.lower():
            steps.append("Identify causal relationships")
            steps.append("Trace causal chains")
            steps.append("Determine root causes")
        
        elif 'similar' in problem.lower() or 'like' in problem.lower():
            steps.append("Identify analogical patterns")
            steps.append("Map source to target domain")
            steps.append("Transfer insights")
        
        else:
            # Generic decomposition
            steps.append("Analyze the problem statement")
            steps.append("Identify key components")
            steps.append("Apply relevant reasoning")
            steps.append("Draw conclusions")
        
        return steps
    
    def _determine_reasoning_type(self, step_description: str) -> ReasoningType:
        """Determine the appropriate reasoning type for a step."""
        step_lower = step_description.lower()
        
        if any(word in step_lower for word in ['if', 'then', 'logical', 'deduce']):
            return ReasoningType.DEDUCTIVE
        elif any(word in step_lower for word in ['cause', 'effect', 'causal']):
            return ReasoningType.CAUSAL
        elif any(word in step_lower for word in ['similar', 'like', 'analogy']):
            return ReasoningType.ANALOGICAL
        elif any(word in step_lower for word in ['generalize', 'pattern', 'induct']):
            return ReasoningType.INDUCTIVE
        else:
            return ReasoningType.DEDUCTIVE  # Default
    
    def _execute_reasoning_step(self, step_description: str, context: str, 
                              reasoning_type: ReasoningType) -> Dict[str, Any]:
        """Execute a single reasoning step."""
        if reasoning_type == ReasoningType.DEDUCTIVE:
            return self._execute_deductive_step(step_description, context)
        elif reasoning_type == ReasoningType.CAUSAL:
            return self._execute_causal_step(step_description, context)
        elif reasoning_type == ReasoningType.ANALOGICAL:
            return self._execute_analogical_step(step_description, context)
        else:
            return {
                'conclusion': f"Executed: {step_description}",
                'confidence': 0.5,
                'explanation': f"Applied {reasoning_type.value} reasoning"
            }
    
    def _execute_deductive_step(self, step_description: str, context: str) -> Dict[str, Any]:
        """Execute a deductive reasoning step."""
        if "modus ponens" in step_description.lower():
            return {
                'conclusion': "Applied modus ponens successfully",
                'confidence': 0.9,
                'explanation': "Used modus ponens rule for deduction"
            }
        else:
            return {
                'conclusion': f"Deductive step: {step_description}",
                'confidence': 0.7,
                'explanation': "Applied deductive reasoning"
            }
    
    def _execute_causal_step(self, step_description: str, context: str) -> Dict[str, Any]:
        """Execute a causal reasoning step."""
        return {
            'conclusion': f"Causal analysis: {step_description}",
            'confidence': 0.6,
            'explanation': "Applied causal reasoning"
        }
    
    def _execute_analogical_step(self, step_description: str, context: str) -> Dict[str, Any]:
        """Execute an analogical reasoning step."""
        return {
            'conclusion': f"Analogical mapping: {step_description}",
            'confidence': 0.5,
            'explanation': "Applied analogical reasoning"
        }
    
    def _update_stats(self, reasoning_chain: ReasoningChain):
        """Update reasoning statistics."""
        self.reasoning_stats['total_chains'] += 1
        if reasoning_chain.overall_confidence > 0.5:
            self.reasoning_stats['successful_chains'] += 1
        
        # Update average confidence
        total_confidence = sum(chain.overall_confidence for chain in self.reasoning_chains)
        self.reasoning_stats['average_confidence'] = total_confidence / len(self.reasoning_chains)
        
        # Update reasoning type usage
        self.reasoning_stats['reasoning_types_used'][reasoning_chain.reasoning_type.value] += 1
    
    def get_reasoning_stats(self) -> Dict[str, Any]:
        """Get comprehensive reasoning statistics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_chains': self.reasoning_stats['total_chains'],
            'successful_chains': self.reasoning_stats['successful_chains'],
            'success_rate': (self.reasoning_stats['successful_chains'] / 
                           self.reasoning_stats['total_chains'] if self.reasoning_stats['total_chains'] > 0 else 0),
            'average_confidence': self.reasoning_stats['average_confidence'],
            'reasoning_types_used': dict(self.reasoning_stats['reasoning_types_used']),
            'logical_rules': len(self.logical_rules),
            'causal_models': len(self.causal_models),
            'analogical_patterns': len(self.analogical_patterns)
        }
    
    def export_reasoning_data(self, filename: str = None) -> str:
        """Export reasoning data to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reasoning_export_{timestamp}.json"
        
        filepath = os.path.join(self.reasoning_dir, filename)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'reasoning_stats': self.get_reasoning_stats(),
            'reasoning_chains': [asdict(chain) for chain in self.reasoning_chains],
            'logical_rules': {k: asdict(v) for k, v in self.logical_rules.items()},
            'causal_models': self.causal_models,
            'analogical_patterns': self.analogical_patterns
        }
        
        # Convert enum values to strings
        for chain in export_data['reasoning_chains']:
            chain['reasoning_type'] = chain['reasoning_type'].value
            for step in chain['steps']:
                step['reasoning_type'] = step['reasoning_type'].value
        
        for rule in export_data['logical_rules'].values():
            rule['rule_type'] = rule['rule_type']
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath 