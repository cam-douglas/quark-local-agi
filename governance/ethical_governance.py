#!/usr/bin/env python3
"""
Ethical Governance System for Quark AI Assistant
===================================================

Implements ethical decision making, value alignment, transparency, accountability,
bias detection, fairness, and regulatory compliance capabilities.

Part of Pillar 21: Governance & Ethics
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


class EthicalPrinciple(Enum):
    """Core ethical principles for AI governance."""
    BENEFICENCE = "beneficence"          # Do good, promote well-being
    NON_MALEFICENCE = "non_maleficence"  # Do no harm
    AUTONOMY = "autonomy"                # Respect human autonomy
    JUSTICE = "justice"                  # Ensure fairness and equity
    TRANSPARENCY = "transparency"        # Be open and explainable
    ACCOUNTABILITY = "accountability"     # Take responsibility for actions
    PRIVACY = "privacy"                  # Protect personal information
    SUSTAINABILITY = "sustainability"    # Consider long-term impacts


class ValueType(Enum):
    """Types of values for alignment assessment."""
    HUMAN_RIGHTS = "human_rights"        # Fundamental human rights
    DEMOCRATIC_VALUES = "democratic_values" # Democratic principles
    CULTURAL_VALUES = "cultural_values"  # Cultural and social values
    PROFESSIONAL_VALUES = "professional_values" # Professional ethics
    PERSONAL_VALUES = "personal_values"  # Individual preferences
    SOCIETAL_VALUES = "societal_values"  # Broader societal values


class BiasType(Enum):
    """Types of bias that can be detected and mitigated."""
    STATISTICAL_BIAS = "statistical_bias"     # Data distribution bias
    REPRESENTATION_BIAS = "representation_bias" # Under/over-representation
    ALGORITHMIC_BIAS = "algorithmic_bias"     # Algorithm-induced bias
    COGNITIVE_BIAS = "cognitive_bias"         # Human cognitive bias
    CONFIRMATION_BIAS = "confirmation_bias"   # Confirmation of existing beliefs
    ANCHORING_BIAS = "anchoring_bias"         # Over-reliance on initial info


class FairnessMetric(Enum):
    """Metrics for measuring fairness."""
    EQUAL_OPPORTUNITY = "equal_opportunity"   # Equal chance of positive outcome
    EQUALIZED_ODDS = "equalized_odds"         # Equal true/false positive rates
    DEMOGRAPHIC_PARITY = "demographic_parity" # Equal positive prediction rates
    INDIVIDUAL_FAIRNESS = "individual_fairness" # Similar individuals treated similarly


class TransparencyLevel(Enum):
    """Levels of transparency and explainability."""
    FULL_TRANSPARENCY = "full_transparency"   # Complete disclosure
    PARTIAL_TRANSPARENCY = "partial_transparency" # Limited disclosure
    BLACK_BOX = "black_box"                   # No explanation
    EXPLAINABLE = "explainable"               # Post-hoc explanations


@dataclass
class EthicalDecision:
    """Represents an ethical decision with reasoning."""
    decision_id: str
    context: str
    options: List[str]
    selected_option: str
    ethical_principles: List[EthicalPrinciple]
    reasoning: str
    confidence: float  # 0.0 to 1.0
    impact_assessment: Dict[str, Any]
    timestamp: float


@dataclass
class ValueAlignment:
    """Represents value alignment assessment."""
    alignment_id: str
    value_type: ValueType
    human_values: List[str]
    ai_behavior: str
    alignment_score: float  # 0.0 to 1.0
    misalignment_areas: List[str]
    recommendations: List[str]
    timestamp: float


@dataclass
class BiasAssessment:
    """Represents bias detection and assessment."""
    assessment_id: str
    bias_type: BiasType
    affected_groups: List[str]
    bias_indicators: List[str]
    severity: float  # 0.0 to 1.0
    mitigation_strategies: List[str]
    timestamp: float


@dataclass
class FairnessReport:
    """Represents fairness analysis and recommendations."""
    report_id: str
    fairness_metrics: List[FairnessMetric]
    group_performance: Dict[str, float]
    fairness_scores: Dict[str, float]
    recommendations: List[str]
    timestamp: float


@dataclass
class AccountabilityRecord:
    """Represents accountability tracking."""
    record_id: str
    action: str
    agent_id: str
    decision_factors: List[str]
    ethical_considerations: List[str]
    outcome: str
    responsibility_assigned: str
    timestamp: float


class EthicalGovernance:
    """Advanced ethical governance system with comprehensive oversight."""
    
    def __init__(self, governance_dir: str = None):
        self.governance_dir = governance_dir or os.path.join(os.path.dirname(__file__), '..', 'governance_data')
        os.makedirs(self.governance_dir, exist_ok=True)
        
        # Ethical decision tracking
        self.ethical_decisions = {}
        self.decision_history = []
        
        # Value alignment tracking
        self.value_alignments = {}
        self.alignment_history = []
        
        # Bias detection
        self.bias_assessments = {}
        self.bias_history = []
        
        # Fairness monitoring
        self.fairness_reports = {}
        self.fairness_history = []
        
        # Accountability tracking
        self.accountability_records = {}
        self.accountability_history = []
        
        # Governance settings
        self.ethical_principles_enabled = True
        self.value_alignment_enabled = True
        self.bias_detection_enabled = True
        self.fairness_monitoring_enabled = True
        self.transparency_level = TransparencyLevel.EXPLAINABLE
        
        # Governance tracking
        self.governance_stats = {
            'ethical_decisions': 0,
            'value_alignments': 0,
            'bias_assessments': 0,
            'fairness_reports': 0,
            'accountability_records': 0
        }
        
        # Load governance components
        self._load_ethical_frameworks()
        self._load_value_frameworks()
        self._load_bias_patterns()
        self._load_fairness_metrics()
    
    def _load_ethical_frameworks(self):
        """Load ethical decision-making frameworks."""
        self.ethical_frameworks = {
            EthicalPrinciple.BENEFICENCE: {
                'description': 'Promote well-being and positive outcomes',
                'questions': ['Will this action benefit people?', 'How can we maximize positive impact?'],
                'weight': 0.25
            },
            EthicalPrinciple.NON_MALEFICENCE: {
                'description': 'Avoid causing harm',
                'questions': ['Could this action cause harm?', 'How can we minimize risks?'],
                'weight': 0.25
            },
            EthicalPrinciple.AUTONOMY: {
                'description': 'Respect human autonomy and choice',
                'questions': ['Does this respect human autonomy?', 'Are we preserving human agency?'],
                'weight': 0.2
            },
            EthicalPrinciple.JUSTICE: {
                'description': 'Ensure fairness and equity',
                'questions': ['Is this fair to all affected parties?', 'Are we treating people equally?'],
                'weight': 0.2
            },
            EthicalPrinciple.TRANSPARENCY: {
                'description': 'Be open and explainable',
                'questions': ['Can we explain this decision?', 'Is the process transparent?'],
                'weight': 0.1
            },
            EthicalPrinciple.ACCOUNTABILITY: {
                'description': 'Take responsibility for actions',
                'questions': ['Who is responsible for this?', 'How can we track accountability?'],
                'weight': 0.1
            },
            EthicalPrinciple.PRIVACY: {
                'description': 'Protect personal information',
                'questions': ['Are we protecting privacy?', 'Is data being handled securely?'],
                'weight': 0.1
            },
            EthicalPrinciple.SUSTAINABILITY: {
                'description': 'Consider long-term impacts',
                'questions': ['What are the long-term consequences?', 'Is this sustainable?'],
                'weight': 0.1
            }
        }
    
    def _load_value_frameworks(self):
        """Load value alignment frameworks."""
        self.value_frameworks = {
            ValueType.HUMAN_RIGHTS: {
                'description': 'Fundamental human rights and dignity',
                'values': ['dignity', 'equality', 'freedom', 'privacy', 'safety'],
                'weight': 0.3
            },
            ValueType.DEMOCRATIC_VALUES: {
                'description': 'Democratic principles and participation',
                'values': ['participation', 'representation', 'accountability', 'transparency'],
                'weight': 0.25
            },
            ValueType.CULTURAL_VALUES: {
                'description': 'Cultural and social values',
                'values': ['respect', 'inclusion', 'diversity', 'tolerance'],
                'weight': 0.2
            },
            ValueType.PROFESSIONAL_VALUES: {
                'description': 'Professional ethics and standards',
                'values': ['competence', 'integrity', 'responsibility', 'excellence'],
                'weight': 0.15
            },
            ValueType.PERSONAL_VALUES: {
                'description': 'Individual preferences and autonomy',
                'values': ['choice', 'autonomy', 'privacy', 'self-determination'],
                'weight': 0.1
            }
        }
    
    def _load_bias_patterns(self):
        """Load bias detection patterns."""
        self.bias_patterns = {
            BiasType.STATISTICAL_BIAS: {
                'indicators': ['skewed_distributions', 'missing_data', 'sampling_bias'],
                'severity_threshold': 0.3
            },
            BiasType.REPRESENTATION_BIAS: {
                'indicators': ['underrepresentation', 'overrepresentation', 'stereotyping'],
                'severity_threshold': 0.4
            },
            BiasType.ALGORITHMIC_BIAS: {
                'indicators': ['discriminatory_patterns', 'unfair_advantage', 'biased_metrics'],
                'severity_threshold': 0.5
            },
            BiasType.COGNITIVE_BIAS: {
                'indicators': ['confirmation_bias', 'anchoring', 'availability_heuristic'],
                'severity_threshold': 0.3
            }
        }
    
    def _load_fairness_metrics(self):
        """Load fairness measurement metrics."""
        self.fairness_metrics = {
            FairnessMetric.EQUAL_OPPORTUNITY: {
                'description': 'Equal chance of positive outcome',
                'calculation': 'positive_rate_equality',
                'threshold': 0.1
            },
            FairnessMetric.EQUALIZED_ODDS: {
                'description': 'Equal true/false positive rates',
                'calculation': 'odds_equality',
                'threshold': 0.1
            },
            FairnessMetric.DEMOGRAPHIC_PARITY: {
                'description': 'Equal positive prediction rates',
                'calculation': 'prediction_rate_equality',
                'threshold': 0.1
            },
            FairnessMetric.INDIVIDUAL_FAIRNESS: {
                'description': 'Similar individuals treated similarly',
                'calculation': 'individual_similarity',
                'threshold': 0.2
            }
        }
    
    def make_ethical_decision(self, context: str, options: List[str], 
                             principles: List[EthicalPrinciple] = None) -> Dict[str, Any]:
        """Make an ethical decision based on principles and context."""
        try:
            decision_id = f"ethical_decision_{int(time.time())}"
            
            # Determine applicable principles if not provided
            if not principles:
                principles = self._determine_applicable_principles(context)
            
            # Evaluate options against ethical principles
            option_scores = {}
            for option in options:
                score = self._evaluate_option_ethics(option, principles, context)
                option_scores[option] = score
            
            # Select best option
            best_option = max(option_scores.items(), key=lambda x: x[1])
            
            # Generate reasoning
            reasoning = self._generate_ethical_reasoning(best_option[0], principles, context)
            
            # Assess impact
            impact_assessment = self._assess_ethical_impact(best_option[0], context)
            
            # Create ethical decision
            decision = EthicalDecision(
                decision_id=decision_id,
                context=context,
                options=options,
                selected_option=best_option[0],
                ethical_principles=principles,
                reasoning=reasoning,
                confidence=best_option[1],
                impact_assessment=impact_assessment,
                timestamp=time.time()
            )
            
            # Store decision
            self.ethical_decisions[decision_id] = decision
            self.decision_history.append(decision)
            
            # Update stats
            self.governance_stats['ethical_decisions'] += 1
            
            return {
                "status": "success",
                "decision_id": decision_id,
                "selected_option": best_option[0],
                "confidence": best_option[1],
                "reasoning": reasoning,
                "principles_used": [p.value for p in principles],
                "impact_assessment": impact_assessment
            }
            
        except Exception as e:
            return {"error": f"Ethical decision failed: {str(e)}"}
    
    def _determine_applicable_principles(self, context: str) -> List[EthicalPrinciple]:
        """Determine which ethical principles apply to the context."""
        context_lower = context.lower()
        applicable_principles = []
        
        # Check for principle indicators
        if any(word in context_lower for word in ['help', 'benefit', 'improve', 'good']):
            applicable_principles.append(EthicalPrinciple.BENEFICENCE)
        
        if any(word in context_lower for word in ['harm', 'risk', 'danger', 'damage']):
            applicable_principles.append(EthicalPrinciple.NON_MALEFICENCE)
        
        if any(word in context_lower for word in ['choice', 'autonomy', 'freedom', 'control']):
            applicable_principles.append(EthicalPrinciple.AUTONOMY)
        
        if any(word in context_lower for word in ['fair', 'equal', 'justice', 'equity']):
            applicable_principles.append(EthicalPrinciple.JUSTICE)
        
        if any(word in context_lower for word in ['explain', 'transparent', 'clear', 'understand']):
            applicable_principles.append(EthicalPrinciple.TRANSPARENCY)
        
        if any(word in context_lower for word in ['responsible', 'accountable', 'blame']):
            applicable_principles.append(EthicalPrinciple.ACCOUNTABILITY)
        
        if any(word in context_lower for word in ['privacy', 'personal', 'confidential']):
            applicable_principles.append(EthicalPrinciple.PRIVACY)
        
        if any(word in context_lower for word in ['future', 'long-term', 'sustainable']):
            applicable_principles.append(EthicalPrinciple.SUSTAINABILITY)
        
        # Default to core principles if none identified
        if not applicable_principles:
            applicable_principles = [
                EthicalPrinciple.BENEFICENCE,
                EthicalPrinciple.NON_MALEFICENCE,
                EthicalPrinciple.JUSTICE
            ]
        
        return applicable_principles
    
    def _evaluate_option_ethics(self, option: str, principles: List[EthicalPrinciple], 
                               context: str) -> float:
        """Evaluate an option against ethical principles."""
        total_score = 0.0
        total_weight = 0.0
        
        for principle in principles:
            framework = self.ethical_frameworks[principle]
            weight = framework['weight']
            
            # Simple evaluation based on keywords
            option_lower = option.lower()
            score = 0.5  # Base score
            
            if principle == EthicalPrinciple.BENEFICENCE:
                if any(word in option_lower for word in ['help', 'benefit', 'improve', 'good']):
                    score += 0.3
            elif principle == EthicalPrinciple.NON_MALEFICENCE:
                if any(word in option_lower for word in ['safe', 'protect', 'prevent', 'avoid']):
                    score += 0.3
            elif principle == EthicalPrinciple.AUTONOMY:
                if any(word in option_lower for word in ['choice', 'freedom', 'control', 'independent']):
                    score += 0.3
            elif principle == EthicalPrinciple.JUSTICE:
                if any(word in option_lower for word in ['fair', 'equal', 'just', 'equitable']):
                    score += 0.3
            elif principle == EthicalPrinciple.TRANSPARENCY:
                if any(word in option_lower for word in ['explain', 'clear', 'open', 'transparent']):
                    score += 0.3
            elif principle == EthicalPrinciple.ACCOUNTABILITY:
                if any(word in option_lower for word in ['responsible', 'accountable', 'track']):
                    score += 0.3
            elif principle == EthicalPrinciple.PRIVACY:
                if any(word in option_lower for word in ['privacy', 'secure', 'confidential']):
                    score += 0.3
            elif principle == EthicalPrinciple.SUSTAINABILITY:
                if any(word in option_lower for word in ['sustainable', 'future', 'long-term']):
                    score += 0.3
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _generate_ethical_reasoning(self, option: str, principles: List[EthicalPrinciple], 
                                   context: str) -> str:
        """Generate ethical reasoning for the selected option."""
        reasoning_parts = []
        
        for principle in principles:
            framework = self.ethical_frameworks[principle]
            reasoning_parts.append(f"Respects {principle.value}: {framework['description']}")
        
        return f"Selected option '{option}' because it: {'; '.join(reasoning_parts)}"
    
    def _assess_ethical_impact(self, option: str, context: str) -> Dict[str, Any]:
        """Assess the ethical impact of an option."""
        return {
            'positive_impacts': ['Improved decision quality', 'Enhanced transparency'],
            'negative_impacts': ['Potential privacy concerns', 'Resource requirements'],
            'stakeholders_affected': ['Users', 'System operators', 'Society'],
            'risk_level': 'medium',
            'mitigation_strategies': ['Regular monitoring', 'Stakeholder feedback']
        }
    
    def assess_value_alignment(self, value_type: ValueType, human_values: List[str], 
                              ai_behavior: str) -> Dict[str, Any]:
        """Assess alignment between human values and AI behavior."""
        try:
            alignment_id = f"alignment_{int(time.time())}"
            
            # Calculate alignment score
            alignment_score = self._calculate_alignment_score(value_type, human_values, ai_behavior)
            
            # Identify misalignment areas
            misalignment_areas = self._identify_misalignments(value_type, human_values, ai_behavior)
            
            # Generate recommendations
            recommendations = self._generate_alignment_recommendations(value_type, misalignment_areas)
            
            # Create value alignment record
            alignment = ValueAlignment(
                alignment_id=alignment_id,
                value_type=value_type,
                human_values=human_values,
                ai_behavior=ai_behavior,
                alignment_score=alignment_score,
                misalignment_areas=misalignment_areas,
                recommendations=recommendations,
                timestamp=time.time()
            )
            
            # Store alignment
            self.value_alignments[alignment_id] = alignment
            self.alignment_history.append(alignment)
            
            # Update stats
            self.governance_stats['value_alignments'] += 1
            
            return {
                "status": "success",
                "alignment_id": alignment_id,
                "alignment_score": alignment_score,
                "misalignment_areas": misalignment_areas,
                "recommendations": recommendations
            }
            
        except Exception as e:
            return {"error": f"Value alignment assessment failed: {str(e)}"}
    
    def _calculate_alignment_score(self, value_type: ValueType, human_values: List[str], 
                                  ai_behavior: str) -> float:
        """Calculate alignment score between human values and AI behavior."""
        framework = self.value_frameworks[value_type]
        framework_values = framework['values']
        
        # Count matching values
        matches = 0
        for human_value in human_values:
            if any(framework_value in human_value.lower() for framework_value in framework_values):
                matches += 1
        
        # Calculate score
        if human_values:
            return min(1.0, matches / len(human_values))
        else:
            return 0.5
    
    def _identify_misalignments(self, value_type: ValueType, human_values: List[str], 
                                ai_behavior: str) -> List[str]:
        """Identify areas of misalignment."""
        misalignments = []
        
        # Simple misalignment detection
        if value_type == ValueType.HUMAN_RIGHTS:
            if 'privacy' in human_values and 'data_collection' in ai_behavior.lower():
                misalignments.append("Privacy concerns with data collection")
        
        if value_type == ValueType.DEMOCRATIC_VALUES:
            if 'transparency' in human_values and 'black_box' in ai_behavior.lower():
                misalignments.append("Lack of transparency in decision making")
        
        return misalignments
    
    def _generate_alignment_recommendations(self, value_type: ValueType, 
                                          misalignments: List[str]) -> List[str]:
        """Generate recommendations for improving value alignment."""
        recommendations = []
        
        for misalignment in misalignments:
            if 'privacy' in misalignment.lower():
                recommendations.append("Implement stronger privacy protections")
            elif 'transparency' in misalignment.lower():
                recommendations.append("Increase transparency in decision making")
            else:
                recommendations.append("Review and adjust AI behavior to better align with values")
        
        return recommendations
    
    def detect_bias(self, data_description: str, affected_groups: List[str] = None) -> Dict[str, Any]:
        """Detect bias in data or algorithms."""
        try:
            assessment_id = f"bias_assessment_{int(time.time())}"
            
            # Determine bias types
            bias_types = self._identify_bias_types(data_description)
            
            # Assess bias severity
            severity = self._assess_bias_severity(data_description, bias_types)
            
            # Generate mitigation strategies
            mitigation_strategies = self._generate_bias_mitigation(bias_types)
            
            # Create bias assessment
            assessment = BiasAssessment(
                assessment_id=assessment_id,
                bias_type=bias_types[0] if bias_types else BiasType.STATISTICAL_BIAS,
                affected_groups=affected_groups or ['general_population'],
                bias_indicators=self._identify_bias_indicators(data_description),
                severity=severity,
                mitigation_strategies=mitigation_strategies,
                timestamp=time.time()
            )
            
            # Store assessment
            self.bias_assessments[assessment_id] = assessment
            self.bias_history.append(assessment)
            
            # Update stats
            self.governance_stats['bias_assessments'] += 1
            
            return {
                "status": "success",
                "assessment_id": assessment_id,
                "bias_types": [b.value for b in bias_types],
                "severity": severity,
                "mitigation_strategies": mitigation_strategies
            }
            
        except Exception as e:
            return {"error": f"Bias detection failed: {str(e)}"}
    
    def _identify_bias_types(self, data_description: str) -> List[BiasType]:
        """Identify types of bias present in the data."""
        bias_types = []
        description_lower = data_description.lower()
        
        for bias_type, pattern in self.bias_patterns.items():
            if any(indicator in description_lower for indicator in pattern['indicators']):
                bias_types.append(bias_type)
        
        return bias_types
    
    def _assess_bias_severity(self, data_description: str, bias_types: List[BiasType]) -> float:
        """Assess the severity of detected bias."""
        if not bias_types:
            return 0.0
        
        # Simple severity calculation
        total_severity = 0.0
        for bias_type in bias_types:
            threshold = self.bias_patterns[bias_type]['severity_threshold']
            total_severity += threshold
        
        return min(1.0, total_severity / len(bias_types))
    
    def _identify_bias_indicators(self, data_description: str) -> List[str]:
        """Identify specific bias indicators in the data."""
        indicators = []
        description_lower = data_description.lower()
        
        if 'skewed' in description_lower or 'imbalanced' in description_lower:
            indicators.append("Skewed data distribution")
        
        if 'missing' in description_lower or 'incomplete' in description_lower:
            indicators.append("Missing or incomplete data")
        
        if 'stereotype' in description_lower or 'bias' in description_lower:
            indicators.append("Stereotypical patterns detected")
        
        return indicators
    
    def _generate_bias_mitigation(self, bias_types: List[BiasType]) -> List[str]:
        """Generate strategies for mitigating detected bias."""
        strategies = []
        
        for bias_type in bias_types:
            if bias_type == BiasType.STATISTICAL_BIAS:
                strategies.append("Collect more representative data")
            elif bias_type == BiasType.REPRESENTATION_BIAS:
                strategies.append("Ensure diverse representation in training data")
            elif bias_type == BiasType.ALGORITHMIC_BIAS:
                strategies.append("Implement fairness-aware algorithms")
            elif bias_type == BiasType.COGNITIVE_BIAS:
                strategies.append("Use objective evaluation metrics")
        
        return strategies
    
    def assess_fairness(self, group_performance: Dict[str, float], 
                       fairness_metrics: List[FairnessMetric] = None) -> Dict[str, Any]:
        """Assess fairness across different groups."""
        try:
            report_id = f"fairness_report_{int(time.time())}"
            
            # Determine metrics to use
            if not fairness_metrics:
                fairness_metrics = list(FairnessMetric)
            
            # Calculate fairness scores
            fairness_scores = {}
            for metric in fairness_metrics:
                score = self._calculate_fairness_score(group_performance, metric)
                fairness_scores[metric.value] = score
            
            # Generate recommendations
            recommendations = self._generate_fairness_recommendations(fairness_scores)
            
            # Create fairness report
            report = FairnessReport(
                report_id=report_id,
                fairness_metrics=fairness_metrics,
                group_performance=group_performance,
                fairness_scores=fairness_scores,
                recommendations=recommendations,
                timestamp=time.time()
            )
            
            # Store report
            self.fairness_reports[report_id] = report
            self.fairness_history.append(report)
            
            # Update stats
            self.governance_stats['fairness_reports'] += 1
            
            return {
                "status": "success",
                "report_id": report_id,
                "fairness_scores": fairness_scores,
                "recommendations": recommendations
            }
            
        except Exception as e:
            return {"error": f"Fairness assessment failed: {str(e)}"}
    
    def _calculate_fairness_score(self, group_performance: Dict[str, float], 
                                 metric: FairnessMetric) -> float:
        """Calculate fairness score for a specific metric."""
        if not group_performance:
            return 0.0
        
        performances = list(group_performance.values())
        
        if metric == FairnessMetric.EQUAL_OPPORTUNITY:
            # Calculate variance in performance
            mean_performance = sum(performances) / len(performances)
            variance = sum((p - mean_performance) ** 2 for p in performances) / len(performances)
            return max(0.0, 1.0 - variance)
        
        elif metric == FairnessMetric.DEMOGRAPHIC_PARITY:
            # Check if all groups have similar performance
            min_performance = min(performances)
            max_performance = max(performances)
            if max_performance > 0:
                return min_performance / max_performance
            else:
                return 1.0
        
        else:
            # Default fairness calculation
            return sum(performances) / len(performances)
    
    def _generate_fairness_recommendations(self, fairness_scores: Dict[str, float]) -> List[str]:
        """Generate recommendations for improving fairness."""
        recommendations = []
        
        for metric, score in fairness_scores.items():
            if score < 0.7:
                if metric == 'equal_opportunity':
                    recommendations.append("Implement equal opportunity measures")
                elif metric == 'demographic_parity':
                    recommendations.append("Ensure demographic parity in outcomes")
                else:
                    recommendations.append(f"Improve {metric} fairness")
        
        return recommendations
    
    def record_accountability(self, action: str, agent_id: str, decision_factors: List[str],
                            ethical_considerations: List[str], outcome: str) -> Dict[str, Any]:
        """Record accountability for AI actions."""
        try:
            record_id = f"accountability_{int(time.time())}"
            
            # Determine responsibility
            responsibility_assigned = self._assign_responsibility(agent_id, action)
            
            # Create accountability record
            record = AccountabilityRecord(
                record_id=record_id,
                action=action,
                agent_id=agent_id,
                decision_factors=decision_factors,
                ethical_considerations=ethical_considerations,
                outcome=outcome,
                responsibility_assigned=responsibility_assigned,
                timestamp=time.time()
            )
            
            # Store record
            self.accountability_records[record_id] = record
            self.accountability_history.append(record)
            
            # Update stats
            self.governance_stats['accountability_records'] += 1
            
            return {
                "status": "success",
                "record_id": record_id,
                "responsibility_assigned": responsibility_assigned,
                "timestamp": record.timestamp
            }
            
        except Exception as e:
            return {"error": f"Accountability recording failed: {str(e)}"}
    
    def _assign_responsibility(self, agent_id: str, action: str) -> str:
        """Assign responsibility for an action."""
        if 'safety' in action.lower():
            return f"{agent_id} (Safety Agent)"
        elif 'ethical' in action.lower():
            return f"{agent_id} (Ethics Committee)"
        elif 'decision' in action.lower():
            return f"{agent_id} (Decision Maker)"
        else:
            return f"{agent_id} (General Agent)"
    
    def get_governance_stats(self) -> Dict[str, Any]:
        """Get comprehensive governance statistics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'ethical_decisions': self.governance_stats['ethical_decisions'],
            'value_alignments': self.governance_stats['value_alignments'],
            'bias_assessments': self.governance_stats['bias_assessments'],
            'fairness_reports': self.governance_stats['fairness_reports'],
            'accountability_records': self.governance_stats['accountability_records'],
            'total_governance_activities': sum(self.governance_stats.values())
        }
    
    def export_governance_data(self, filename: str = None) -> str:
        """Export governance data to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"governance_export_{timestamp}.json"
        
        filepath = os.path.join(self.governance_dir, filename)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'governance_stats': self.get_governance_stats(),
            'ethical_decisions': {k: asdict(v) for k, v in self.ethical_decisions.items()},
            'value_alignments': {k: asdict(v) for k, v in self.value_alignments.items()},
            'bias_assessments': {k: asdict(v) for k, v in self.bias_assessments.items()},
            'fairness_reports': {k: asdict(v) for k, v in self.fairness_reports.items()},
            'accountability_records': {k: asdict(v) for k, v in self.accountability_records.items()}
        }
        
        # Convert enum values to strings
        for decision in export_data['ethical_decisions'].values():
            decision['ethical_principles'] = [p.value for p in decision['ethical_principles']]
        
        for alignment in export_data['value_alignments'].values():
            alignment['value_type'] = alignment['value_type'].value
        
        for assessment in export_data['bias_assessments'].values():
            assessment['bias_type'] = assessment['bias_type'].value
        
        for report in export_data['fairness_reports'].values():
            report['fairness_metrics'] = [m.value for m in report['fairness_metrics']]
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath 