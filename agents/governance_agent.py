#!/usr/bin/env python3
"""
Governance Agent for Quark AI Assistant
==========================================

Manages ethical decision making, value alignment, transparency, accountability,
bias detection, fairness, and regulatory compliance.

Part of Pillar 21: Governance & Ethics
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import asdict

from agents.base import Agent
from governance.ethical_governance import (
    EthicalGovernance, EthicalPrinciple, ValueType, BiasType, FairnessMetric,
    EthicalDecision, ValueAlignment, BiasAssessment, FairnessReport, AccountabilityRecord
)


class GovernanceAgent(Agent):
    """Advanced governance agent for ethical oversight and compliance."""
    
    def __init__(self, model_name: str = "governance_agent", governance_dir: str = None):
        super().__init__(model_name)
        self.governance_dir = governance_dir or os.path.join(os.path.dirname(__file__), '..', 'governance_data')
        os.makedirs(self.governance_dir, exist_ok=True)
        
        # Initialize ethical governance system
        self.governance_engine = EthicalGovernance(self.governance_dir)
        
        # Governance operation settings
        self.ethical_principles_enabled = True
        self.value_alignment_enabled = True
        self.bias_detection_enabled = True
        self.fairness_monitoring_enabled = True
        self.transparency_level = "explainable"
        
        # Governance tracking
        self.governance_operations = []
        self.governance_performance = {
            'total_operations': 0,
            'ethical_decisions': 0,
            'value_alignments': 0,
            'bias_assessments': 0,
            'fairness_reports': 0,
            'accountability_records': 0
        }
    
    def load_model(self):
        """Load governance models and components."""
        try:
            # Initialize ethical governance system
            self.governance_engine = EthicalGovernance(self.governance_dir)
            
            return True
        except Exception as e:
            print(f"Error loading governance models: {e}")
            return False
    
    def generate(self, input_data: str, operation: str = "make_ethical_decision", **kwargs) -> Dict[str, Any]:
        """
        Generate governance operations or perform governance management.
        
        Args:
            input_data: Context or governance request
            operation: Governance operation to perform
            **kwargs: Additional parameters
            
        Returns:
            Governance operation result
        """
        try:
            if operation == "make_ethical_decision":
                return self._make_ethical_decision(input_data, **kwargs)
            elif operation == "assess_value_alignment":
                return self._assess_value_alignment(input_data, **kwargs)
            elif operation == "detect_bias":
                return self._detect_bias(input_data, **kwargs)
            elif operation == "assess_fairness":
                return self._assess_fairness(input_data, **kwargs)
            elif operation == "record_accountability":
                return self._record_accountability(input_data, **kwargs)
            elif operation == "get_governance_stats":
                return self._get_governance_stats()
            elif operation == "export_governance_data":
                return self._export_governance_data(**kwargs)
            elif operation == "analyze_ethics":
                return self._analyze_ethics(input_data, **kwargs)
            elif operation == "compliance_check":
                return self._compliance_check(input_data, **kwargs)
            else:
                return {"error": f"Unknown governance operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Governance operation failed: {str(e)}"}
    
    def _make_ethical_decision(self, decision_data: str, options: List[str] = None,
                              principles: List[str] = None) -> Dict[str, Any]:
        """Make an ethical decision based on context and options."""
        try:
            # Parse decision data
            if isinstance(decision_data, str):
                context = decision_data
                if not options:
                    options = self._generate_decision_options(context)
            else:
                context = decision_data.get('context', 'general decision')
                options = decision_data.get('options', [])
                principles = decision_data.get('principles', [])
            
            # Convert principles to enums if provided
            principle_enums = None
            if principles:
                try:
                    principle_enums = [EthicalPrinciple(p) for p in principles]
                except ValueError as e:
                    return {"error": f"Invalid ethical principle: {e}"}
            
            # Make ethical decision
            result = self.governance_engine.make_ethical_decision(context, options, principle_enums)
            
            # Track operation
            self.governance_operations.append({
                'operation': 'make_ethical_decision',
                'context': context,
                'options_count': len(options),
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.governance_performance['ethical_decisions'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to make ethical decision: {str(e)}"}
    
    def _generate_decision_options(self, context: str) -> List[str]:
        """Generate decision options based on context."""
        context_lower = context.lower()
        
        if any(word in context_lower for word in ['safety', 'security']):
            return ['Implement safety measures', 'Conduct risk assessment', 'Review protocols']
        elif any(word in context_lower for word in ['privacy', 'data']):
            return ['Enhance privacy protections', 'Limit data collection', 'Implement encryption']
        elif any(word in context_lower for word in ['fairness', 'equity']):
            return ['Ensure equal treatment', 'Implement bias detection', 'Review algorithms']
        elif any(word in context_lower for word in ['transparency', 'explain']):
            return ['Provide explanations', 'Increase transparency', 'Document decisions']
        else:
            return ['Proceed with caution', 'Take calculated risk', 'Maintain current approach']
    
    def _assess_value_alignment(self, alignment_data: str, value_type: str = None,
                               human_values: List[str] = None, ai_behavior: str = None) -> Dict[str, Any]:
        """Assess alignment between human values and AI behavior."""
        try:
            # Parse alignment data
            if isinstance(alignment_data, str):
                # Extract values from text
                if not human_values:
                    human_values = self._extract_human_values(alignment_data)
                if not ai_behavior:
                    ai_behavior = alignment_data
                if not value_type:
                    value_type = 'human_rights'
            else:
                value_type = alignment_data.get('value_type', 'human_rights')
                human_values = alignment_data.get('human_values', [])
                ai_behavior = alignment_data.get('ai_behavior', '')
            
            # Convert value type to enum
            try:
                value_type_enum = ValueType(value_type)
            except ValueError:
                return {"error": f"Invalid value type: {value_type}"}
            
            # Assess value alignment
            result = self.governance_engine.assess_value_alignment(value_type_enum, human_values, ai_behavior)
            
            # Track operation
            self.governance_operations.append({
                'operation': 'assess_value_alignment',
                'value_type': value_type,
                'human_values_count': len(human_values),
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.governance_performance['value_alignments'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to assess value alignment: {str(e)}"}
    
    def _extract_human_values(self, alignment_data: str) -> List[str]:
        """Extract human values from text."""
        values = []
        data_lower = alignment_data.lower()
        
        if any(word in data_lower for word in ['privacy', 'confidential']):
            values.append('privacy')
        if any(word in data_lower for word in ['fair', 'equal', 'just']):
            values.append('fairness')
        if any(word in data_lower for word in ['transparent', 'open', 'clear']):
            values.append('transparency')
        if any(word in data_lower for word in ['safe', 'secure', 'protect']):
            values.append('safety')
        if any(word in data_lower for word in ['autonomy', 'freedom', 'choice']):
            values.append('autonomy')
        
        return values if values else ['privacy', 'fairness']
    
    def _detect_bias(self, bias_data: str, affected_groups: List[str] = None) -> Dict[str, Any]:
        """Detect bias in data or algorithms."""
        try:
            # Parse bias data
            if isinstance(bias_data, str):
                data_description = bias_data
            else:
                data_description = bias_data.get('data_description', '')
                affected_groups = bias_data.get('affected_groups', [])
            
            # Detect bias
            result = self.governance_engine.detect_bias(data_description, affected_groups)
            
            # Track operation
            self.governance_operations.append({
                'operation': 'detect_bias',
                'data_description': data_description[:100] + '...' if len(data_description) > 100 else data_description,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.governance_performance['bias_assessments'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to detect bias: {str(e)}"}
    
    def _assess_fairness(self, fairness_data: str, group_performance: Dict[str, float] = None,
                         fairness_metrics: List[str] = None) -> Dict[str, Any]:
        """Assess fairness across different groups."""
        try:
            # Parse fairness data
            if isinstance(fairness_data, str):
                # Generate sample group performance
                if not group_performance:
                    group_performance = self._generate_sample_performance()
                if not fairness_metrics:
                    fairness_metrics = ['equal_opportunity', 'demographic_parity']
            else:
                group_performance = fairness_data.get('group_performance', {})
                fairness_metrics = fairness_data.get('fairness_metrics', [])
            
            # Convert metrics to enums
            metric_enums = []
            for metric in fairness_metrics:
                try:
                    metric_enums.append(FairnessMetric(metric))
                except ValueError:
                    return {"error": f"Invalid fairness metric: {metric}"}
            
            # Assess fairness
            result = self.governance_engine.assess_fairness(group_performance, metric_enums)
            
            # Track operation
            self.governance_operations.append({
                'operation': 'assess_fairness',
                'groups_count': len(group_performance),
                'metrics_count': len(fairness_metrics),
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.governance_performance['fairness_reports'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to assess fairness: {str(e)}"}
    
    def _generate_sample_performance(self) -> Dict[str, float]:
        """Generate sample group performance data."""
        return {
            'group_a': 0.8,
            'group_b': 0.7,
            'group_c': 0.9,
            'group_d': 0.6
        }
    
    def _record_accountability(self, accountability_data: str, action: str = None,
                              agent_id: str = None, decision_factors: List[str] = None,
                              ethical_considerations: List[str] = None, outcome: str = None) -> Dict[str, Any]:
        """Record accountability for AI actions."""
        try:
            # Parse accountability data
            if isinstance(accountability_data, str):
                # Extract information from text
                if not action:
                    action = self._extract_action(accountability_data)
                if not agent_id:
                    agent_id = 'governance_agent'
                if not decision_factors:
                    decision_factors = self._extract_decision_factors(accountability_data)
                if not ethical_considerations:
                    ethical_considerations = self._extract_ethical_considerations(accountability_data)
                if not outcome:
                    outcome = 'action_completed'
            else:
                action = accountability_data.get('action', 'general_action')
                agent_id = accountability_data.get('agent_id', 'governance_agent')
                decision_factors = accountability_data.get('decision_factors', [])
                ethical_considerations = accountability_data.get('ethical_considerations', [])
                outcome = accountability_data.get('outcome', 'action_completed')
            
            # Record accountability
            result = self.governance_engine.record_accountability(
                action, agent_id, decision_factors, ethical_considerations, outcome
            )
            
            # Track operation
            self.governance_operations.append({
                'operation': 'record_accountability',
                'action': action,
                'agent_id': agent_id,
                'result': result,
                'timestamp': time.time()
            })
            
            # Update performance stats
            if result.get('status') == 'success':
                self.governance_performance['accountability_records'] += 1
            
            self._update_performance_stats(result)
            
            return result
            
        except Exception as e:
            return {"error": f"Failed to record accountability: {str(e)}"}
    
    def _extract_action(self, accountability_data: str) -> str:
        """Extract action from accountability data."""
        data_lower = accountability_data.lower()
        
        if any(word in data_lower for word in ['decision', 'choose', 'select']):
            return 'ethical_decision_making'
        elif any(word in data_lower for word in ['bias', 'fairness', 'equity']):
            return 'bias_detection_and_mitigation'
        elif any(word in data_lower for word in ['privacy', 'data', 'security']):
            return 'privacy_protection'
        elif any(word in data_lower for word in ['transparency', 'explain', 'clear']):
            return 'transparency_enhancement'
        else:
            return 'general_governance_action'
    
    def _extract_decision_factors(self, accountability_data: str) -> List[str]:
        """Extract decision factors from accountability data."""
        factors = []
        data_lower = accountability_data.lower()
        
        if any(word in data_lower for word in ['safety', 'risk']):
            factors.append('safety_considerations')
        if any(word in data_lower for word in ['privacy', 'confidential']):
            factors.append('privacy_concerns')
        if any(word in data_lower for word in ['fairness', 'equity']):
            factors.append('fairness_requirements')
        if any(word in data_lower for word in ['transparency', 'explain']):
            factors.append('transparency_needs')
        
        return factors if factors else ['general_ethical_considerations']
    
    def _extract_ethical_considerations(self, accountability_data: str) -> List[str]:
        """Extract ethical considerations from accountability data."""
        considerations = []
        data_lower = accountability_data.lower()
        
        if any(word in data_lower for word in ['benefit', 'help', 'good']):
            considerations.append('beneficence')
        if any(word in data_lower for word in ['harm', 'risk', 'danger']):
            considerations.append('non_maleficence')
        if any(word in data_lower for word in ['autonomy', 'choice', 'freedom']):
            considerations.append('autonomy')
        if any(word in data_lower for word in ['justice', 'fair', 'equal']):
            considerations.append('justice')
        
        return considerations if considerations else ['general_ethical_principles']
    
    def _get_governance_stats(self) -> Dict[str, Any]:
        """Get comprehensive governance statistics."""
        try:
            # Get engine stats
            engine_stats = self.governance_engine.get_governance_stats()
            
            # Get agent performance stats
            performance_stats = {
                'total_operations': self.governance_performance['total_operations'],
                'ethical_decisions': self.governance_performance['ethical_decisions'],
                'value_alignments': self.governance_performance['value_alignments'],
                'bias_assessments': self.governance_performance['bias_assessments'],
                'fairness_reports': self.governance_performance['fairness_reports'],
                'accountability_records': self.governance_performance['accountability_records']
            }
            
            # Calculate success rates
            if performance_stats['total_operations'] > 0:
                performance_stats['ethical_decision_rate'] = (
                    performance_stats['ethical_decisions'] / performance_stats['total_operations']
                )
                performance_stats['value_alignment_rate'] = (
                    performance_stats['value_alignments'] / performance_stats['total_operations']
                )
                performance_stats['bias_detection_rate'] = (
                    performance_stats['bias_assessments'] / performance_stats['total_operations']
                )
                performance_stats['fairness_assessment_rate'] = (
                    performance_stats['fairness_reports'] / performance_stats['total_operations']
                )
            else:
                performance_stats.update({
                    'ethical_decision_rate': 0.0,
                    'value_alignment_rate': 0.0,
                    'bias_detection_rate': 0.0,
                    'fairness_assessment_rate': 0.0
                })
            
            # Get recent operations
            recent_operations = self.governance_operations[-10:] if self.governance_operations else []
            
            return {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "engine_stats": engine_stats,
                "performance_stats": performance_stats,
                "recent_operations": recent_operations,
                "settings": {
                    "ethical_principles_enabled": self.ethical_principles_enabled,
                    "value_alignment_enabled": self.value_alignment_enabled,
                    "bias_detection_enabled": self.bias_detection_enabled,
                    "fairness_monitoring_enabled": self.fairness_monitoring_enabled,
                    "transparency_level": self.transparency_level
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to get governance stats: {str(e)}"}
    
    def _export_governance_data(self, filename: str = None) -> Dict[str, Any]:
        """Export governance data to JSON."""
        try:
            export_file = self.governance_engine.export_governance_data(filename)
            
            return {
                "status": "success",
                "export_file": export_file,
                "export_timestamp": datetime.now().isoformat(),
                "message": f"Governance data exported to: {export_file}"
            }
            
        except Exception as e:
            return {"error": f"Failed to export governance data: {str(e)}"}
    
    def _analyze_ethics(self, analysis_data: str) -> Dict[str, Any]:
        """Analyze ethical aspects of decisions and actions."""
        try:
            # Analyze ethical decisions
            ethical_decisions = self.governance_engine.ethical_decisions
            decision_analysis = {
                'total_decisions': len(ethical_decisions),
                'principles_used': {},
                'confidence_distribution': {'high': 0, 'medium': 0, 'low': 0},
                'impact_assessments': []
            }
            
            if ethical_decisions:
                # Analyze principles used
                for decision in ethical_decisions.values():
                    for principle in decision.ethical_principles:
                        principle_name = principle.value
                        decision_analysis['principles_used'][principle_name] = (
                            decision_analysis['principles_used'].get(principle_name, 0) + 1
                        )
                    
                    # Analyze confidence levels
                    if decision.confidence >= 0.8:
                        decision_analysis['confidence_distribution']['high'] += 1
                    elif decision.confidence >= 0.5:
                        decision_analysis['confidence_distribution']['medium'] += 1
                    else:
                        decision_analysis['confidence_distribution']['low'] += 1
                    
                    # Collect impact assessments
                    decision_analysis['impact_assessments'].append(decision.impact_assessment)
            
            return {
                "status": "success",
                "analysis": decision_analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to analyze ethics: {str(e)}"}
    
    def _compliance_check(self, compliance_data: str) -> Dict[str, Any]:
        """Check compliance with ethical guidelines and regulations."""
        try:
            # Parse compliance data
            if isinstance(compliance_data, str):
                context = compliance_data
            else:
                context = compliance_data.get('context', 'general compliance check')
            
            # Perform compliance checks
            compliance_results = {
                'ethical_principles_compliance': True,
                'value_alignment_compliance': True,
                'bias_detection_compliance': True,
                'fairness_compliance': True,
                'transparency_compliance': True,
                'accountability_compliance': True,
                'recommendations': []
            }
            
            # Check each compliance area
            if not self.ethical_principles_enabled:
                compliance_results['ethical_principles_compliance'] = False
                compliance_results['recommendations'].append("Enable ethical principles framework")
            
            if not self.value_alignment_enabled:
                compliance_results['value_alignment_compliance'] = False
                compliance_results['recommendations'].append("Enable value alignment assessment")
            
            if not self.bias_detection_enabled:
                compliance_results['bias_detection_compliance'] = False
                compliance_results['recommendations'].append("Enable bias detection mechanisms")
            
            if not self.fairness_monitoring_enabled:
                compliance_results['fairness_compliance'] = False
                compliance_results['recommendations'].append("Enable fairness monitoring")
            
            if self.transparency_level == "black_box":
                compliance_results['transparency_compliance'] = False
                compliance_results['recommendations'].append("Increase transparency levels")
            
            # Calculate overall compliance
            compliance_score = sum(compliance_results.values()) / len(compliance_results)
            compliance_results['overall_compliance_score'] = compliance_score
            
            return {
                "status": "success",
                "compliance_results": compliance_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to check compliance: {str(e)}"}
    
    def _update_performance_stats(self, result: Dict[str, Any]):
        """Update governance performance statistics."""
        self.governance_performance['total_operations'] += 1
    
    def get_governance_recommendations(self, context: str = None) -> Dict[str, Any]:
        """Get governance-related recommendations."""
        try:
            recommendations = []
            
            # Check ethical decision performance
            decision_rate = (self.governance_performance['ethical_decisions'] / 
                           self.governance_performance['total_operations'] 
                           if self.governance_performance['total_operations'] > 0 else 0)
            
            if decision_rate < 0.3:
                recommendations.append({
                    'type': 'ethical_decisions',
                    'priority': 'high',
                    'message': f'Ethical decision rate is {decision_rate:.1%}, consider improving ethical decision-making processes',
                    'action': 'enhance_ethical_frameworks'
                })
            
            # Check value alignment performance
            alignment_rate = (self.governance_performance['value_alignments'] / 
                            self.governance_performance['total_operations'] 
                            if self.governance_performance['total_operations'] > 0 else 0)
            
            if alignment_rate < 0.2:
                recommendations.append({
                    'type': 'value_alignment',
                    'priority': 'high',
                    'message': f'Value alignment rate is {alignment_rate:.1%}, consider increasing value alignment assessments',
                    'action': 'increase_value_alignment_checks'
                })
            
            # Check bias detection performance
            bias_rate = (self.governance_performance['bias_assessments'] / 
                        self.governance_performance['total_operations'] 
                        if self.governance_performance['total_operations'] > 0 else 0)
            
            if bias_rate < 0.2:
                recommendations.append({
                    'type': 'bias_detection',
                    'priority': 'medium',
                    'message': f'Bias detection rate is {bias_rate:.1%}, consider improving bias detection capabilities',
                    'action': 'enhance_bias_detection'
                })
            
            return {
                "status": "success",
                "recommendations": recommendations,
                "recommendations_count": len(recommendations)
            }
            
        except Exception as e:
            return {"error": f"Failed to get recommendations: {str(e)}"} 