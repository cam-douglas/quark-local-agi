#!/usr/bin/env python3
"""
Safety Agent for Quark AI Assistant
=======================================

Implements comprehensive safety monitoring, content filtering, and ethical
practices to ensure the AI system remains safe and aligned with human values.

Part of Pillar 15: Safety & Alignment
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from .base import Agent
from core.safety_guardrails import SafetyGuardrails, ChangeType, ChangeSeverity
from alignment.content_filtering import ContentFilter, FilterResult
from alignment.ethical_practices import EthicalPractices, EthicalAssessment
from alignment.alignment_monitor import AlignmentMonitor, AlignmentReport
from alignment.rlhf_agent import RLHFAgent
from alignment.adversarial_testing import AdversarialTesting


@dataclass
class SafetyAssessment:
    """Comprehensive safety assessment result."""
    is_safe: bool
    content_filter_result: FilterResult
    ethical_assessments: List[EthicalAssessment]
    alignment_report: Optional[AlignmentReport]
    safety_score: float
    recommendations: List[str]
    timestamp: float


class SafetyAgent(Agent):
    """Comprehensive safety agent for monitoring and ensuring AI safety."""
    
    def __init__(self, model_name: str = "safety_agent", safety_dir: str = None):
        super().__init__(model_name)
        self.safety_dir = safety_dir or os.path.join(os.path.dirname(__file__), '..', 'safety')
        os.makedirs(self.safety_dir, exist_ok=True)
        
        # Safety components
        self.safety_guardrails = SafetyGuardrails()
        self.content_filter = ContentFilter()
        self.ethical_practices = EthicalPractices()
        self.alignment_monitor = AlignmentMonitor()
        self.rlhf_agent = RLHFAgent()
        self.adversarial_testing = AdversarialTesting()
        
        # Safety settings
        self.safety_enabled = True
        self.content_filtering_enabled = True
        self.ethics_monitoring_enabled = True
        self.alignment_monitoring_enabled = True
        self.rlhf_enabled = True
        self.adversarial_testing_enabled = True
        
        # Safety tracking
        self.safety_assessments = []
        self.safety_violations = []
        self.safety_improvements = []
        
    def load_model(self):
        """Load safety models and components."""
        try:
            # Initialize all safety components
            self.safety_guardrails = SafetyGuardrails()
            self.content_filter = ContentFilter()
            self.ethical_practices = EthicalPractices()
            self.alignment_monitor = AlignmentMonitor()
            self.rlhf_agent = RLHFAgent()
            self.adversarial_testing = AdversarialTesting()
            
            return True
        except Exception as e:
            print(f"Error loading safety models: {e}")
            return False
    
    def generate(self, input_data: str, operation: str = "assess_safety", **kwargs) -> Dict[str, Any]:
        """
        Generate safety assessment or perform safety operations.
        
        Args:
            input_data: Input text or data to assess
            operation: Safety operation to perform
            **kwargs: Additional parameters
            
        Returns:
            Safety assessment or operation result
        """
        if not self.safety_enabled:
            return {"status": "safety_disabled", "message": "Safety system is disabled"}
        
        try:
            if operation == "assess_safety":
                return self._assess_safety(input_data, **kwargs)
            elif operation == "filter_content":
                return self._filter_content(input_data, **kwargs)
            elif operation == "assess_ethics":
                return self._assess_ethics(input_data, **kwargs)
            elif operation == "measure_alignment":
                return self._measure_alignment(input_data, **kwargs)
            elif operation == "collect_feedback":
                return self._collect_feedback(input_data, **kwargs)
            elif operation == "run_adversarial_tests":
                return self._run_adversarial_tests(input_data, **kwargs)
            elif operation == "get_safety_report":
                return self._get_safety_report()
            elif operation == "export_safety_data":
                return self._export_safety_data(**kwargs)
            else:
                return {"error": f"Unknown safety operation: {operation}"}
                
        except Exception as e:
            return {"error": f"Safety operation failed: {str(e)}"}
    
    def _assess_safety(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform comprehensive safety assessment."""
        if context is None:
            context = {}
        
        # Content filtering
        filter_result = self.content_filter.filter_content(content, context)
        
        # Ethical assessment
        ethical_assessments = self.ethical_practices.assess_ethical_compliance(content, context)
        
        # Alignment measurement
        interaction_data = {
            'request': content,
            'response': context.get('response', ''),
            'context': context
        }
        alignment_report = self.alignment_monitor.measure_alignment(interaction_data)
        
        # Calculate overall safety score
        safety_score = self._calculate_safety_score(filter_result, ethical_assessments, alignment_report)
        
        # Generate recommendations
        recommendations = self._generate_safety_recommendations(
            filter_result, ethical_assessments, alignment_report
        )
        
        # Create comprehensive assessment
        assessment = SafetyAssessment(
            is_safe=filter_result.is_safe and safety_score >= 0.7,
            content_filter_result=filter_result,
            ethical_assessments=ethical_assessments,
            alignment_report=alignment_report,
            safety_score=safety_score,
            recommendations=recommendations,
            timestamp=time.time()
        )
        
        self.safety_assessments.append(assessment)
        
        return {
            "status": "success",
            "assessment": {
                "is_safe": assessment.is_safe,
                "safety_score": assessment.safety_score,
                "content_filter": {
                    "is_safe": filter_result.is_safe,
                    "categories": [cat.value for cat in filter_result.categories],
                    "severity": filter_result.severity.value,
                    "confidence": filter_result.confidence,
                    "explanation": filter_result.explanation
                },
                "ethics": {
                    "total_assessments": len(ethical_assessments),
                    "average_score": sum(a.score for a in ethical_assessments) / max(1, len(ethical_assessments)),
                    "issues_found": sum(len(a.issues) for a in ethical_assessments)
                },
                "alignment": {
                    "overall_score": alignment_report.overall_score,
                    "overall_status": alignment_report.overall_status.value,
                    "recommendations": alignment_report.recommendations
                },
                "recommendations": recommendations
            }
        }
    
    def _filter_content(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Filter content for safety."""
        if not self.content_filtering_enabled:
            return {"status": "filtering_disabled", "is_safe": True}
        
        filter_result = self.content_filter.filter_content(content, context)
        
        return {
            "status": "success",
            "is_safe": filter_result.is_safe,
            "categories": [cat.value for cat in filter_result.categories],
            "severity": filter_result.severity.value,
            "confidence": filter_result.confidence,
            "flagged_terms": filter_result.flagged_terms,
            "explanation": filter_result.explanation
        }
    
    def _assess_ethics(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess ethical compliance."""
        if not self.ethics_monitoring_enabled:
            return {"status": "ethics_disabled"}
        
        assessments = self.ethical_practices.assess_ethical_compliance(content, context)
        
        return {
            "status": "success",
            "total_assessments": len(assessments),
            "assessments": [
                {
                    "principle": assessment.principle.value,
                    "score": assessment.score,
                    "issues": assessment.issues,
                    "recommendations": assessment.recommendations
                }
                for assessment in assessments
            ],
            "average_score": sum(a.score for a in assessments) / max(1, len(assessments)),
            "total_issues": sum(len(a.issues) for a in assessments)
        }
    
    def _measure_alignment(self, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Measure alignment with human values."""
        if not self.alignment_monitoring_enabled:
            return {"status": "alignment_disabled"}
        
        interaction_data = {
            'request': content,
            'response': context.get('response', ''),
            'context': context
        }
        
        report = self.alignment_monitor.measure_alignment(interaction_data)
        
        return {
            "status": "success",
            "overall_score": report.overall_score,
            "overall_status": report.overall_status.value,
            "measurements": [
                {
                    "metric": measurement.metric.value,
                    "score": measurement.score,
                    "status": measurement.status.value,
                    "evidence": measurement.evidence
                }
                for measurement in report.measurements
            ],
            "recommendations": report.recommendations
        }
    
    def _collect_feedback(self, feedback_data: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Collect human feedback for RLHF."""
        if not self.rlhf_enabled:
            return {"status": "rlhf_disabled"}
        
        try:
            feedback = json.loads(feedback_data) if isinstance(feedback_data, str) else feedback_data
            
            result = self.rlhf_agent.collect_feedback(
                prompt=feedback.get('prompt', ''),
                response=feedback.get('response', ''),
                feedback_type=feedback.get('feedback_type', 'rating'),
                rating=feedback.get('rating', 0),
                feedback_text=feedback.get('feedback_text', ''),
                context=context
            )
            
            return {
                "status": "success",
                "feedback_id": result.get('feedback_id'),
                "feedback_collected": True
            }
        except Exception as e:
            return {"error": f"Failed to collect feedback: {str(e)}"}
    
    def _run_adversarial_tests(self, test_data: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run adversarial testing."""
        if not self.adversarial_testing_enabled:
            return {"status": "adversarial_testing_disabled"}
        
        try:
            test_config = json.loads(test_data) if isinstance(test_data, str) else test_data
            
            result = self.adversarial_testing.run_test_suite(
                test_categories=test_config.get('categories', []),
                custom_prompts=test_config.get('custom_prompts', []),
                context=context
            )
            
            return {
                "status": "success",
                "tests_run": result.get('tests_run', 0),
                "vulnerabilities_found": result.get('vulnerabilities_found', 0),
                "overall_score": result.get('overall_score', 0.0),
                "test_results": result.get('test_results', [])
            }
        except Exception as e:
            return {"error": f"Adversarial testing failed: {str(e)}"}
    
    def _calculate_safety_score(self, filter_result: FilterResult, 
                               ethical_assessments: List[EthicalAssessment],
                               alignment_report: AlignmentReport) -> float:
        """Calculate overall safety score."""
        scores = []
        
        # Content filter score
        if filter_result.is_safe:
            scores.append(1.0)
        else:
            scores.append(1.0 - filter_result.confidence)
        
        # Ethics score
        if ethical_assessments:
            ethics_score = sum(a.score for a in ethical_assessments) / len(ethical_assessments)
            scores.append(ethics_score)
        
        # Alignment score
        scores.append(alignment_report.overall_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _generate_safety_recommendations(self, filter_result: FilterResult,
                                       ethical_assessments: List[EthicalAssessment],
                                       alignment_report: AlignmentReport) -> List[str]:
        """Generate safety recommendations."""
        recommendations = []
        
        # Content filter recommendations
        if not filter_result.is_safe:
            recommendations.append(f"Content flagged: {filter_result.explanation}")
        
        # Ethics recommendations
        for assessment in ethical_assessments:
            if assessment.score < 0.8:
                recommendations.extend(assessment.recommendations)
        
        # Alignment recommendations
        if alignment_report.overall_score < 0.7:
            recommendations.extend(alignment_report.recommendations)
        
        return recommendations
    
    def _get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report."""
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "safety_enabled": self.safety_enabled,
            "content_filtering_enabled": self.content_filtering_enabled,
            "ethics_monitoring_enabled": self.ethics_monitoring_enabled,
            "alignment_monitoring_enabled": self.alignment_monitoring_enabled,
            "rlhf_enabled": self.rlhf_enabled,
            "adversarial_testing_enabled": self.adversarial_testing_enabled,
            "total_assessments": len(self.safety_assessments),
            "total_violations": len(self.safety_violations),
            "total_improvements": len(self.safety_improvements),
            "content_filter_stats": self.content_filter.get_filter_stats(),
            "ethics_report": self.ethical_practices.get_ethics_report(),
            "alignment_stats": self.alignment_monitor.get_alignment_stats(),
            "recent_assessments": [
                {
                    "is_safe": assessment.is_safe,
                    "safety_score": assessment.safety_score,
                    "timestamp": assessment.timestamp
                }
                for assessment in self.safety_assessments[-10:]  # Last 10
            ],
            "alignment_stats": self.alignment_monitor.get_alignment_stats()
        }
    
    def _export_safety_data(self, filename: str = None) -> Dict[str, Any]:
        """Export safety data."""
        try:
            # Export from each component
            filter_data = self.content_filter.export_filter_data()
            ethics_data = self.ethical_practices.export_ethics_data()
            alignment_data = self.alignment_monitor.export_alignment_data()
            
            # Create comprehensive export
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"safety_export_{timestamp}.json"
            
            filepath = os.path.join(self.safety_dir, filename)
            
            # Convert enum values to strings for JSON serialization
            safety_assessments_serializable = []
            for assessment in self.safety_assessments:
                assessment_dict = asdict(assessment)
                # Convert content filter result
                if assessment_dict['content_filter_result']:
                    filter_result = assessment_dict['content_filter_result']
                    filter_result['severity'] = filter_result['severity'].value
                    filter_result['categories'] = [cat.value for cat in filter_result['categories']]
                # Convert ethical assessments
                if assessment_dict['ethical_assessments']:
                    for ethical_assessment in assessment_dict['ethical_assessments']:
                        ethical_assessment['principle'] = ethical_assessment['principle'].value
                # Convert alignment report
                if assessment_dict['alignment_report']:
                    alignment_report = assessment_dict['alignment_report']
                    alignment_report['overall_status'] = alignment_report['overall_status'].value
                    # Convert measurements
                    if alignment_report['measurements']:
                        for measurement in alignment_report['measurements']:
                            measurement['metric'] = measurement['metric'].value
                            measurement['status'] = measurement['status'].value
                safety_assessments_serializable.append(assessment_dict)
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'safety_report': self._get_safety_report(),
                'safety_assessments': safety_assessments_serializable,
                'safety_violations': self.safety_violations,
                'safety_improvements': self.safety_improvements,
                'component_exports': {
                    'content_filter': filter_data,
                    'ethics': ethics_data,
                    'alignment': alignment_data
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return {
                "status": "success",
                "export_file": filepath,
                "export_size": len(json.dumps(export_data))
            }
            
        except Exception as e:
            return {"error": f"Export failed: {str(e)}"} 