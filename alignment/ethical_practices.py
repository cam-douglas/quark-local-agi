#!/usr/bin/env python3
"""
Ethical Practices for Quark AI Assistant
============================================

Implements ethical AI practices including fairness, transparency,
accountability, and responsible AI development.

Part of Pillar 15: Safety & Alignment
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from core.safety_guardrails import SafetyGuardrails, ChangeType, ChangeSeverity


class EthicalPrinciple(Enum):
    """Core ethical principles for AI systems."""
    FAIRNESS = "fairness"
    TRANSPARENCY = "transparency"
    ACCOUNTABILITY = "accountability"
    PRIVACY = "privacy"
    BENEFICENCE = "beneficence"
    NON_MALEFICENCE = "non_maleficence"


class BiasType(Enum):
    """Types of bias that can be detected."""
    GENDER = "gender"
    RACIAL = "racial"
    AGE = "age"
    LANGUAGE = "language"


@dataclass
class EthicalAssessment:
    """Result of ethical assessment."""
    principle: EthicalPrinciple
    score: float
    issues: List[str]
    recommendations: List[str]
    timestamp: float


@dataclass
class BiasDetection:
    """Result of bias detection."""
    bias_type: BiasType
    confidence: float
    evidence: List[str]
    impact_assessment: str
    mitigation_suggestions: List[str]


class EthicalPractices:
    """Comprehensive ethical practices framework."""
    
    def __init__(self, ethics_dir: str = None):
        self.ethics_dir = ethics_dir or os.path.join(os.path.dirname(__file__), '..', 'alignment', 'ethics_data')
        os.makedirs(self.ethics_dir, exist_ok=True)
        
        # Ethics settings
        self.ethics_enabled = True
        self.bias_detection_enabled = True
        self.transparency_required = True
        self.fairness_threshold = 0.8
        
        # Initialize safety guardrails
        self.safety_guardrails = SafetyGuardrails()
        
        # Ethics tracking
        self.ethics_assessments = []
        self.bias_detections = []
        self.transparency_logs = []
        
        # Load ethical guidelines
        self._load_ethical_guidelines()
        
    def _load_ethical_guidelines(self):
        """Load ethical guidelines and principles."""
        self.ethical_guidelines = {
            EthicalPrinciple.FAIRNESS: {
                "description": "Ensure fair treatment for all users",
                "requirements": [
                    "Avoid discriminatory language or behavior",
                    "Provide equal access to capabilities",
                    "Detect and mitigate bias in responses"
                ]
            },
            EthicalPrinciple.TRANSPARENCY: {
                "description": "Be open about capabilities and limitations",
                "requirements": [
                    "Explain reasoning and decisions",
                    "Disclose limitations and uncertainties",
                    "Provide clear explanations"
                ]
            },
            EthicalPrinciple.ACCOUNTABILITY: {
                "description": "Take responsibility for actions and decisions",
                "requirements": [
                    "Log all significant decisions",
                    "Provide explanations for actions",
                    "Allow for human oversight"
                ]
            },
            EthicalPrinciple.PRIVACY: {
                "description": "Protect user privacy and data",
                "requirements": [
                    "Minimize data collection",
                    "Secure data storage",
                    "Respect user consent"
                ]
            },
            EthicalPrinciple.BENEFICENCE: {
                "description": "Act to benefit users and society",
                "requirements": [
                    "Provide helpful and accurate information",
                    "Consider societal impact",
                    "Promote positive outcomes"
                ]
            },
            EthicalPrinciple.NON_MALEFICENCE: {
                "description": "Avoid causing harm",
                "requirements": [
                    "Prevent dangerous actions",
                    "Block harmful content",
                    "Protect user safety"
                ]
            }
        }
        
    def assess_ethical_compliance(self, action: str, context: Dict[str, Any] = None) -> List[EthicalAssessment]:
        """Assess ethical compliance of an action."""
        if not self.ethics_enabled:
            return []
        
        assessments = []
        
        for principle in EthicalPrinciple:
            assessment = self._assess_principle(principle, action, context)
            assessments.append(assessment)
            self.ethics_assessments.append(assessment)
        
        return assessments
    
    def _assess_principle(self, principle: EthicalPrinciple, action: str, 
                         context: Dict[str, Any] = None) -> EthicalAssessment:
        """Assess compliance with a specific ethical principle."""
        issues = []
        recommendations = []
        score = 1.0
        
        # Simple heuristic assessment
        if principle == EthicalPrinciple.FAIRNESS:
            score, issues, recommendations = self._assess_fairness(action, context)
        elif principle == EthicalPrinciple.TRANSPARENCY:
            score, issues, recommendations = self._assess_transparency(action, context)
        elif principle == EthicalPrinciple.ACCOUNTABILITY:
            score, issues, recommendations = self._assess_accountability(action, context)
        elif principle == EthicalPrinciple.PRIVACY:
            score, issues, recommendations = self._assess_privacy(action, context)
        elif principle == EthicalPrinciple.BENEFICENCE:
            score, issues, recommendations = self._assess_beneficence(action, context)
        elif principle == EthicalPrinciple.NON_MALEFICENCE:
            score, issues, recommendations = self._assess_non_maleficence(action, context)
        
        return EthicalAssessment(
            principle=principle,
            score=score,
            issues=issues,
            recommendations=recommendations,
            timestamp=time.time()
        )
    
    def _assess_fairness(self, action: str, context: Dict[str, Any] = None) -> tuple:
        """Assess fairness of an action."""
        issues = []
        recommendations = []
        score = 1.0
        
        # Check for discriminatory language
        discriminatory_terms = ['only', 'just', 'merely', 'simply', 'obviously']
        
        for term in discriminatory_terms:
            if term in action.lower():
                issues.append(f"Potentially dismissive language: '{term}'")
                recommendations.append("Use more inclusive and respectful language")
                score -= 0.1
        
        return max(0.0, score), issues, recommendations
    
    def _assess_transparency(self, action: str, context: Dict[str, Any] = None) -> tuple:
        """Assess transparency of an action."""
        issues = []
        recommendations = []
        score = 1.0
        
        # Check for unclear language
        unclear_indicators = ['somehow', 'magically', 'automatically', 'just works']
        
        for indicator in unclear_indicators:
            if indicator in action.lower():
                issues.append(f"Unclear explanation: '{indicator}'")
                recommendations.append("Provide clear, step-by-step explanations")
                score -= 0.15
        
        return max(0.0, score), issues, recommendations
    
    def _assess_accountability(self, action: str, context: Dict[str, Any] = None) -> tuple:
        """Assess accountability of an action."""
        issues = []
        recommendations = []
        score = 1.0
        
        # Check for responsibility avoidance
        avoidance_terms = ['not my fault', 'not responsible', 'can\'t help']
        
        for term in avoidance_terms:
            if term in action.lower():
                issues.append(f"Accountability avoidance: '{term}'")
                recommendations.append("Take responsibility and provide solutions")
                score -= 0.2
        
        return max(0.0, score), issues, recommendations
    
    def _assess_privacy(self, action: str, context: Dict[str, Any] = None) -> tuple:
        """Assess privacy implications of an action."""
        issues = []
        recommendations = []
        score = 1.0
        
        # Check for personal information handling
        privacy_indicators = ['personal', 'private', 'confidential', 'password']
        
        for indicator in privacy_indicators:
            if indicator in action.lower():
                issues.append(f"Privacy concern: '{indicator}' mentioned")
                recommendations.append("Ensure proper data protection and consent")
                score -= 0.15
        
        return max(0.0, score), issues, recommendations
    
    def _assess_beneficence(self, action: str, context: Dict[str, Any] = None) -> tuple:
        """Assess beneficence of an action."""
        issues = []
        recommendations = []
        score = 1.0
        
        # Check for helpfulness
        helpful_indicators = ['help', 'assist', 'support', 'benefit', 'improve']
        
        helpful_count = sum(1 for indicator in helpful_indicators if indicator in action.lower())
        if helpful_count == 0:
            issues.append("Action may not be clearly beneficial")
            recommendations.append("Ensure the action provides clear value to the user")
            score -= 0.1
        
        return max(0.0, score), issues, recommendations
    
    def _assess_non_maleficence(self, action: str, context: Dict[str, Any] = None) -> tuple:
        """Assess non-maleficence of an action."""
        issues = []
        recommendations = []
        score = 1.0
        
        # Check for potential harm
        harm_indicators = ['harm', 'damage', 'hurt', 'danger', 'risk']
        
        for indicator in harm_indicators:
            if indicator in action.lower():
                issues.append(f"Potential harm indicator: '{indicator}'")
                recommendations.append("Ensure safety measures are in place")
                score -= 0.2
        
        return max(0.0, score), issues, recommendations
    
    def detect_bias(self, content: str, context: Dict[str, Any] = None) -> List[BiasDetection]:
        """Detect bias in content."""
        if not self.bias_detection_enabled:
            return []
        
        detections = []
        
        for bias_type in BiasType:
            detection = self._detect_bias_type(bias_type, content, context)
            if detection.confidence > 0.3:  # Only report significant bias
                detections.append(detection)
                self.bias_detections.append(detection)
        
        return detections
    
    def _detect_bias_type(self, bias_type: BiasType, content: str, 
                          context: Dict[str, Any] = None) -> BiasDetection:
        """Detect specific type of bias."""
        import re
        
        evidence = []
        confidence = 0.0
        
        if bias_type == BiasType.GENDER:
            # Check for gender stereotypes
            gender_patterns = [
                r'\b(men|man|male)\s+(are|is)\s+(strong|aggressive)',
                r'\b(women|woman|female)\s+(are|is)\s+(emotional|nurturing)'
            ]
            
            for pattern in gender_patterns:
                matches = re.findall(pattern, content.lower())
                if matches:
                    evidence.extend(matches)
                    confidence += 0.2
        
        elif bias_type == BiasType.RACIAL:
            # Check for racial stereotypes
            racial_patterns = [
                r'\b(race|ethnicity)\s+(determines|affects)',
                r'\b(certain\s+races)\s+(are|tend\s+to)'
            ]
            
            for pattern in racial_patterns:
                matches = re.findall(pattern, content.lower())
                if matches:
                    evidence.extend(matches)
                    confidence += 0.25
        
        impact_assessment = self._assess_bias_impact(bias_type, confidence)
        mitigation_suggestions = self._generate_bias_mitigation(bias_type, evidence)
        
        return BiasDetection(
            bias_type=bias_type,
            confidence=confidence,
            evidence=evidence,
            impact_assessment=impact_assessment,
            mitigation_suggestions=mitigation_suggestions
        )
    
    def _assess_bias_impact(self, bias_type: BiasType, confidence: float) -> str:
        """Assess the impact of detected bias."""
        if confidence > 0.8:
            return f"High impact: {bias_type.value} bias likely to affect user experience"
        elif confidence > 0.5:
            return f"Medium impact: {bias_type.value} bias may affect some users"
        else:
            return f"Low impact: {bias_type.value} bias detected but minimal impact"
    
    def _generate_bias_mitigation(self, bias_type: BiasType, evidence: List[str]) -> List[str]:
        """Generate suggestions for bias mitigation."""
        suggestions = []
        
        if bias_type == BiasType.GENDER:
            suggestions.extend([
                "Use gender-neutral language",
                "Avoid gender stereotypes",
                "Consider diverse perspectives"
            ])
        elif bias_type == BiasType.RACIAL:
            suggestions.extend([
                "Avoid racial generalizations",
                "Use diverse examples",
                "Focus on individual characteristics"
            ])
        
        return suggestions
    
    def log_transparency(self, action: str, explanation: str, context: Dict[str, Any] = None):
        """Log transparency information for an action."""
        if not self.transparency_required:
            return
        
        log_entry = {
            'action': action,
            'explanation': explanation,
            'context': context or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.transparency_logs.append(log_entry)
    
    def get_ethics_report(self) -> Dict[str, Any]:
        """Generate comprehensive ethics report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'ethics_enabled': self.ethics_enabled,
            'bias_detection_enabled': self.bias_detection_enabled,
            'total_assessments': len(self.ethics_assessments),
            'total_bias_detections': len(self.bias_detections),
            'transparency_logs': len(self.transparency_logs),
            'recent_assessments': [
                {
                    'principle': assessment.principle.value,
                    'score': assessment.score,
                    'issues_count': len(assessment.issues)
                }
                for assessment in self.ethics_assessments[-10:]  # Last 10
            ],
            'recent_bias_detections': [
                {
                    'bias_type': detection.bias_type.value,
                    'confidence': detection.confidence
                }
                for detection in self.bias_detections[-10:]  # Last 10
            ]
        }
    
    def export_ethics_data(self, filename: str = None) -> str:
        """Export ethics data to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ethics_data_{timestamp}.json"
        
        filepath = os.path.join(self.ethics_dir, filename)
        
        # Convert enum values to strings for JSON serialization
        ethics_assessments_serializable = []
        for assessment in self.ethics_assessments:
            assessment_dict = asdict(assessment)
            assessment_dict['principle'] = assessment_dict['principle'].value
            ethics_assessments_serializable.append(assessment_dict)
        
        bias_detections_serializable = []
        for detection in self.bias_detections:
            detection_dict = asdict(detection)
            detection_dict['bias_type'] = detection_dict['bias_type'].value
            bias_detections_serializable.append(detection_dict)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'ethics_report': self.get_ethics_report(),
            'ethics_assessments': ethics_assessments_serializable,
            'bias_detections': bias_detections_serializable,
            'transparency_logs': self.transparency_logs
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath 