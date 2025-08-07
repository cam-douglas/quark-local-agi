"""
Explainability, Transparency & Auditing Agent
Pillar 27: Human-readable rationales, audit logs, and transparency mechanisms
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid
from pathlib import Path
from collections import defaultdict

from .base import Agent as BaseAgent


class ExplanationType(Enum):
    """Types of explanations"""
    DECISION_RATIONALE = "decision_rationale"
    FEATURE_IMPORTANCE = "feature_importance"
    CONFIDENCE_SCORE = "confidence_score"
    ALTERNATIVE_OPTIONS = "alternative_options"
    UNCERTAINTY_ANALYSIS = "uncertainty_analysis"


class TransparencyLevel(Enum):
    """Levels of transparency"""
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"
    EXPERT = "expert"


class AuditEventType(Enum):
    """Types of audit events"""
    DECISION_MADE = "decision_made"
    MODEL_ACCESSED = "model_accessed"
    DATA_ACCESSED = "data_accessed"
    CONFIGURATION_CHANGED = "configuration_changed"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"


@dataclass
class Explanation:
    """Explanation for a decision or action"""
    explanation_id: str
    decision_id: str
    explanation_type: ExplanationType
    content: str
    confidence: float
    factors: List[Dict[str, Any]]
    alternatives: List[Dict[str, Any]]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class AuditEvent:
    """Audit event for tracking system activities"""
    event_id: str
    event_type: AuditEventType
    component: str
    action: str
    user_id: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    severity: str
    metadata: Dict[str, Any]


@dataclass
class TransparencyReport:
    """Comprehensive transparency report"""
    report_id: str
    component: str
    transparency_level: TransparencyLevel
    explanations: List[Explanation]
    audit_events: List[AuditEvent]
    performance_metrics: Dict[str, Any]
    data_usage: Dict[str, Any]
    model_info: Dict[str, Any]
    generated_time: datetime
    metadata: Dict[str, Any]


class ExplainabilityAgent(BaseAgent):
    """
    Explainability, Transparency & Auditing Agent
    
    Implements human-readable decision rationales, comprehensive audit logs,
    and transparency mechanisms for all system decisions.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("explainability")
        self.config = config or {}
        
        # Add logger
        import logging
        self.logger = logging.getLogger(__name__)
        
        # Explanation storage
        self.explanations: Dict[str, Explanation] = {}
        self.audit_events: List[AuditEvent] = []
        self.transparency_reports: List[TransparencyReport] = []
        
        # Explanation generators
        self.explanation_generators = {
            ExplanationType.DECISION_RATIONALE: self._generate_decision_rationale,
            ExplanationType.FEATURE_IMPORTANCE: self._generate_feature_importance,
            ExplanationType.CONFIDENCE_SCORE: self._generate_confidence_explanation,
            ExplanationType.ALTERNATIVE_OPTIONS: self._generate_alternatives,
            ExplanationType.UNCERTAINTY_ANALYSIS: self._generate_uncertainty_analysis
        }
        
        # Audit loggers
        self.audit_loggers = {
            AuditEventType.DECISION_MADE: self._log_decision_event,
            AuditEventType.MODEL_ACCESSED: self._log_model_access,
            AuditEventType.DATA_ACCESSED: self._log_data_access,
            AuditEventType.CONFIGURATION_CHANGED: self._log_config_change,
            AuditEventType.ERROR_OCCURRED: self._log_error_event,
            AuditEventType.PERFORMANCE_METRIC: self._log_performance_metric
        }
        
        # Performance tracking
        self.explanation_stats = defaultdict(int)
        self.audit_stats = defaultdict(int)
        self.transparency_stats = defaultdict(int)
        
        # Configuration
        self.max_explanations = 10000
        self.max_audit_events = 50000
        self.audit_retention_days = 365
        self.explanation_retention_days = 180
        
        # Initialize explainability systems
        self._initialize_explainability_systems()
    
    def load_model(self):
        """Load the explainability model"""
        return None  # Placeholder for actual model loading
    
    def generate(self, prompt: str, **kwargs):
        """Generate explainability response"""
        return f"Explainability response to: {prompt}"
    
    def _initialize_explainability_systems(self):
        """Initialize explainability systems"""
        self.logger.info("Initializing explainability systems...")
        
        # Set up explanation generators
        self._setup_explanation_generators()
        
        # Initialize audit loggers
        self._setup_audit_loggers()
        
        # Set up performance tracking
        self._setup_performance_tracking()
        
        self.logger.info("Explainability systems initialized successfully")
    
    def _setup_explanation_generators(self):
        """Set up explanation generators"""
        self.logger.info("Setting up explanation generators...")
        
        # Configure explanation generators
        for explanation_type, generator_func in self.explanation_generators.items():
            self.logger.info(f"Configured explanation generator: {explanation_type.value}")
    
    def _setup_audit_loggers(self):
        """Set up audit loggers"""
        self.logger.info("Setting up audit loggers...")
        
        # Configure audit loggers
        for event_type, logger_func in self.audit_loggers.items():
            self.logger.info(f"Configured audit logger: {event_type.value}")
    
    def _setup_performance_tracking(self):
        """Set up performance tracking"""
        self.logger.info("Setting up performance tracking...")
        
        # Initialize performance metrics
        metrics = ["explanations_generated", "audit_events_logged", "transparency_reports_created"]
        for metric in metrics:
            self.explanation_stats[metric] = 0
    
    async def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming messages for explainability"""
        start_time = time.time()
        
        try:
            # Extract explainability parameters
            action = message.get("action", "explain")
            decision_id = message.get("decision_id", "")
            explanation_type = message.get("explanation_type", "decision_rationale")
            transparency_level = message.get("transparency_level", "detailed")
            
            if action == "explain":
                result = await self._generate_explanation(decision_id, explanation_type, message)
            elif action == "audit":
                result = await self._log_audit_event(message)
            elif action == "transparency_report":
                result = await self._generate_transparency_report(message)
            elif action == "get_explanations":
                result = await self._get_explanations(message)
            elif action == "get_audit_log":
                result = await self._get_audit_log(message)
            else:
                raise ValueError(f"Unknown action: {action}")
            
            return {
                "status": "success",
                "action": action,
                "result": result,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            self.logger.error(f"Error in explainability: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _generate_explanation(self, decision_id: str, explanation_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an explanation for a decision"""
        self.logger.info(f"Generating explanation for decision: {decision_id}")
        
        # Get explanation generator
        explanation_type_enum = ExplanationType(explanation_type)
        generator_func = self.explanation_generators.get(explanation_type_enum)
        if not generator_func:
            raise ValueError(f"Unknown explanation type: {explanation_type}")
        
        # Generate explanation
        explanation_content = await generator_func(decision_id, context)
        
        # Create explanation object
        explanation_id = f"explanation_{uuid.uuid4().hex[:8]}"
        explanation = Explanation(
            explanation_id=explanation_id,
            decision_id=decision_id,
            explanation_type=explanation_type_enum,
            content=explanation_content["content"],
            confidence=explanation_content["confidence"],
            factors=explanation_content["factors"],
            alternatives=explanation_content["alternatives"],
            timestamp=datetime.now(),
            metadata=context
        )
        
        # Store explanation
        self.explanations[explanation_id] = explanation
        
        # Update statistics
        self.explanation_stats["explanations_generated"] += 1
        
        return {
            "explanation_id": explanation_id,
            "decision_id": decision_id,
            "explanation_type": explanation_type,
            "content": explanation.content,
            "confidence": explanation.confidence,
            "factors": explanation.factors,
            "alternatives": explanation.alternatives
        }
    
    async def _generate_decision_rationale(self, decision_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate decision rationale explanation"""
        decision_type = context.get("decision_type", "general")
        factors = context.get("factors", [])
        
        # Simulate decision rationale generation
        rationale_content = f"The decision was made based on the following factors: {', '.join(factors)}. "
        rationale_content += "The system analyzed multiple options and selected the most appropriate one based on the given criteria."
        
        return {
            "content": rationale_content,
            "confidence": 0.85,
            "factors": factors,
            "alternatives": [
                {"option": "Alternative A", "reason": "Different criteria weighting"},
                {"option": "Alternative B", "reason": "Conservative approach"}
            ]
        }
    
    async def _generate_feature_importance(self, decision_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate feature importance explanation"""
        features = context.get("features", {})
        
        # Simulate feature importance analysis
        importance_content = "Feature importance analysis shows the following key factors: "
        for feature, importance in features.items():
            importance_content += f"{feature} ({importance:.2f}), "
        
        return {
            "content": importance_content,
            "confidence": 0.90,
            "factors": [{"feature": k, "importance": v} for k, v in features.items()],
            "alternatives": []
        }
    
    async def _generate_confidence_explanation(self, decision_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate confidence score explanation"""
        confidence_score = context.get("confidence_score", 0.8)
        
        # Simulate confidence explanation
        if confidence_score > 0.9:
            confidence_text = "High confidence due to strong evidence and clear patterns."
        elif confidence_score > 0.7:
            confidence_text = "Moderate confidence with some uncertainty in the data."
        else:
            confidence_text = "Low confidence due to limited data or conflicting information."
        
        return {
            "content": f"Confidence score: {confidence_score:.2f}. {confidence_text}",
            "confidence": confidence_score,
            "factors": [{"factor": "data_quality", "impact": confidence_score}],
            "alternatives": []
        }
    
    async def _generate_alternatives(self, decision_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alternative options explanation"""
        alternatives = context.get("alternatives", [])
        
        # Simulate alternatives generation
        alternatives_content = "The following alternatives were considered: "
        for i, alt in enumerate(alternatives, 1):
            alternatives_content += f"{i}. {alt['option']} - {alt['reason']}. "
        
        return {
            "content": alternatives_content,
            "confidence": 0.75,
            "factors": [],
            "alternatives": alternatives
        }
    
    async def _generate_uncertainty_analysis(self, decision_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate uncertainty analysis explanation"""
        uncertainty_sources = context.get("uncertainty_sources", [])
        
        # Simulate uncertainty analysis
        uncertainty_content = "Uncertainty analysis identified the following sources of uncertainty: "
        for source in uncertainty_sources:
            uncertainty_content += f"{source}, "
        
        return {
            "content": uncertainty_content,
            "confidence": 0.70,
            "factors": [{"source": source, "impact": "uncertainty"} for source in uncertainty_sources],
            "alternatives": []
        }
    
    async def _log_audit_event(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Log an audit event"""
        event_type = AuditEventType(message.get("event_type", "decision_made"))
        component = message.get("component", "unknown")
        action = message.get("action", "unknown")
        user_id = message.get("user_id")
        details = message.get("details", {})
        severity = message.get("severity", "info")
        
        # Get audit logger
        logger_func = self.audit_loggers.get(event_type)
        if not logger_func:
            raise ValueError(f"Unknown event type: {event_type}")
        
        # Log audit event
        event_id = f"audit_{uuid.uuid4().hex[:8]}"
        audit_event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            component=component,
            action=action,
            user_id=user_id,
            timestamp=datetime.now(),
            details=details,
            severity=severity,
            metadata=message
        )
        
        # Store audit event
        self.audit_events.append(audit_event)
        
        # Update statistics
        self.audit_stats["audit_events_logged"] += 1
        
        return {
            "event_id": event_id,
            "event_type": event_type.value,
            "component": component,
            "action": action,
            "timestamp": audit_event.timestamp.isoformat(),
            "severity": severity
        }
    
    async def _log_decision_event(self, event_id: str, event_type: AuditEventType, component: str, action: str, user_id: Optional[str], details: Dict[str, Any], severity: str, metadata: Dict[str, Any]) -> None:
        """Log decision event"""
        self.logger.info(f"Decision event logged: {event_id} - {action}")
    
    async def _log_model_access(self, event_id: str, event_type: AuditEventType, component: str, action: str, user_id: Optional[str], details: Dict[str, Any], severity: str, metadata: Dict[str, Any]) -> None:
        """Log model access event"""
        self.logger.info(f"Model access event logged: {event_id} - {action}")
    
    async def _log_data_access(self, event_id: str, event_type: AuditEventType, component: str, action: str, user_id: Optional[str], details: Dict[str, Any], severity: str, metadata: Dict[str, Any]) -> None:
        """Log data access event"""
        self.logger.info(f"Data access event logged: {event_id} - {action}")
    
    async def _log_config_change(self, event_id: str, event_type: AuditEventType, component: str, action: str, user_id: Optional[str], details: Dict[str, Any], severity: str, metadata: Dict[str, Any]) -> None:
        """Log configuration change event"""
        self.logger.info(f"Configuration change event logged: {event_id} - {action}")
    
    async def _log_error_event(self, event_id: str, event_type: AuditEventType, component: str, action: str, user_id: Optional[str], details: Dict[str, Any], severity: str, metadata: Dict[str, Any]) -> None:
        """Log error event"""
        self.logger.warning(f"Error event logged: {event_id} - {action}")
    
    async def _log_performance_metric(self, event_id: str, event_type: AuditEventType, component: str, action: str, user_id: Optional[str], details: Dict[str, Any], severity: str, metadata: Dict[str, Any]) -> None:
        """Log performance metric event"""
        self.logger.info(f"Performance metric event logged: {event_id} - {action}")
    
    async def _generate_transparency_report(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a transparency report"""
        component = message.get("component", "system")
        transparency_level = TransparencyLevel(message.get("transparency_level", "detailed"))
        
        # Get relevant explanations and audit events
        component_explanations = [
            exp for exp in self.explanations.values()
            if exp.metadata.get("component") == component
        ]
        
        component_audit_events = [
            event for event in self.audit_events
            if event.component == component
        ]
        
        # Generate report
        report_id = f"transparency_report_{uuid.uuid4().hex[:8]}"
        report = TransparencyReport(
            report_id=report_id,
            component=component,
            transparency_level=transparency_level,
            explanations=component_explanations,
            audit_events=component_audit_events,
            performance_metrics={
                "explanations_count": len(component_explanations),
                "audit_events_count": len(component_audit_events),
                "transparency_level": transparency_level.value
            },
            data_usage={
                "data_sources": ["internal", "external"],
                "data_types": ["text", "numerical", "categorical"],
                "privacy_compliance": "GDPR compliant"
            },
            model_info={
                "model_type": "transformer",
                "version": "1.0.0",
                "training_data": "diverse dataset",
                "bias_mitigation": "active"
            },
            generated_time=datetime.now(),
            metadata=message
        )
        
        # Store report
        self.transparency_reports.append(report)
        
        # Update statistics
        self.transparency_stats["transparency_reports_created"] += 1
        
        return {
            "report_id": report_id,
            "component": component,
            "transparency_level": transparency_level.value,
            "explanations_count": len(component_explanations),
            "audit_events_count": len(component_audit_events),
            "generated_time": report.generated_time.isoformat()
        }
    
    async def _get_explanations(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get explanations for a decision"""
        decision_id = message.get("decision_id", "")
        explanation_type = message.get("explanation_type")
        
        # Filter explanations
        filtered_explanations = []
        for explanation in self.explanations.values():
            if decision_id and explanation.decision_id != decision_id:
                continue
            if explanation_type and explanation.explanation_type.value != explanation_type:
                continue
            filtered_explanations.append(explanation)
        
        return {
            "explanations": [
                {
                    "explanation_id": exp.explanation_id,
                    "decision_id": exp.decision_id,
                    "explanation_type": exp.explanation_type.value,
                    "content": exp.content,
                    "confidence": exp.confidence,
                    "timestamp": exp.timestamp.isoformat()
                }
                for exp in filtered_explanations
            ],
            "count": len(filtered_explanations)
        }
    
    async def _get_audit_log(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Get audit log entries"""
        component = message.get("component")
        event_type = message.get("event_type")
        start_time = message.get("start_time")
        end_time = message.get("end_time")
        
        # Filter audit events
        filtered_events = []
        for event in self.audit_events:
            if component and event.component != component:
                continue
            if event_type and event.event_type.value != event_type:
                continue
            if start_time and event.timestamp < datetime.fromisoformat(start_time):
                continue
            if end_time and event.timestamp > datetime.fromisoformat(end_time):
                continue
            filtered_events.append(event)
        
        return {
            "audit_events": [
                {
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "component": event.component,
                    "action": event.action,
                    "timestamp": event.timestamp.isoformat(),
                    "severity": event.severity
                }
                for event in filtered_events
            ],
            "count": len(filtered_events)
        }
    
    async def get_explainability_stats(self) -> Dict[str, Any]:
        """Get explainability statistics"""
        return {
            "total_explanations": len(self.explanations),
            "total_audit_events": len(self.audit_events),
            "total_transparency_reports": len(self.transparency_reports),
            "explanation_stats": dict(self.explanation_stats),
            "audit_stats": dict(self.audit_stats),
            "transparency_stats": dict(self.transparency_stats)
        }
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent"""
        return {
            "name": "ExplainabilityAgent",
            "description": "Explainability, transparency and auditing agent with human-readable rationales and comprehensive audit logs",
            "capabilities": [
                "Decision rationale generation",
                "Feature importance analysis",
                "Confidence score explanation",
                "Alternative options generation",
                "Uncertainty analysis",
                "Comprehensive audit logging",
                "Transparency report generation",
                "Audit log retrieval"
            ],
            "status": "active",
            "explanation_types": [exp_type.value for exp_type in ExplanationType],
            "transparency_levels": [level.value for level in TransparencyLevel],
            "audit_event_types": [event_type.value for event_type in AuditEventType],
            "stats": {
                "total_explanations": len(self.explanations),
                "total_audit_events": len(self.audit_events),
                "total_reports": len(self.transparency_reports),
                "explanation_stats": dict(self.explanation_stats)
            }
        } 