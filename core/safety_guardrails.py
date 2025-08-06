#!/usr/bin/env python3
"""
Safety Guardrails for Meta-Model AI Assistant
Ensures safe self-improvement with user confirmation for significant changes
"""

import time
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class ChangeSeverity(Enum):
    """Severity levels for changes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ChangeType(Enum):
    """Types of changes that can be made."""
    MODEL_FINE_TUNING = "model_fine_tuning"
    CAPABILITY_LEARNING = "capability_learning"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    PIPELINE_MODIFICATION = "pipeline_modification"
    MEMORY_CLEANUP = "memory_cleanup"
    CONFIGURATION_CHANGE = "configuration_change"

@dataclass
class SafetyCheck:
    """A safety check for a proposed change."""
    change_id: str
    change_type: ChangeType
    severity: ChangeSeverity
    description: str
    impact_analysis: Dict[str, Any]
    requires_confirmation: bool
    auto_approve_threshold: float
    timestamp: float
    status: str  # pending, approved, rejected, expired

class SafetyGuardrails:
    def __init__(self, safety_dir: str = None):
        self.safety_dir = safety_dir or os.path.join(os.path.dirname(__file__), '..', 'safety')
        os.makedirs(self.safety_dir, exist_ok=True)
        
        # Safety settings
        self.safety_enabled = True
        self.auto_approve_low_risk = True
        self.require_confirmation_high_risk = True
        self.max_changes_per_hour = 5
        self.change_timeout_hours = 24
        
        # Change tracking
        self.pending_changes = {}
        self.approved_changes = []
        self.rejected_changes = []
        
        # Safety thresholds
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.8,
            'critical': 0.9
        }
        
    def propose_change(self, change_type: ChangeType, description: str, 
                      impact_analysis: Dict[str, Any], severity: ChangeSeverity = ChangeSeverity.MEDIUM) -> str:
        """Propose a change that requires safety review."""
        if not self.safety_enabled:
            return "safety_disabled"
        
        change_id = f"change_{int(time.time() * 1000)}"
        
        # Check rate limiting
        recent_changes = self._get_recent_changes(3600)  # Last hour
        if len(recent_changes) >= self.max_changes_per_hour:
            return "rate_limited"
        
        # Create safety check
        safety_check = SafetyCheck(
            change_id=change_id,
            change_type=change_type,
            severity=severity,
            description=description,
            impact_analysis=impact_analysis,
            requires_confirmation=self._requires_confirmation(severity),
            auto_approve_threshold=self.risk_thresholds.get(severity.value, 0.5),
            timestamp=time.time(),
            status="pending"
        )
        
        self.pending_changes[change_id] = safety_check
        
        return change_id
        
    def _requires_confirmation(self, severity: ChangeSeverity) -> bool:
        """Determine if a change requires user confirmation."""
        if severity in [ChangeSeverity.HIGH, ChangeSeverity.CRITICAL]:
            return True
        elif severity == ChangeSeverity.MEDIUM and self.require_confirmation_high_risk:
            return True
        return False
        
    def _get_recent_changes(self, time_window: int) -> List[SafetyCheck]:
        """Get changes from the last time_window seconds."""
        cutoff_time = time.time() - time_window
        recent = []
        
        for change in self.approved_changes + self.rejected_changes:
            if change.timestamp >= cutoff_time:
                recent.append(change)
        
        return recent
        
    def approve_change(self, change_id: str, user_confirmation: bool = False) -> Dict[str, Any]:
        """Approve a pending change."""
        if change_id not in self.pending_changes:
            return {'status': 'error', 'message': 'Change not found'}
        
        safety_check = self.pending_changes[change_id]
        
        # Check if change has expired
        if time.time() - safety_check.timestamp > (self.change_timeout_hours * 3600):
            safety_check.status = "expired"
            return {'status': 'expired', 'message': 'Change has expired'}
        
        # Check if user confirmation is required but not provided
        if safety_check.requires_confirmation and not user_confirmation:
            return {'status': 'confirmation_required', 'message': 'User confirmation required'}
        
        # Approve the change
        safety_check.status = "approved"
        self.approved_changes.append(safety_check)
        del self.pending_changes[change_id]
        
        return {
            'status': 'approved',
            'change_id': change_id,
            'change_type': safety_check.change_type.value,
            'severity': safety_check.severity.value
        }
        
    def reject_change(self, change_id: str, reason: str = "") -> Dict[str, Any]:
        """Reject a pending change."""
        if change_id not in self.pending_changes:
            return {'status': 'error', 'message': 'Change not found'}
        
        safety_check = self.pending_changes[change_id]
        safety_check.status = "rejected"
        self.rejected_changes.append(safety_check)
        del self.pending_changes[change_id]
        
        return {
            'status': 'rejected',
            'change_id': change_id,
            'reason': reason
        }
        
    def get_pending_changes(self) -> List[Dict[str, Any]]:
        """Get all pending changes that require review."""
        changes = []
        
        for change_id, safety_check in self.pending_changes.items():
            # Check if change has expired
            if time.time() - safety_check.timestamp > (self.change_timeout_hours * 3600):
                safety_check.status = "expired"
                continue
            
            changes.append({
                'change_id': change_id,
                'change_type': safety_check.change_type.value,
                'severity': safety_check.severity.value,
                'description': safety_check.description,
                'impact_analysis': safety_check.impact_analysis,
                'requires_confirmation': safety_check.requires_confirmation,
                'timestamp': safety_check.timestamp,
                'age_hours': (time.time() - safety_check.timestamp) / 3600
            })
        
        return changes
        
    def assess_change_risk(self, change_type: ChangeType, impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the risk of a proposed change."""
        risk_score = 0.0
        risk_factors = []
        
        # Risk assessment based on change type
        if change_type == ChangeType.MODEL_FINE_TUNING:
            risk_score += 0.4
            risk_factors.append("Model fine-tuning can affect core behavior")
        elif change_type == ChangeType.PIPELINE_MODIFICATION:
            risk_score += 0.3
            risk_factors.append("Pipeline changes affect processing flow")
        elif change_type == ChangeType.CAPABILITY_LEARNING:
            risk_score += 0.2
            risk_factors.append("New capabilities may introduce unexpected behavior")
        
        # Risk assessment based on impact
        if impact_analysis.get('affects_core_functionality', False):
            risk_score += 0.3
            risk_factors.append("Affects core functionality")
        
        if impact_analysis.get('data_modification', False):
            risk_score += 0.2
            risk_factors.append("Modifies stored data")
        
        if impact_analysis.get('performance_impact', 'none') == 'high':
            risk_score += 0.2
            risk_factors.append("High performance impact")
        
        if impact_analysis.get('user_experience_impact', 'none') == 'high':
            risk_score += 0.2
            risk_factors.append("High user experience impact")
        
        # Determine severity
        if risk_score >= self.risk_thresholds['critical']:
            severity = ChangeSeverity.CRITICAL
        elif risk_score >= self.risk_thresholds['high']:
            severity = ChangeSeverity.HIGH
        elif risk_score >= self.risk_thresholds['medium']:
            severity = ChangeSeverity.MEDIUM
        else:
            severity = ChangeSeverity.LOW
        
        return {
            'risk_score': risk_score,
            'severity': severity.value,
            'risk_factors': risk_factors,
            'requires_confirmation': self._requires_confirmation(severity),
            'recommendation': 'approve' if risk_score < 0.5 else 'review'
        }
        
    def generate_safety_report(self) -> Dict[str, Any]:
        """Generate a comprehensive safety report."""
        return {
            'timestamp': datetime.now().isoformat(),
            'safety_enabled': self.safety_enabled,
            'pending_changes': len(self.pending_changes),
            'approved_changes_24h': len([c for c in self.approved_changes if time.time() - c.timestamp < 86400]),
            'rejected_changes_24h': len([c for c in self.rejected_changes if time.time() - c.timestamp < 86400]),
            'change_rate': len(self._get_recent_changes(3600)),  # Changes per hour
            'safety_settings': {
                'auto_approve_low_risk': self.auto_approve_low_risk,
                'require_confirmation_high_risk': self.require_confirmation_high_risk,
                'max_changes_per_hour': self.max_changes_per_hour,
                'change_timeout_hours': self.change_timeout_hours
            }
        }
        
    def export_safety_data(self, filename: str = None) -> str:
        """Export safety data to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"safety_data_{timestamp}.json"
        
        filepath = os.path.join(self.safety_dir, filename)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'pending_changes': {cid: asdict(change) for cid, change in self.pending_changes.items()},
            'approved_changes': [asdict(change) for change in self.approved_changes],
            'rejected_changes': [asdict(change) for change in self.rejected_changes],
            'safety_report': self.generate_safety_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath 