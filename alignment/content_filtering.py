#!/usr/bin/env python3
"""
Content Filtering for Quark AI Assistant
============================================

Implements comprehensive content filtering to prevent harmful, inappropriate,
or unsafe content from being generated or processed.

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


class ContentCategory(Enum):
    """Categories of content that can be filtered."""
    HARM = "harm"
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    SEXUAL_CONTENT = "sexual_content"
    MISINFORMATION = "misinformation"
    PERSONAL_INFO = "personal_info"
    ILLEGAL_ACTIVITY = "illegal_activity"
    SELF_HARM = "self_harm"
    SPAM = "spam"
    TOXIC = "toxic"


class FilterSeverity(Enum):
    """Severity levels for content filtering."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FilterResult:
    """Result of content filtering."""
    is_safe: bool
    categories: List[ContentCategory]
    severity: FilterSeverity
    confidence: float
    flagged_terms: List[str]
    explanation: str
    timestamp: float


class ContentFilter:
    """Comprehensive content filtering system."""
    
    def __init__(self, filter_dir: str = None):
        self.filter_dir = filter_dir or os.path.join(os.path.dirname(__file__), '..', 'alignment', 'content_filters')
        os.makedirs(self.filter_dir, exist_ok=True)
        
        # Filter settings
        self.filtering_enabled = True
        self.block_threshold = 0.7
        self.warn_threshold = 0.5
        self.log_all_checks = True
        
        # Initialize safety guardrails
        self.safety_guardrails = SafetyGuardrails()
        
        # Load filter patterns
        self._load_filter_patterns()
        
        # Statistics
        self.filter_stats = {
            'total_checks': 0,
            'blocked_content': 0,
            'warned_content': 0,
            'safe_content': 0
        }
        
    def _load_filter_patterns(self):
        """Load filter patterns from configuration."""
        self.harm_patterns = [
            r'\b(kill|murder|assassinate|execute)\b',
            r'\b(bomb|explosive|weapon)\b',
            r'\b(harm|hurt|injure|damage)\b',
            r'\b(suicide|self-harm|end.*life)\b'
        ]
        
        self.violence_patterns = [
            r'\b(fight|attack|assault|battle)\b',
            r'\b(weapon|gun|knife|sword)\b',
            r'\b(violence|aggressive|hostile)\b'
        ]
        
        self.hate_speech_patterns = [
            r'\b(hate|racist|sexist|discriminate)\b',
            r'\b(superior|inferior|slave|master)\b',
            r'\b(exterminate|eliminate|purge)\b'
        ]
        
        self.misinformation_patterns = [
            r'\b(conspiracy|hoax|fake.*news)\b',
            r'\b(government.*cover.*up)\b',
            r'\b(secret.*truth.*hidden)\b'
        ]
        
        self.personal_info_patterns = [
            r'\b(ssn|social.*security)\b',
            r'\b(credit.*card|bank.*account)\b',
            r'\b(phone.*number|address)\b',
            r'\b(password|pin|secret)\b'
        ]
        
    def filter_content(self, content: str, context: Dict[str, Any] = None) -> FilterResult:
        """
        Filter content for safety and appropriateness.
        
        Args:
            content: The content to filter
            context: Additional context for filtering
            
        Returns:
            FilterResult with safety assessment
        """
        if not self.filtering_enabled:
            return FilterResult(
                is_safe=True,
                categories=[],
                severity=FilterSeverity.LOW,
                confidence=1.0,
                flagged_terms=[],
                explanation="Filtering disabled",
                timestamp=time.time()
            )
        
        # Update statistics
        self.filter_stats['total_checks'] += 1
        
        # Check for harmful content
        categories = []
        flagged_terms = []
        max_severity = FilterSeverity.LOW
        confidence = 0.0
        
        # Check each category
        checks = [
            (self._check_harm, ContentCategory.HARM),
            (self._check_violence, ContentCategory.VIOLENCE),
            (self._check_hate_speech, ContentCategory.HATE_SPEECH),
            (self._check_misinformation, ContentCategory.MISINFORMATION),
            (self._check_personal_info, ContentCategory.PERSONAL_INFO)
        ]
        
        for check_func, category in checks:
            result = check_func(content)
            if result['detected']:
                categories.append(category)
                flagged_terms.extend(result['terms'])
                if result['severity'].value > max_severity.value:
                    max_severity = result['severity']
                confidence = max(confidence, result['confidence'])
        
        # Determine if content is safe
        is_safe = confidence < self.block_threshold
        
        # Update statistics
        if not is_safe:
            self.filter_stats['blocked_content'] += 1
        elif confidence > self.warn_threshold:
            self.filter_stats['warned_content'] += 1
        else:
            self.filter_stats['safe_content'] += 1
        
        # Generate explanation
        explanation = self._generate_explanation(categories, flagged_terms, confidence)
        
        return FilterResult(
            is_safe=is_safe,
            categories=categories,
            severity=max_severity,
            confidence=confidence,
            flagged_terms=flagged_terms,
            explanation=explanation,
            timestamp=time.time()
        )
    
    def _check_harm(self, content: str) -> Dict[str, Any]:
        """Check for harmful content."""
        import re
        
        detected_terms = []
        for pattern in self.harm_patterns:
            matches = re.findall(pattern, content.lower())
            detected_terms.extend(matches)
        
        if detected_terms:
            return {
                'detected': True,
                'terms': detected_terms,
                'severity': FilterSeverity.HIGH,
                'confidence': min(0.8, len(detected_terms) * 0.2)
            }
        
        return {'detected': False, 'terms': [], 'severity': FilterSeverity.LOW, 'confidence': 0.0}
    
    def _check_violence(self, content: str) -> Dict[str, Any]:
        """Check for violent content."""
        import re
        
        detected_terms = []
        for pattern in self.violence_patterns:
            matches = re.findall(pattern, content.lower())
            detected_terms.extend(matches)
        
        if detected_terms:
            return {
                'detected': True,
                'terms': detected_terms,
                'severity': FilterSeverity.MEDIUM,
                'confidence': min(0.7, len(detected_terms) * 0.15)
            }
        
        return {'detected': False, 'terms': [], 'severity': FilterSeverity.LOW, 'confidence': 0.0}
    
    def _check_hate_speech(self, content: str) -> Dict[str, Any]:
        """Check for hate speech."""
        import re
        
        detected_terms = []
        for pattern in self.hate_speech_patterns:
            matches = re.findall(pattern, content.lower())
            detected_terms.extend(matches)
        
        if detected_terms:
            return {
                'detected': True,
                'terms': detected_terms,
                'severity': FilterSeverity.HIGH,
                'confidence': min(0.8, len(detected_terms) * 0.2)
            }
        
        return {'detected': False, 'terms': [], 'severity': FilterSeverity.LOW, 'confidence': 0.0}
    
    def _check_misinformation(self, content: str) -> Dict[str, Any]:
        """Check for potential misinformation."""
        import re
        
        detected_terms = []
        for pattern in self.misinformation_patterns:
            matches = re.findall(pattern, content.lower())
            detected_terms.extend(matches)
        
        if detected_terms:
            return {
                'detected': True,
                'terms': detected_terms,
                'severity': FilterSeverity.MEDIUM,
                'confidence': min(0.6, len(detected_terms) * 0.15)
            }
        
        return {'detected': False, 'terms': [], 'severity': FilterSeverity.LOW, 'confidence': 0.0}
    
    def _check_personal_info(self, content: str) -> Dict[str, Any]:
        """Check for personal information."""
        import re
        
        detected_terms = []
        for pattern in self.personal_info_patterns:
            matches = re.findall(pattern, content.lower())
            detected_terms.extend(matches)
        
        if detected_terms:
            return {
                'detected': True,
                'terms': detected_terms,
                'severity': FilterSeverity.CRITICAL,
                'confidence': min(0.9, len(detected_terms) * 0.25)
            }
        
        return {'detected': False, 'terms': [], 'severity': FilterSeverity.LOW, 'confidence': 0.0}
    
    def _generate_explanation(self, categories: List[ContentCategory], 
                             flagged_terms: List[str], confidence: float) -> str:
        """Generate explanation for filtering decision."""
        if not categories:
            return "Content appears safe"
        
        category_names = [cat.value.replace('_', ' ').title() for cat in categories]
        explanation = f"Content flagged for: {', '.join(category_names)}"
        
        if flagged_terms:
            explanation += f" (terms: {', '.join(flagged_terms[:3])})"
        
        explanation += f" (confidence: {confidence:.2f})"
        
        return explanation
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get filtering statistics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'filtering_enabled': self.filtering_enabled,
            'total_checks': self.filter_stats['total_checks'],
            'blocked_content': self.filter_stats['blocked_content'],
            'warned_content': self.filter_stats['warned_content'],
            'safe_content': self.filter_stats['safe_content'],
            'block_rate': (self.filter_stats['blocked_content'] / max(1, self.filter_stats['total_checks'])),
            'warn_rate': (self.filter_stats['warned_content'] / max(1, self.filter_stats['total_checks']))
        }
    
    def export_filter_data(self, filename: str = None) -> str:
        """Export filter data to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"filter_data_{timestamp}.json"
        
        filepath = os.path.join(self.filter_dir, filename)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'filter_stats': self.get_filter_stats(),
            'filter_settings': {
                'filtering_enabled': self.filtering_enabled,
                'block_threshold': self.block_threshold,
                'warn_threshold': self.warn_threshold
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath 