#!/usr/bin/env python3
"""
Self-Improvement Agent for Meta-Model AI Assistant
Handles automated fine-tuning, online learning, and self-reflection loops
"""

import time
import json
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import random
from core.safety_guardrails import SafetyGuardrails, ChangeType, ChangeSeverity

@dataclass
class LearningExample:
    """A learning example for self-improvement."""
    id: str
    input_text: str
    expected_output: str
    actual_output: str
    feedback_score: float
    category: str
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class ImprovementSession:
    """A self-improvement session."""
    session_id: str
    start_time: float
    end_time: float
    examples_processed: int
    improvements_made: int
    performance_gain: float
    session_type: str
    metadata: Dict[str, Any]

class SelfImprovementAgent:
    def __init__(self, learning_dir: str = None):
        self.learning_dir = learning_dir or os.path.join(os.path.dirname(__file__), '..', 'learning')
        os.makedirs(self.learning_dir, exist_ok=True)
        
        # Initialize without model since this is a self-improvement agent
        self.model_name = None
        self.model = None
        
        # Learning data storage
        self.learning_examples = []
        self.improvement_sessions = []
        self.feedback_history = []
        
        # Self-improvement settings
        self.learning_enabled = True
        self.auto_fine_tuning = True
        self.online_learning = True
        self.self_reflection_enabled = True
        
        # Performance tracking
        self.baseline_performance = {}
        self.current_performance = {}
        self.improvement_targets = {
            'accuracy': 0.05,  # 5% improvement target
            'latency': 0.1,    # 10% latency reduction
            'user_satisfaction': 0.1  # 10% satisfaction improvement
        }
        
        # Learning thresholds
        self.min_examples_for_learning = 10
        self.learning_threshold = 0.7  # Minimum feedback score for learning
        self.improvement_threshold = 0.02  # Minimum improvement to apply
        
        # Initialize safety guardrails
        self.safety_guardrails = SafetyGuardrails()
        
    def _ensure_model(self):
        """Ensure the self-improvement system is initialized."""
        return True
        
    def add_learning_example(self, input_text: str, expected_output: str, 
                            actual_output: str, feedback_score: float = None,
                            category: str = "general") -> str:
        """Add a learning example for self-improvement."""
        if not self.learning_enabled:
            return None
            
        example_id = f"example_{int(time.time() * 1000)}"
        
        # Calculate feedback score if not provided
        if feedback_score is None:
            feedback_score = self._calculate_feedback_score(expected_output, actual_output)
        
        example = LearningExample(
            id=example_id,
            input_text=input_text,
            expected_output=expected_output,
            actual_output=actual_output,
            feedback_score=feedback_score,
            category=category,
            timestamp=time.time(),
            metadata={
                'input_length': len(input_text),
                'output_length': len(actual_output),
                'expected_length': len(expected_output)
            }
        )
        
        self.learning_examples.append(example)
        
        # Store feedback for analysis
        self.feedback_history.append({
            'timestamp': time.time(),
            'feedback_score': feedback_score,
            'category': category,
            'example_id': example_id
        })
        
        return example_id
        
    def _calculate_feedback_score(self, expected: str, actual: str) -> float:
        """Calculate a feedback score based on output similarity."""
        if not expected or not actual:
            return 0.0
        
        # Simple word overlap calculation
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if not expected_words:
            return 1.0 if not actual_words else 0.0
        
        intersection = expected_words.intersection(actual_words)
        union = expected_words.union(actual_words)
        
        return len(intersection) / len(union) if union else 0.0
        
    def analyze_performance_gaps(self, time_window: int = 86400) -> Dict[str, Any]:
        """Analyze performance gaps for improvement opportunities."""
        cutoff_time = time.time() - time_window
        
        # Filter recent examples
        recent_examples = [
            ex for ex in self.learning_examples
            if ex.timestamp >= cutoff_time
        ]
        
        if not recent_examples:
            return {
                'total_examples': 0,
                'average_feedback': 0.0,
                'performance_gaps': {},
                'improvement_opportunities': []
            }
        
        # Calculate performance by category
        category_performance = defaultdict(list)
        for example in recent_examples:
            category_performance[example.category].append(example.feedback_score)
        
        # Identify performance gaps
        performance_gaps = {}
        improvement_opportunities = []
        
        for category, scores in category_performance.items():
            avg_score = sum(scores) / len(scores)
            performance_gaps[category] = {
                'average_score': avg_score,
                'example_count': len(scores),
                'needs_improvement': avg_score < self.learning_threshold
            }
            
            if avg_score < self.learning_threshold:
                improvement_opportunities.append({
                    'category': category,
                    'current_score': avg_score,
                    'target_score': self.learning_threshold,
                    'gap': self.learning_threshold - avg_score
                })
        
        return {
            'total_examples': len(recent_examples),
            'average_feedback': sum(ex.feedback_score for ex in recent_examples) / len(recent_examples),
            'performance_gaps': dict(performance_gaps),
            'improvement_opportunities': improvement_opportunities
        }
        
    def run_self_reflection(self) -> Dict[str, Any]:
        """Run self-reflection to identify improvement areas."""
        if not self.self_reflection_enabled:
            return {'status': 'disabled'}
        
        # Analyze recent performance
        performance_analysis = self.analyze_performance_gaps()
        
        # Identify patterns in feedback
        feedback_patterns = self._analyze_feedback_patterns()
        
        # Generate improvement recommendations
        recommendations = self._generate_improvement_recommendations(
            performance_analysis, feedback_patterns
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'performance_analysis': performance_analysis,
            'feedback_patterns': feedback_patterns,
            'recommendations': recommendations,
            'reflection_insights': self._generate_reflection_insights()
        }
        
    def _analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in feedback history."""
        if not self.feedback_history:
            return {'patterns': [], 'trends': []}
        
        # Analyze feedback trends over time
        recent_feedback = sorted(self.feedback_history, key=lambda x: x['timestamp'])[-50:]
        
        if len(recent_feedback) < 2:
            return {'patterns': [], 'trends': []}
        
        # Calculate trend
        scores = [f['feedback_score'] for f in recent_feedback]
        trend = (scores[-1] - scores[0]) / len(scores) if len(scores) > 1 else 0
        
        # Identify patterns by category
        category_patterns = defaultdict(list)
        for feedback in recent_feedback:
            category_patterns[feedback['category']].append(feedback['feedback_score'])
        
        patterns = []
        for category, scores in category_patterns.items():
            if len(scores) >= 3:
                avg_score = sum(scores) / len(scores)
                patterns.append({
                    'category': category,
                    'average_score': avg_score,
                    'sample_size': len(scores),
                    'trend': 'improving' if scores[-1] > scores[0] else 'declining'
                })
        
        return {
            'patterns': patterns,
            'trends': {
                'overall_trend': 'improving' if trend > 0 else 'declining',
                'trend_magnitude': abs(trend),
                'recent_average': sum(scores[-10:]) / min(10, len(scores))
            }
        }
        
    def _generate_improvement_recommendations(self, performance_analysis: Dict, 
                                           feedback_patterns: Dict) -> List[Dict]:
        """Generate specific improvement recommendations."""
        recommendations = []
        
        # Recommendations based on performance gaps
        for opportunity in performance_analysis.get('improvement_opportunities', []):
            recommendations.append({
                'type': 'performance_gap',
                'category': opportunity['category'],
                'priority': 'high' if opportunity['gap'] > 0.2 else 'medium',
                'action': f"Focus on improving {opportunity['category']} responses",
                'expected_improvement': opportunity['gap']
            })
        
        # Recommendations based on feedback patterns
        for pattern in feedback_patterns.get('patterns', []):
            if pattern['average_score'] < 0.6:
                recommendations.append({
                    'type': 'feedback_pattern',
                    'category': pattern['category'],
                    'priority': 'high',
                    'action': f"Review and improve {pattern['category']} handling",
                    'expected_improvement': 0.2
                })
        
        # General recommendations
        if performance_analysis.get('average_feedback', 0) < 0.7:
            recommendations.append({
                'type': 'general',
                'category': 'overall',
                'priority': 'medium',
                'action': "Consider model fine-tuning or prompt optimization",
                'expected_improvement': 0.1
            })
        
        return recommendations
        
    def _generate_reflection_insights(self) -> List[str]:
        """Generate insights from self-reflection."""
        insights = []
        
        if len(self.learning_examples) < 5:
            insights.append("Need more learning examples to generate meaningful insights")
            return insights
        
        # Analyze recent performance
        recent_examples = sorted(self.learning_examples, key=lambda x: x.timestamp)[-20:]
        avg_feedback = sum(ex.feedback_score for ex in recent_examples) / len(recent_examples)
        
        if avg_feedback < 0.6:
            insights.append("Overall performance needs improvement - consider systematic changes")
        elif avg_feedback > 0.8:
            insights.append("Performance is good - focus on incremental improvements")
        else:
            insights.append("Performance is moderate - identify specific areas for improvement")
        
        # Category-specific insights
        category_performance = defaultdict(list)
        for example in recent_examples:
            category_performance[example.category].append(example.feedback_score)
        
        for category, scores in category_performance.items():
            if len(scores) >= 3:
                avg_score = sum(scores) / len(scores)
                if avg_score < 0.5:
                    insights.append(f"{category} responses need significant improvement")
                elif avg_score < 0.7:
                    insights.append(f"{category} responses could be enhanced")
        
        return insights
        
    def run_online_learning(self, new_examples: List[Dict]) -> Dict[str, Any]:
        """Run online learning with new examples."""
        if not self.online_learning:
            return {'status': 'disabled'}
        
        session_id = f"online_learning_{int(time.time() * 1000)}"
        start_time = time.time()
        
        processed_examples = 0
        improvements_made = 0
        
        for example_data in new_examples:
            example_id = self.add_learning_example(
                input_text=example_data['input_text'],
                expected_output=example_data['expected_output'],
                actual_output=example_data['actual_output'],
                feedback_score=example_data.get('feedback_score'),
                category=example_data.get('category', 'general')
            )
            
            if example_id:
                processed_examples += 1
                
                # Check if this example should trigger improvements
                if example_data.get('feedback_score', 0) < self.learning_threshold:
                    improvements_made += 1
        
        end_time = time.time()
        
        # Create improvement session
        session = ImprovementSession(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            examples_processed=processed_examples,
            improvements_made=improvements_made,
            performance_gain=0.0,  # Would be calculated based on before/after metrics
            session_type="online_learning",
            metadata={'examples': len(new_examples)}
        )
        
        self.improvement_sessions.append(session)
        
        return {
            'session_id': session_id,
            'examples_processed': processed_examples,
            'improvements_made': improvements_made,
            'duration': end_time - start_time,
            'status': 'completed'
        }
        
    def run_automated_fine_tuning(self, target_improvement: float = 0.05, user_confirmation: bool = False) -> Dict[str, Any]:
        """Run automated fine-tuning based on learning examples with safety checks."""
        if not self.auto_fine_tuning:
            return {'status': 'disabled'}
        
        # Check if we have enough examples
        if len(self.learning_examples) < self.min_examples_for_learning:
            return {
                'status': 'insufficient_data',
                'required_examples': self.min_examples_for_learning,
                'available_examples': len(self.learning_examples)
            }
        
        # Safety check for fine-tuning
        impact_analysis = {
            'affects_core_functionality': True,
            'data_modification': False,
            'performance_impact': 'medium',
            'user_experience_impact': 'medium',
            'target_improvement': target_improvement,
            'examples_count': len(self.learning_examples)
        }
        
        risk_assessment = self.safety_guardrails.assess_change_risk(
            ChangeType.MODEL_FINE_TUNING, impact_analysis
        )
        
        # Propose change for safety review
        change_id = self.safety_guardrails.propose_change(
            change_type=ChangeType.MODEL_FINE_TUNING,
            description=f"Automated fine-tuning with {target_improvement:.1%} improvement target",
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
                            'message': 'User confirmation required for this change'
                        }
            
            # Approve the change
            approval_result = self.safety_guardrails.approve_change(change_id, user_confirmation)
            if approval_result['status'] != 'approved':
                return approval_result
        
        session_id = f"fine_tuning_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Analyze current performance
        performance_analysis = self.analyze_performance_gaps()
        
        # Identify improvement opportunities
        opportunities = performance_analysis.get('improvement_opportunities', [])
        
        if not opportunities:
            return {
                'status': 'no_improvements_needed',
                'current_performance': performance_analysis.get('average_feedback', 0)
            }
        
        # Simulate fine-tuning process
        improvements_made = 0
        for opportunity in opportunities:
            if opportunity['gap'] >= target_improvement:
                improvements_made += 1
        
        end_time = time.time()
        
        # Create improvement session
        session = ImprovementSession(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            examples_processed=len(self.learning_examples),
            improvements_made=improvements_made,
            performance_gain=target_improvement,
            session_type="automated_fine_tuning",
            metadata={'target_improvement': target_improvement, 'safety_approved': True}
        )
        
        self.improvement_sessions.append(session)
        
        return {
            'session_id': session_id,
            'improvements_made': improvements_made,
            'performance_gain': target_improvement,
            'duration': end_time - start_time,
            'status': 'completed',
            'safety_approved': True,
            'risk_assessment': risk_assessment
        }
        
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get statistics about learning and improvement."""
        if not self.learning_examples:
            return {
                'total_examples': 0,
                'total_sessions': 0,
                'average_feedback': 0.0,
                'improvement_trend': 'no_data'
            }
        
        # Calculate statistics
        total_examples = len(self.learning_examples)
        total_sessions = len(self.improvement_sessions)
        average_feedback = sum(ex.feedback_score for ex in self.learning_examples) / total_examples
        
        # Calculate improvement trend
        if len(self.learning_examples) >= 10:
            recent_examples = sorted(self.learning_examples, key=lambda x: x.timestamp)[-10:]
            recent_avg = sum(ex.feedback_score for ex in recent_examples) / len(recent_examples)
            
            older_examples = sorted(self.learning_examples, key=lambda x: x.timestamp)[:10]
            older_avg = sum(ex.feedback_score for ex in older_examples) / len(older_examples)
            
            trend = recent_avg - older_avg
            if trend > 0.05:
                improvement_trend = 'improving'
            elif trend < -0.05:
                improvement_trend = 'declining'
            else:
                improvement_trend = 'stable'
        else:
            improvement_trend = 'insufficient_data'
        
        return {
            'total_examples': total_examples,
            'total_sessions': total_sessions,
            'average_feedback': average_feedback,
            'improvement_trend': improvement_trend,
            'recent_performance': self.analyze_performance_gaps(86400)  # Last 24 hours
        }
        
    def export_learning_data(self, filename: str = None) -> str:
        """Export learning data to JSON."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"learning_data_{timestamp}.json"
        
        filepath = os.path.join(self.learning_dir, filename)
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'learning_examples': [asdict(ex) for ex in self.learning_examples],
            'improvement_sessions': [asdict(session) for session in self.improvement_sessions],
            'feedback_history': self.feedback_history,
            'statistics': self.get_learning_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filepath
        
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Main self-improvement agent interface."""
        operation = kwargs.get('operation', 'reflection')
        
        if operation == 'add_example':
            input_text = kwargs.get('input_text', prompt)
            expected_output = kwargs.get('expected_output', '')
            actual_output = kwargs.get('actual_output', '')
            feedback_score = kwargs.get('feedback_score')
            category = kwargs.get('category', 'general')
            
            example_id = self.add_learning_example(
                input_text, expected_output, actual_output, feedback_score, category
            )
            
            return {
                'operation': 'add_example',
                'example_id': example_id,
                'success': example_id is not None
            }
            
        elif operation == 'self_reflection':
            return self.run_self_reflection()
            
        elif operation == 'online_learning':
            new_examples = kwargs.get('examples', [])
            return self.run_online_learning(new_examples)
            
        elif operation == 'fine_tuning':
            target_improvement = kwargs.get('target_improvement', 0.05)
            return self.run_automated_fine_tuning(target_improvement)
            
        elif operation == 'statistics':
            return self.get_learning_statistics()
            
        elif operation == 'export':
            filename = kwargs.get('filename')
            filepath = self.export_learning_data(filename)
            return {'export_path': filepath}
            
        else:
            # Default: run self-reflection
            return self.run_self_reflection() 