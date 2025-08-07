#!/usr/bin/env python3
"""
Self-Reflection Agent for Quark AI Assistant
Performs introspection and self-analysis for continuous improvement

Part of Pillar 16: Meta-Learning & Self-Reflection
"""

import time
import json
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

@dataclass
class ReflectionSession:
    """A self-reflection session."""
    session_id: str
    start_time: float
    end_time: float
    reflection_type: str  # performance, behavior, capability, strategy
    insights: List[Dict[str, Any]]
    actions_taken: List[Dict[str, Any]]
    improvement_score: float

@dataclass
class ReflectionInsight:
    """An insight from self-reflection."""
    insight_id: str
    insight_type: str  # pattern, weakness, strength, opportunity
    description: str
    confidence: float
    actionable: bool
    priority: str  # low, medium, high, critical
    timestamp: float
    context: Dict[str, Any]

class SelfReflectionAgent:
    def __init__(self, reflection_dir: str = None):
        self.reflection_dir = reflection_dir or os.path.join(os.path.dirname(__file__), '..', 'meta_learning')
        os.makedirs(self.reflection_dir, exist_ok=True)
        
        # Reflection data
        self.reflection_sessions = []
        self.reflection_insights = []
        self.improvement_history = []
        
        # Reflection settings
        self.reflection_enabled = True
        self.auto_reflection = True
        self.reflection_interval = 3600  # 1 hour between reflections
        self.last_reflection = 0
        
        # Reflection capabilities
        self.reflection_capabilities = {
            'performance_analysis': True,
            'behavior_patterns': True,
            'capability_assessment': True,
            'strategy_evaluation': True,
            'goal_alignment': True
        }
        
        # Load existing reflection data
        self._load_reflection_data()
        
    def _load_reflection_data(self):
        """Load existing reflection data from disk."""
        try:
            # Load reflection sessions
            sessions_file = os.path.join(self.reflection_dir, 'reflection_sessions.json')
            if os.path.exists(sessions_file):
                with open(sessions_file, 'r') as f:
                    data = json.load(f)
                    self.reflection_sessions = [ReflectionSession(**s) for s in data]
            
            # Load reflection insights
            insights_file = os.path.join(self.reflection_dir, 'reflection_insights.json')
            if os.path.exists(insights_file):
                with open(insights_file, 'r') as f:
                    data = json.load(f)
                    self.reflection_insights = [ReflectionInsight(**i) for i in data]
                    
        except Exception as e:
            print(f"Warning: Could not load reflection data: {e}")
    
    def _save_reflection_data(self):
        """Save reflection data to disk."""
        try:
            # Save reflection sessions
            sessions_file = os.path.join(self.reflection_dir, 'reflection_sessions.json')
            with open(sessions_file, 'w') as f:
                json.dump([asdict(s) for s in self.reflection_sessions], f, indent=2)
            
            # Save reflection insights
            insights_file = os.path.join(self.reflection_dir, 'reflection_insights.json')
            with open(insights_file, 'w') as f:
                json.dump([asdict(i) for i in self.reflection_insights], f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save reflection data: {e}")
    
    def run_reflection_session(self, reflection_type: str = "performance", 
                             context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run a self-reflection session."""
        if not self.reflection_enabled:
            return {'status': 'disabled', 'message': 'Self-reflection is disabled'}
        
        # Check if enough time has passed since last reflection
        if time.time() - self.last_reflection < self.reflection_interval:
            return {'status': 'too_soon', 'message': 'Reflection interval not met'}
        
        session_id = f"reflection_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Perform reflection based on type
        if reflection_type == "performance":
            insights = self._reflect_on_performance(context)
        elif reflection_type == "behavior":
            insights = self._reflect_on_behavior(context)
        elif reflection_type == "capability":
            insights = self._reflect_on_capabilities(context)
        elif reflection_type == "strategy":
            insights = self._reflect_on_strategy(context)
        else:
            insights = self._reflect_on_general(context)
        
        # Generate actionable insights
        actionable_insights = [i for i in insights if i.actionable]
        
        # Calculate improvement score
        improvement_score = self._calculate_improvement_score(insights)
        
        # Create reflection session
        session = ReflectionSession(
            session_id=session_id,
            start_time=start_time,
            end_time=time.time(),
            reflection_type=reflection_type,
            insights=insights,
            actions_taken=[],
            improvement_score=improvement_score
        )
        
        self.reflection_sessions.append(session)
        self.reflection_insights.extend(insights)
        self.last_reflection = time.time()
        
        self._save_reflection_data()
        
        return {
            'status': 'completed',
            'session_id': session_id,
            'reflection_type': reflection_type,
            'insights_count': len(insights),
            'actionable_insights': len(actionable_insights),
            'improvement_score': improvement_score,
            'insights': [asdict(i) for i in insights]
        }
    
    def _reflect_on_performance(self, context: Dict[str, Any] = None) -> List[ReflectionInsight]:
        """Reflect on performance patterns and trends."""
        insights = []
        
        # Analyze recent performance data
        recent_sessions = [s for s in self.reflection_sessions 
                         if time.time() - s.end_time < 86400]  # Last 24 hours
        
        if recent_sessions:
            # Calculate performance trends
            performance_scores = [s.improvement_score for s in recent_sessions]
            avg_score = np.mean(performance_scores)
            score_trend = np.polyfit(range(len(performance_scores)), performance_scores, 1)[0]
            
            # Generate insights
            if score_trend > 0.05:
                insights.append(ReflectionInsight(
                    insight_id=f"insight_{int(time.time() * 1000)}",
                    insight_type="strength",
                    description="Performance is showing consistent improvement",
                    confidence=0.8,
                    actionable=False,
                    priority="low",
                    timestamp=time.time(),
                    context={'trend': score_trend, 'avg_score': avg_score}
                ))
            elif score_trend < -0.05:
                insights.append(ReflectionInsight(
                    insight_id=f"insight_{int(time.time() * 1000)}",
                    insight_type="weakness",
                    description="Performance is declining and needs attention",
                    confidence=0.8,
                    actionable=True,
                    priority="high",
                    timestamp=time.time(),
                    context={'trend': score_trend, 'avg_score': avg_score}
                ))
            
            # Check for performance patterns
            if np.std(performance_scores) > 0.2:
                insights.append(ReflectionInsight(
                    insight_id=f"insight_{int(time.time() * 1000)}",
                    insight_type="pattern",
                    description="Performance shows high variability",
                    confidence=0.7,
                    actionable=True,
                    priority="medium",
                    timestamp=time.time(),
                    context={'std': np.std(performance_scores)}
                ))
        
        return insights
    
    def _reflect_on_behavior(self, context: Dict[str, Any] = None) -> List[ReflectionInsight]:
        """Reflect on behavioral patterns and decision-making."""
        insights = []
        
        # Analyze decision patterns
        decision_patterns = self._analyze_decision_patterns()
        
        if decision_patterns.get('repetitive_decisions', 0) > 0.7:
            insights.append(ReflectionInsight(
                insight_id=f"insight_{int(time.time() * 1000)}",
                insight_type="pattern",
                description="High reliance on repetitive decision patterns",
                confidence=0.7,
                actionable=True,
                priority="medium",
                timestamp=time.time(),
                context={'repetitive_ratio': decision_patterns['repetitive_decisions']}
            ))
        
        if decision_patterns.get('adaptive_decisions', 0) < 0.3:
            insights.append(ReflectionInsight(
                insight_id=f"insight_{int(time.time() * 1000)}",
                insight_type="weakness",
                description="Low adaptive decision-making capability",
                confidence=0.6,
                actionable=True,
                priority="high",
                timestamp=time.time(),
                context={'adaptive_ratio': decision_patterns['adaptive_decisions']}
            ))
        
        return insights
    
    def _reflect_on_capabilities(self, context: Dict[str, Any] = None) -> List[ReflectionInsight]:
        """Reflect on current capabilities and potential improvements."""
        insights = []
        
        # Assess current capabilities
        capability_assessment = self._assess_capabilities()
        
        for capability, assessment in capability_assessment.items():
            if assessment['strength'] < 0.5:
                insights.append(ReflectionInsight(
                    insight_id=f"insight_{int(time.time() * 1000)}",
                    insight_type="weakness",
                    description=f"Capability '{capability}' needs improvement",
                    confidence=0.7,
                    actionable=True,
                    priority="medium",
                    timestamp=time.time(),
                    context={'capability': capability, 'strength': assessment['strength']}
                ))
            elif assessment['strength'] > 0.8:
                insights.append(ReflectionInsight(
                    insight_id=f"insight_{int(time.time() * 1000)}",
                    insight_type="strength",
                    description=f"Capability '{capability}' is well-developed",
                    confidence=0.8,
                    actionable=False,
                    priority="low",
                    timestamp=time.time(),
                    context={'capability': capability, 'strength': assessment['strength']}
                ))
        
        return insights
    
    def _reflect_on_strategy(self, context: Dict[str, Any] = None) -> List[ReflectionInsight]:
        """Reflect on strategic approach and effectiveness."""
        insights = []
        
        # Analyze strategic effectiveness
        strategy_effectiveness = self._analyze_strategy_effectiveness()
        
        if strategy_effectiveness.get('goal_alignment', 0) < 0.7:
            insights.append(ReflectionInsight(
                insight_id=f"insight_{int(time.time() * 1000)}",
                insight_type="weakness",
                description="Strategy may not be well-aligned with goals",
                confidence=0.6,
                actionable=True,
                priority="high",
                timestamp=time.time(),
                context={'goal_alignment': strategy_effectiveness['goal_alignment']}
            ))
        
        if strategy_effectiveness.get('adaptability', 0) < 0.5:
            insights.append(ReflectionInsight(
                insight_id=f"insight_{int(time.time() * 1000)}",
                insight_type="weakness",
                description="Strategy lacks adaptability to changing conditions",
                confidence=0.7,
                actionable=True,
                priority="medium",
                timestamp=time.time(),
                context={'adaptability': strategy_effectiveness['adaptability']}
            ))
        
        return insights
    
    def _reflect_on_general(self, context: Dict[str, Any] = None) -> List[ReflectionInsight]:
        """General reflection combining multiple aspects."""
        insights = []
        
        # Combine insights from different reflection types
        performance_insights = self._reflect_on_performance(context)
        behavior_insights = self._reflect_on_behavior(context)
        capability_insights = self._reflect_on_capabilities(context)
        strategy_insights = self._reflect_on_strategy(context)
        
        insights.extend(performance_insights)
        insights.extend(behavior_insights)
        insights.extend(capability_insights)
        insights.extend(strategy_insights)
        
        return insights
    
    def _analyze_decision_patterns(self) -> Dict[str, float]:
        """Analyze decision-making patterns."""
        # Placeholder implementation
        return {
            'repetitive_decisions': 0.6,
            'adaptive_decisions': 0.4,
            'novel_decisions': 0.2
        }
    
    def _assess_capabilities(self) -> Dict[str, Dict[str, float]]:
        """Assess current capabilities."""
        # Placeholder implementation
        return {
            'reasoning': {'strength': 0.7, 'confidence': 0.8},
            'planning': {'strength': 0.6, 'confidence': 0.7},
            'learning': {'strength': 0.8, 'confidence': 0.9},
            'adaptation': {'strength': 0.5, 'confidence': 0.6}
        }
    
    def _analyze_strategy_effectiveness(self) -> Dict[str, float]:
        """Analyze strategic effectiveness."""
        # Placeholder implementation
        return {
            'goal_alignment': 0.7,
            'adaptability': 0.6,
            'efficiency': 0.8,
            'effectiveness': 0.7
        }
    
    def _calculate_improvement_score(self, insights: List[ReflectionInsight]) -> float:
        """Calculate improvement score based on insights."""
        if not insights:
            return 0.0
        
        # Weight insights by priority and confidence
        total_weight = 0.0
        weighted_score = 0.0
        
        priority_weights = {
            'low': 1.0,
            'medium': 2.0,
            'high': 3.0,
            'critical': 4.0
        }
        
        for insight in insights:
            weight = priority_weights.get(insight.priority, 1.0) * insight.confidence
            
            # Positive score for strengths, negative for weaknesses
            if insight.insight_type == 'strength':
                score = 0.1
            elif insight.insight_type == 'weakness':
                score = -0.1
            elif insight.insight_type == 'opportunity':
                score = 0.05
            else:
                score = 0.0
            
            weighted_score += score * weight
            total_weight += weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def get_reflection_statistics(self) -> Dict[str, Any]:
        """Get reflection statistics."""
        return {
            'total_sessions': len(self.reflection_sessions),
            'total_insights': len(self.reflection_insights),
            'actionable_insights': len([i for i in self.reflection_insights if i.actionable]),
            'reflection_enabled': self.reflection_enabled,
            'auto_reflection': self.auto_reflection,
            'last_reflection': self.last_reflection,
            'reflection_capabilities': self.reflection_capabilities,
            'recent_improvement_score': self._get_recent_improvement_score()
        }
    
    def _get_recent_improvement_score(self) -> float:
        """Get the most recent improvement score."""
        if not self.reflection_sessions:
            return 0.0
        
        recent_sessions = sorted(self.reflection_sessions, key=lambda x: x.end_time, reverse=True)
        return recent_sessions[0].improvement_score if recent_sessions else 0.0 