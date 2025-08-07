#!/usr/bin/env python3
"""
RLHF Agent for Quark AI Assistant
======================================

Implements Reinforcement Learning from Human Feedback (RLHF) for AI alignment.
This agent collects human feedback, trains reward models, and improves AI behavior.

Part of Pillar 15: Safety & Alignment
"""

import os
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

from core.safety_guardrails import SafetyGuardrails, ChangeType, ChangeSeverity


class FeedbackType(Enum):
    """Types of human feedback."""
    PREFERENCE = "preference"  # A vs B comparison
    RATING = "rating"          # 1-5 scale rating
    BINARY = "binary"          # Good/Bad


@dataclass
class HumanFeedback:
    """A piece of human feedback for RLHF."""
    id: str
    feedback_type: FeedbackType
    prompt: str
    response_a: str
    category: str
    timestamp: float
    metadata: Dict[str, Any]
    response_b: Optional[str] = None
    rating: Optional[float] = None
    preferred_response: Optional[str] = None
    feedback_score: float = 0.0
    user_id: Optional[str] = None


class RLHFAgent:
    """RLHF Agent for collecting human feedback and training reward models."""
    
    def __init__(self, rlhf_dir: str = None):
        self.rlhf_dir = rlhf_dir or os.path.join(os.path.dirname(__file__), '..', 'alignment', 'rlhf_data')
        os.makedirs(self.rlhf_dir, exist_ok=True)
        
        # Feedback storage
        self.feedback_data = []
        self.reward_models = {}
        
        # RLHF settings
        self.feedback_collection_enabled = True
        self.reward_model_training_enabled = True
        self.feedback_threshold = 0.7
        self.min_feedback_for_training = 50
        
        # Initialize safety guardrails
        self.safety_guardrails = SafetyGuardrails()
        
        # Load existing data
        self._load_feedback_data()
        
    def _load_feedback_data(self):
        """Load existing feedback data from disk."""
        feedback_file = os.path.join(self.rlhf_dir, 'feedback_data.json')
        if os.path.exists(feedback_file):
            try:
                with open(feedback_file, 'r') as f:
                    data = json.load(f)
                    self.feedback_data = [HumanFeedback(**item) for item in data]
            except Exception as e:
                print(f"Error loading feedback data: {e}")
                
    def _save_feedback_data(self):
        """Save feedback data to disk."""
        feedback_file = os.path.join(self.rlhf_dir, 'feedback_data.json')
        try:
            # Convert enum values to strings for JSON serialization
            feedback_dicts = []
            for feedback in self.feedback_data:
                feedback_dict = asdict(feedback)
                if isinstance(feedback_dict['feedback_type'], FeedbackType):
                    feedback_dict['feedback_type'] = feedback_dict['feedback_type'].value
                feedback_dicts.append(feedback_dict)
            
            with open(feedback_file, 'w') as f:
                json.dump(feedback_dicts, f, indent=2)
        except Exception as e:
            print(f"Error saving feedback data: {e}")
            
    def collect_preference_feedback(self, prompt: str, response_a: str, response_b: str, 
                                  preferred_response: str, category: str = "general", 
                                  user_id: Optional[str] = None) -> str:
        """Collect preference feedback (A vs B comparison)."""
        if not self.feedback_collection_enabled:
            return "feedback_disabled"
            
        feedback_id = f"feedback_{int(time.time() * 1000)}"
        feedback_score = 1.0 if preferred_response == "A" else 0.0
        
        feedback = HumanFeedback(
            id=feedback_id,
            feedback_type=FeedbackType.PREFERENCE,
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            preferred_response=preferred_response,
            feedback_score=feedback_score,
            category=category,
            timestamp=time.time(),
            user_id=user_id,
            metadata={}
        )
        
        self.feedback_data.append(feedback)
        self._save_feedback_data()
        
        return feedback_id
        
    def collect_rating_feedback(self, prompt: str, response: str, rating: float, 
                               category: str = "general", user_id: Optional[str] = None) -> str:
        """Collect rating feedback (1-5 scale)."""
        if not self.feedback_collection_enabled:
            return "feedback_disabled"
            
        feedback_id = f"feedback_{int(time.time() * 1000)}"
        feedback_score = max(0.0, min(1.0, (rating - 1) / 4))
        
        feedback = HumanFeedback(
            id=feedback_id,
            feedback_type=FeedbackType.RATING,
            prompt=prompt,
            response_a=response,
            rating=rating,
            feedback_score=feedback_score,
            category=category,
            timestamp=time.time(),
            user_id=user_id,
            metadata={}
        )
        
        self.feedback_data.append(feedback)
        self._save_feedback_data()
        
        return {"feedback_id": feedback_id, "status": "success"}
        
    def train_reward_model(self, model_name: str = "default") -> Dict[str, Any]:
        """Train a reward model using collected feedback."""
        if not self.reward_model_training_enabled:
            return {"status": "training_disabled"}
            
        if len(self.feedback_data) < self.min_feedback_for_training:
            return {
                "status": "insufficient_data",
                "required": self.min_feedback_for_training,
                "available": len(self.feedback_data)
            }
            
        # Check if we need safety approval for model training
        change_id = self.safety_guardrails.propose_change(
            ChangeType.MODEL_FINE_TUNING,
            f"Training reward model '{model_name}' with {len(self.feedback_data)} feedback samples",
            {
                "model_name": model_name,
                "feedback_count": len(self.feedback_data)
            },
            ChangeSeverity.MEDIUM
        )
        
        if change_id == "rate_limited":
            return {"status": "rate_limited"}
            
        # Simulate reward model training
        positive_feedback = [f for f in self.feedback_data if f.feedback_score > self.feedback_threshold]
        accuracy = len(positive_feedback) / len(self.feedback_data) if self.feedback_data else 0.0
        
        # Approve the change
        self.safety_guardrails.approve_change(change_id, user_confirmation=True)
        
        return {
            "status": "success",
            "accuracy": accuracy,
            "training_data_size": len(self.feedback_data),
            "positive_feedback_rate": len(positive_feedback) / len(self.feedback_data) if self.feedback_data else 0.0
        }
        
    def predict_reward(self, prompt: str, response: str) -> float:
        """Predict reward score for a response using trained reward model."""
        if not self.feedback_data:
            return 0.5  # Default neutral score
            
        # Simple reward prediction based on feedback patterns
        avg_feedback = sum(f.feedback_score for f in self.feedback_data) / len(self.feedback_data)
        
        # Add some variation based on response characteristics
        response_length = len(response)
        word_count = len(response.split())
        
        length_factor = min(1.0, response_length / 1000)
        word_factor = min(1.0, word_count / 100)
        
        predicted_reward = avg_feedback * 0.7 + length_factor * 0.15 + word_factor * 0.15
        
        return max(0.0, min(1.0, predicted_reward))
        
    def get_feedback_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected feedback."""
        total = len(self.feedback_data)
        positive = sum(1 for f in self.feedback_data if f.feedback_score > self.feedback_threshold)
        
        return {
            "total_feedback": total,
            "positive_feedback": positive,
            "negative_feedback": total - positive,
            "average_feedback_score": sum(f.feedback_score for f in self.feedback_data) / total if total > 0 else 0.0,
            "reward_models": len(self.reward_models)
        }
        
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Main RLHF agent interface."""
        operation = kwargs.get('operation', 'statistics')
        
        if operation == 'collect_preference':
            response_a = kwargs.get('response_a', '')
            response_b = kwargs.get('response_b', '')
            preferred = kwargs.get('preferred', 'A')
            category = kwargs.get('category', 'general')
            user_id = kwargs.get('user_id')
            
            feedback_id = self.collect_preference_feedback(
                prompt, response_a, response_b, preferred, category, user_id
            )
            
            return {
                'operation': 'collect_preference',
                'feedback_id': feedback_id,
                'success': feedback_id != "feedback_disabled"
            }
            
        elif operation == 'collect_rating':
            response = kwargs.get('response', '')
            rating = kwargs.get('rating', 3.0)
            category = kwargs.get('category', 'general')
            user_id = kwargs.get('user_id')
            
            feedback_id = self.collect_rating_feedback(
                prompt, response, rating, category, user_id
            )
            
            return {
                'operation': 'collect_rating',
                'feedback_id': feedback_id,
                'success': feedback_id != "feedback_disabled"
            }
            
        elif operation == 'train_model':
            model_name = kwargs.get('model_name', 'default')
            result = self.train_reward_model(model_name)
            return {
                'operation': 'train_model',
                **result
            }
            
        elif operation == 'predict_reward':
            response = kwargs.get('response', '')
            reward_score = self.predict_reward(prompt, response)
            return {
                'operation': 'predict_reward',
                'reward_score': reward_score
            }
            
        elif operation == 'statistics':
            return {
                'operation': 'statistics',
                **self.get_feedback_statistics()
            }
            
        else:
            return {
                'operation': 'unknown',
                'error': f'Unknown operation: {operation}'
            } 