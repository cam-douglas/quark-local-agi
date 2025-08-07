#!/usr/bin/env python3
from core.use_cases_tasks import PILLARS
from agents.intent_classifier import IntentClassifier

class Router:
    def __init__(self):
        self.intent_classifier = IntentClassifier()

    def route(self, prompt: str) -> str:
        # Use the intent classifier to determine the category
        task_name = self.intent_classifier.classify(prompt)
        
        # Map task name back to pillar name
        pillar_name = self._get_pillar_for_task(task_name)
        
        return pillar_name
    
    def classify_intent(self, prompt: str) -> dict:
        """Classify the intent of a prompt and return category and confidence."""
        try:
            # Use the intent classifier to determine the category
            task_name = self.intent_classifier.classify(prompt)
            
            # Map task name back to pillar name
            category = self._get_pillar_for_task(task_name)
            
            # For now, return a default confidence of 0.8
            # In a real implementation, this would come from the classifier
            confidence = 0.8
            
            return {
                "category": category,
                "confidence": confidence,
                "task_name": task_name
            }
        except Exception as e:
            # Return a fallback classification
            return {
                "category": "Natural Language Understanding",
                "confidence": 0.5,
                "task_name": "general_query"
            }
    
    def _get_pillar_for_task(self, task_name: str) -> str:
        """Map a task name back to its pillar name."""
        for pillar_name, tasks in PILLARS.items():
            if task_name in tasks:
                return pillar_name
        
        # If no match found, return the task name as fallback
        return task_name

