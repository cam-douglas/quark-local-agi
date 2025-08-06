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
    
    def _get_pillar_for_task(self, task_name: str) -> str:
        """Map a task name back to its pillar name."""
        for pillar_name, tasks in PILLARS.items():
            if task_name in tasks:
                return pillar_name
        
        # If no match found, return the task name as fallback
        return task_name

