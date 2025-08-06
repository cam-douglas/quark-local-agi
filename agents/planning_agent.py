#!/usr/bin/env python3
from transformers import pipeline
from agents.base import Agent

class PlanningAgent(Agent):
    def load_model(self):
        return pipeline("text2text-generation", model=self.model_name)

    def generate(self, prompt: str, **kwargs):
        self._ensure_model()
        # Generate a plan for the prompt
        planning_prompt = f"Create a step-by-step plan for: {prompt}"
        return self.model(planning_prompt, max_length=200)[0]["generated_text"]

