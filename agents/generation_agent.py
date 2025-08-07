# File: quark/agents/generation_agent.py
#!/usr/bin/env python3
"""
GenerationAgent: handles creative writing and report generation.
"""
from transformers import pipeline
from .base import Agent

class GenerationAgent(Agent):
    def load_model(self):
        return pipeline(
            "text-generation", model=self.model_name,
            return_full_text=False
        )

    def generate(self, prompt: str, **kwargs):
        self._ensure_model()
        outputs = self.model(prompt, max_length=200)
        first = outputs[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
        return first

